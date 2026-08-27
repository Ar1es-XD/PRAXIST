"""
SAM optimizer research — baseline training script.

Supports: sgd, adam, sam, asam, and custom optimizer classes.

Usage:
    python train.py --optimizer sam --dataset cifar100 --epochs 200 --seed 42
    python train.py --optimizer custom --variant-path path/to/optimizer.py --dataset cifar100
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def get_resnet18(num_classes: int = 100, input_size: int = 32):
    """Get ResNet-18 adapted to input image size.

    For 32x32 (CIFAR): replace 7x7 stride-2 conv with 3x3 stride-1, drop maxpool.
    For 64x64 (Tiny-ImageNet): keep stride-1 first conv but retain maxpool to
        preserve enough spatial reduction (final featuremap stays 4x4).
    For larger inputs: use the stock ResNet-18 stem.
    """
    from torchvision.models import resnet18
    model = resnet18(num_classes=num_classes)
    if input_size <= 32:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif input_size <= 64:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # keep maxpool to halve spatial dim
    return model


# ---------------------------------------------------------------------------
# SAM Optimizer
# ---------------------------------------------------------------------------

class SAM(optim.Optimizer):
    """Sharpness-Aware Minimization optimizer."""

    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        defaults = dict(rho=rho)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()

    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def step(self, closure=None):
        raise NotImplementedError("Use first_step() and second_step()")

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)


class ASAM(SAM):
    """Adaptive Sharpness-Aware Minimization (Kwon et al., 2021).

    Scales the perturbation elementwise by |w|, so the effective neighborhood
    adapts to each parameter's magnitude:

        e_w = rho * (w^2 * g) / || |w| * g ||_2

    Note: ASAM is typically run with a larger rho than SAM (e.g. 0.5-2.0).
    """

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._adaptive_grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (p * p) * p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    def _adaptive_grad_norm(self):
        norm = torch.norm(
            torch.stack([
                (torch.abs(p) * p.grad).norm(p=2)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def get_dataset(name: str, data_dir: str = "./data"):
    """Get CIFAR-10 / CIFAR-100 / Tiny-ImageNet with standard augmentation.

    Returns (train_set, test_set, num_classes, input_size).
    """
    if name in ("cifar10", "cifar100"):
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        Dataset = torchvision.datasets.CIFAR100 if name == "cifar100" else torchvision.datasets.CIFAR10
        num_classes = 100 if name == "cifar100" else 10
        # download=False: cache is expected to be pre-fetched at <data_dir>/<cifar-N-…>.
        # Multi-process runners would otherwise race on the same partial tar download.
        train_set = Dataset(data_dir, train=True, download=False, transform=train_transform)
        test_set = Dataset(data_dir, train=False, download=False, transform=test_transform)
        return train_set, test_set, num_classes, 32

    if name == "tiny-imagenet":
        # Tiny-ImageNet-200: 200 classes, 64x64, 100k train / 10k val.
        # Expects layout under <data_dir>/tiny-imagenet-200/{train,val}/<class>/*.JPEG
        # (val/ must be pre-reorganized into class folders — see download script.)
        root = os.path.join(data_dir, "tiny-imagenet-200")
        train_root = os.path.join(root, "train")
        val_root = os.path.join(root, "val")
        if not os.path.isdir(train_root) or not os.path.isdir(val_root):
            raise FileNotFoundError(
                f"Tiny-ImageNet not found at {root}. "
                f"Expected layout: {root}/train/<class>/images/*.JPEG and {root}/val/<class>/*.JPEG"
            )
        normalize = transforms.Normalize(
            mean=[0.4802, 0.4481, 0.3975],
            std=[0.2770, 0.2691, 0.2821],
        )
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        # Train images live under train/<class>/images/*.JPEG (an extra "images" level).
        # Walk into that automatically by using a dataset that flattens it.
        from torchvision.datasets import ImageFolder

        class TinyImageNetTrain(ImageFolder):
            """ImageFolder over train/<class>/images/*.JPEG (flattens the inner 'images' dir)."""
            def find_classes(self, directory):
                classes = sorted(d.name for d in os.scandir(directory) if d.is_dir())
                class_to_idx = {c: i for i, c in enumerate(classes)}
                return classes, class_to_idx
            def make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None, allow_empty=False):
                samples = []
                for cls, idx in class_to_idx.items():
                    img_dir = os.path.join(directory, cls, "images")
                    if not os.path.isdir(img_dir):
                        img_dir = os.path.join(directory, cls)
                    for fname in sorted(os.listdir(img_dir)):
                        if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                            samples.append((os.path.join(img_dir, fname), idx))
                return samples

        train_set = TinyImageNetTrain(train_root, transform=train_transform)
        test_set = torchvision.datasets.ImageFolder(val_root, transform=test_transform)
        return train_set, test_set, 200, 64

    raise ValueError(f"Unknown dataset: {name}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, use_sam=False, bf16=False):
    """Run one supervised training epoch and return mean loss and accuracy."""

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (bf16 and device.type == "cuda") else _NullCtx()

    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        if use_sam:
            with amp_ctx:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.first_step()
            optimizer.zero_grad()

            with amp_ctx:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.second_step()
            optimizer.zero_grad()
        else:
            optimizer.zero_grad()
            with amp_ctx:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, correct / total


class _NullCtx:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


@torch.no_grad()
def evaluate(model, loader, criterion, device, bf16=False):
    """Evaluate a model on one loader and return mean loss and accuracy."""

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (bf16 and device.type == "cuda") else _NullCtx()

    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with amp_ctx:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, correct / total


def main():
    """Run the standalone baseline training CLI."""

    parser = argparse.ArgumentParser(description="SAM baseline training")
    parser.add_argument("--optimizer", default="sam", choices=["sgd", "adam", "sam", "asam", "custom"])
    parser.add_argument("--variant-path", default="", help="Path to custom optimizer .py file")
    parser.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100", "tiny-imagenet"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--rho", type=float, default=0.05, help="SAM perturbation radius")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("TRAIN_NUM_WORKERS", "4")))
    args = parser.parse_args()

    # Refuse to bypass the governor when running under Praxist.
    # If GPU_GOVERNOR_DIR is set, an orchestrator is coordinating GPUs;
    # this script doesn't acquire a slot, so direct invocation would
    # silently steal GPU 0 from the legitimate slot-holder. The evaluator
    # at evaluations/pareto_tiered/run.py is the right entrypoint for peers.
    if os.environ.get("GPU_GOVERNOR_DIR") and not os.environ.get("ALLOW_DIRECT_TRAIN"):
        print(
            "ERROR: train.py invoked directly while GPU_GOVERNOR_DIR is set.\n"
            "       This bypasses the per-GPU process governor and will\n"
            "       contend for cuda:0 with other peers.\n"
            "       Use evaluations/pareto_tiered/run.py instead (it integrates\n"
            "       the governor and supports the T1/T2/T3 protocol).\n"
            "       To force direct invocation anyway (NOT recommended),\n"
            "       set ALLOW_DIRECT_TRAIN=1.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # cudnn benchmark for fixed input shapes (we don't need full determinism here —
    # the 5-seed pool gives us statistical significance, and speed matters for the
    # <40min/variant budget).
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Data
    train_set, test_set, num_classes, input_size = get_dataset(args.dataset, args.data_dir)
    _nw = args.num_workers
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=_nw, pin_memory=(_nw > 0), persistent_workers=(_nw > 0),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=max(args.batch_size, 256), shuffle=False,
        num_workers=_nw, pin_memory=(_nw > 0), persistent_workers=(_nw > 0),
    )

    # Model
    model = get_resnet18(num_classes=num_classes, input_size=input_size).to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    use_sam = False
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.01, weight_decay=args.weight_decay)
    elif args.optimizer == "sam":
        optimizer = SAM(model.parameters(), optim.SGD, rho=args.rho, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        use_sam = True
    elif args.optimizer == "asam":
        optimizer = ASAM(model.parameters(), optim.SGD, rho=args.rho, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        use_sam = True
    elif args.optimizer == "custom":
        if not args.variant_path:
            print("Error: --variant-path required for custom optimizer")
            sys.exit(1)
        # Dynamic import of custom optimizer
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_opt", args.variant_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        optimizer = mod.create_optimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, rho=args.rho)
        use_sam = getattr(mod, "USE_SAM_STEPS", False)

    # LR scheduler (cosine)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer if not use_sam else optimizer.base_optimizer, T_max=args.epochs)

    # Training
    results = {"args": vars(args), "epochs": []}
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, use_sam, bf16=args.bf16)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, bf16=args.bf16)
        scheduler.step()

        results["epochs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": scheduler.get_last_lr()[0],
        })

        if test_acc > best_acc:
            best_acc = test_acc

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, best={best_acc:.4f}")

    total_time = time.time() - start_time

    results["summary"] = {
        "best_test_acc": best_acc,
        "final_test_acc": results["epochs"][-1]["test_acc"],
        "final_train_acc": results["epochs"][-1]["train_acc"],
        "train_test_gap": results["epochs"][-1]["train_acc"] - results["epochs"][-1]["test_acc"],
        "total_time_seconds": total_time,
        "seed": args.seed,
        "optimizer": args.optimizer,
        "dataset": args.dataset,
    }

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / f"{args.optimizer}_{args.dataset}_seed{args.seed}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    # Generate figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_data = results["epochs"]
        ep_nums = [e["epoch"] for e in epochs_data]
        train_losses = [e["train_loss"] for e in epochs_data]
        test_losses = [e["test_loss"] for e in epochs_data]
        train_accs = [e["train_acc"] for e in epochs_data]
        test_accs = [e["test_acc"] for e in epochs_data]
        gaps = [e["train_acc"] - e["test_acc"] for e in epochs_data]

        # ── Figure 1: Training curves (loss + accuracy) ──────────────
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(ep_nums, train_losses, label="Train Loss", linewidth=1.5)
        ax1.plot(ep_nums, test_losses, label="Test Loss", linewidth=1.5)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"{args.optimizer} — Loss ({args.dataset}, seed={args.seed})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(ep_nums, train_accs, label="Train Acc", linewidth=1.5)
        ax2.plot(ep_nums, test_accs, label="Test Acc", linewidth=1.5)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title(f"{args.optimizer} — Accuracy ({args.dataset}, seed={args.seed})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig1.tight_layout()
        fig1_file = output_dir / f"{args.optimizer}_{args.dataset}_seed{args.seed}_curves.png"
        fig1.savefig(fig1_file, dpi=150)
        plt.close(fig1)
        print(f"Training curves saved to {fig1_file}")

        # ── Figure 2: Task metrics (test acc, gap, generalization) ───
        summary = results["summary"]
        fig2, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Test accuracy trajectory with best marked
        axes[0].plot(ep_nums, test_accs, color="#2196F3", linewidth=1.5)
        best_epoch = max(range(len(test_accs)), key=lambda i: test_accs[i])
        axes[0].scatter([ep_nums[best_epoch]], [test_accs[best_epoch]],
                        color="red", s=80, zorder=5, label=f"Best: {best_acc:.4f} (ep {best_epoch})")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Test Accuracy")
        axes[0].set_title(f"Test Accuracy — {args.optimizer}")
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Panel 2: Generalization gap (train - test)
        axes[1].plot(ep_nums, gaps, color="#FF5722", linewidth=1.5)
        axes[1].axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        axes[1].fill_between(ep_nums, 0, gaps, alpha=0.15, color="#FF5722")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Train Acc − Test Acc")
        axes[1].set_title(f"Generalization Gap — {args.optimizer}")
        axes[1].annotate(f"Final gap: {summary['train_test_gap']:.4f}",
                         xy=(0.98, 0.95), xycoords="axes fraction",
                         ha="right", va="top", fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
        axes[1].grid(True, alpha=0.3)

        # Panel 3: Summary metrics bar chart
        metric_names = ["Best Test Acc", "Final Test Acc", "Final Train Acc"]
        metric_vals = [summary["best_test_acc"], summary["final_test_acc"], summary["final_train_acc"]]
        colors = ["#4CAF50", "#2196F3", "#FF9800"]
        bars = axes[2].bar(metric_names, metric_vals, color=colors, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, metric_vals):
            axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                         f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        axes[2].set_ylabel("Accuracy")
        axes[2].set_title(f"Key Metrics — {args.optimizer}")
        axes[2].set_ylim(0, min(1.0, max(metric_vals) * 1.15))
        axes[2].grid(True, axis="y", alpha=0.3)

        fig2.suptitle(f"{args.optimizer} | {args.dataset} | seed={args.seed} | "
                      f"time={summary['total_time_seconds']:.0f}s", fontsize=11, y=1.01)
        fig2.tight_layout()
        fig2_file = output_dir / f"{args.optimizer}_{args.dataset}_seed{args.seed}_metrics.png"
        fig2.savefig(fig2_file, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Task metrics figure saved to {fig2_file}")

    except ImportError:
        print("matplotlib not available, skipping figure generation")
    except Exception as e:
        print(f"Warning: figure generation failed: {e}")

    print(f"\nDone. Best test accuracy: {best_acc:.4f}")
    print(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
