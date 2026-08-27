"""
Sharpness probes — compute loss landscape sharpness proxies.

Provides:
- Top Hessian eigenvalue estimation (power iteration)
- Average sharpness within a neighborhood
"""

import argparse
import importlib.util
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict


def _load_baseline_helpers():
    baseline_path = Path(__file__).resolve().parent.parent / "baseline" / "train.py"
    spec = importlib.util.spec_from_file_location("_sam_baseline_train", baseline_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import baseline harness from {baseline_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_resnet18, module.get_dataset


def _resolve_data_dir(value: str | None) -> str:
    resolver_path = Path(__file__).resolve().parents[2] / "dataset_metadata" / "resolve_dataset.py"
    if not resolver_path.exists():
        return str(value or "./data")
    spec = importlib.util.spec_from_file_location("_sam_dataset_resolver", resolver_path)
    if spec is None or spec.loader is None:
        return str(value or "./data")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module.resolve_dataset_root(value))


def hessian_top_eigenvalue(
    model: nn.Module,
    criterion: nn.Module,
    data_loader,
    device: torch.device,
    n_iterations: int = 20,
) -> float:
    """Estimate the top eigenvalue of the Hessian using power iteration.

    This is a proxy for sharpness: flatter minima have smaller top eigenvalues.
    """
    model.eval()

    # Get a single batch for Hessian estimation
    inputs, targets = next(iter(data_loader))
    inputs, targets = inputs.to(device), targets.to(device)

    # Initialize random vector
    params = [p for p in model.parameters() if p.requires_grad]
    v = [torch.randn_like(p) for p in params]

    # Normalize
    v_norm = torch.sqrt(sum(torch.sum(vi ** 2) for vi in v))
    v = [vi / v_norm for vi in v]

    eigenvalue = 0.0

    for _ in range(n_iterations):
        # Compute Hv (Hessian-vector product)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        grads = torch.autograd.grad(loss, params, create_graph=True)

        Hv = torch.autograd.grad(
            grads, params,
            grad_outputs=v,
            retain_graph=False,
        )

        # Rayleigh quotient
        eigenvalue = sum(
            torch.sum(hvi * vi) for hvi, vi in zip(Hv, v)
        ).item()

        # Update v = Hv / ||Hv||
        v = [hvi.detach() for hvi in Hv]
        v_norm = torch.sqrt(sum(torch.sum(vi ** 2) for vi in v))
        if v_norm > 0:
            v = [vi / v_norm for vi in v]

    return eigenvalue


def compute_sharpness_metrics(
    model: nn.Module,
    criterion: nn.Module,
    data_loader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute a suite of sharpness metrics."""
    top_eigen = hessian_top_eigenvalue(model, criterion, data_loader, device)

    return {
        "sharpness_top_eigen": top_eigen,
    }


def main():
    """Run the sharpness-probe CLI for a saved checkpoint."""

    parser = argparse.ArgumentParser(description="Compute sharpness probes")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--output", default="sharpness_results.json")
    args = parser.parse_args()
    args.data_dir = _resolve_data_dir(args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    get_resnet18, get_dataset = _load_baseline_helpers()

    _, test_set, num_classes, input_size = get_dataset(args.dataset, args.data_dir)
    model = get_resnet18(num_classes=num_classes, input_size=input_size).to(device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    metrics = compute_sharpness_metrics(model, criterion, test_loader, device)

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Sharpness metrics: {metrics}")


if __name__ == "__main__":
    main()
