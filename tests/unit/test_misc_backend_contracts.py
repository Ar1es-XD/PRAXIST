from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class MiscBackendContractsTest(unittest.TestCase):
    def test_http_utils_event_wait_schedule_and_prompt_artifacts(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            event_wait,
            prompt_artifacts,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            http_utils,
            schedule,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
                http_utils.get_server_url()
            with patch.dict(os.environ, {"SERVER_URL": "http://server/"}, clear=True):
                self.assertEqual(http_utils.get_server_url(), "http://server")

            self.assertEqual(http_utils.validate_safe_identifier(" good-id_1 ", "id"), "good-id_1")
            for bad in ("", "../x", "bad/slash", "bad space"):
                with self.assertRaises(ValueError):
                    http_utils.validate_safe_identifier(bad, "id")
            self.assertEqual(
                http_utils.validate_safe_path(str(root / "ok"), "path", allowed_base=str(root)),
                str((root / "ok").resolve()),
            )
            with self.assertRaises(ValueError):
                http_utils.validate_safe_path(
                    str(root.parent / "escape"), "path", allowed_base=str(root)
                )

            class FakeResponse:
                def __init__(self, status_code: int, payload: dict) -> None:
                    self.status_code = status_code
                    self._payload = payload

                def json(self) -> dict:
                    return self._payload

                def raise_for_status(self) -> None:
                    if self.status_code >= 400:
                        raise RuntimeError("http")

            class FakeAsyncClient:
                def __init__(self, timeout: int) -> None:
                    self.timeout = timeout

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *_args):
                    return None

                async def post(self, *_args, **_kwargs):
                    return FakeResponse(400, {"error": "bad"})

                async def get(self, *_args, **_kwargs):
                    return FakeResponse(200, {"ok": True})

            with (
                patch.object(http_utils, "HAS_HTTPX", True),
                patch.object(
                    http_utils, "httpx", SimpleNamespace(AsyncClient=FakeAsyncClient), create=True
                ),
            ):
                self.assertEqual(
                    asyncio.run(http_utils.async_http_post("http://x", {"a": 1})),
                    {"error": "bad"},
                )
                self.assertEqual(
                    asyncio.run(http_utils.async_http_get("http://x", params={"q": 1})),
                    {"ok": True},
                )
            with (
                patch.object(http_utils, "HAS_HTTPX", False),
                patch.object(http_utils, "HAS_REQUESTS", True),
                patch.object(
                    http_utils,
                    "requests",
                    SimpleNamespace(
                        post=lambda *a, **k: FakeResponse(200, {"posted": True}),
                        get=lambda *a, **k: FakeResponse(200, {"got": True}),
                    ),
                    create=True,
                ),
            ):
                self.assertEqual(
                    asyncio.run(http_utils.async_http_post("http://x", {})), {"posted": True}
                )
                self.assertEqual(asyncio.run(http_utils.async_http_get("http://x")), {"got": True})
            with (
                patch.object(http_utils, "HAS_HTTPX", False),
                patch.object(http_utils, "HAS_REQUESTS", False),
                self.assertRaises(ImportError),
            ):
                asyncio.run(http_utils.async_http_get("http://x"))

            existing_dir = root / "watch"
            existing_dir.mkdir()
            child_file = existing_dir / "child.txt"
            child_file.write_text("x", encoding="utf-8")
            roots = event_wait._candidate_watch_roots(
                [child_file, existing_dir, root / "missing" / "x"]
            )
            self.assertEqual(roots[0], existing_dir.resolve())

            async def fast_sleep(_seconds: float):
                return None

            with patch.object(event_wait.asyncio, "sleep", fast_sleep):
                self.assertFalse(
                    asyncio.run(
                        event_wait._sleep_with_stop_checks(
                            0,
                            stop_check=None,
                            stop_check_interval_seconds=1,
                        )
                    )
                )
                self.assertTrue(
                    asyncio.run(
                        event_wait._sleep_with_stop_checks(
                            10,
                            stop_check=lambda: True,
                            stop_check_interval_seconds=1,
                        )
                    )
                )
                self.assertEqual(
                    asyncio.run(
                        event_wait.wait_for_filesystem_event(
                            [root / "does-not-exist" / "file"],
                            timeout_seconds=0,
                        )
                    ).reason,
                    "no_watch_paths",
                )

            class FailingWaiter:
                def __init__(self, *_args, **_kwargs) -> None:
                    raise RuntimeError("no inotify")

            with (
                patch.object(event_wait, "_InotifyWaiter", FailingWaiter),
                patch.object(event_wait.asyncio, "sleep", fast_sleep),
            ):
                self.assertEqual(
                    asyncio.run(
                        event_wait.wait_for_filesystem_event(
                            [existing_dir],
                            timeout_seconds=0,
                            stop_check=lambda: False,
                        )
                    ).reason,
                    "fallback_elapsed",
                )

            self.assertEqual(schedule.epoch_fraction(-1, 10), 0.0)
            self.assertEqual(schedule.epoch_fraction(20, 10), 1.0)
            self.assertEqual(schedule.epoch_fraction(20, 10, clamp=False), 2.0)
            self.assertAlmostEqual(schedule.linear_schedule(0.5, start=0, end=10), 5.0)
            self.assertAlmostEqual(schedule.cosine_schedule(0.0, start=1, end=3), 1.0)
            self.assertAlmostEqual(schedule.peaked_schedule(0.0, start=1, peak=3, peak_at=0), 3.0)
            self.assertAlmostEqual(schedule.peaked_schedule(1.0, start=1, peak=3, peak_at=1), 3.0)
            self.assertAlmostEqual(
                schedule.warmup_then_schedule(
                    0.05,
                    warmup_fraction=0.1,
                    warmup_start=0,
                    base_start=1,
                    base_end=0,
                ),
                0.5,
            )
            self.assertEqual(
                schedule.warmup_then_schedule(
                    0.5,
                    warmup_fraction=0,
                    warmup_start=0,
                    base_start=1,
                    base_end=0,
                    base_kind="linear",
                ),
                0.5,
            )
            hits = schedule.scan_for_step_anti_pattern(
                '''
                """total_steps in docs"""
                # max_steps comment
                value = total_steps + num_training_steps
                '''
            )
            self.assertEqual(hits[0][2], "total_steps")

            artifact = {
                "artifact_id": "a",
                "artifact_type": "t",
                "payload_path": "p",
                "secret": "drop",
            }
            self.assertNotIn("secret", prompt_artifacts.compact_artifact_ref(artifact))
            manifest_path = root / "layout.json"
            prompt_path = root / "prompts" / "p.md"
            prompt_path.parent.mkdir()
            with patch.dict(os.environ, {"PRAXIST_RUN_ID": "run"}, clear=False):
                out = prompt_artifacts.persist_prompt_layout_artifacts(
                    run_dir=root,
                    prompt_text="hello",
                    prompt_path=prompt_path,
                    manifest={"schema_version": "x"},
                    manifest_path=manifest_path,
                    peer_id="gen0_peer0",
                    gen_id=0,
                )
            self.assertIn("rendered_prompt_ref", out)
            self.assertTrue(manifest_path.exists())
            with patch(
                "praxist.core.storage.ArtifactWriter.persist_text",
                side_effect=RuntimeError("disk"),
            ):
                self.assertEqual(
                    prompt_artifacts.persist_prompt_layout_artifacts(
                        run_dir=root,
                        prompt_text="hello",
                        prompt_path=prompt_path,
                        manifest={"schema_version": "x"},
                        manifest_path=manifest_path,
                        peer_id="gen0_peer0",
                        gen_id=0,
                    ),
                    {"schema_version": "x"},
                )


if __name__ == "__main__":
    unittest.main()
