from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.plugins.workflow_stages.research_loop.backend.experiment_scheduler import (
    ExperimentSchedulerService,
)
from praxist.plugins.workflow_stages.research_loop.backend.experiment_scheduler_client import (
    recover_environment,
    sensitive_environment_matches,
    submit_and_wait,
)
from tests.unit.test_central_experiment_scheduler import (
    _BindingMissingAllocator,
    _BlockedAllocator,
    _GPUAllocator,
    _settings,
)


class ExperimentRecoveryEdgeTest(unittest.TestCase):
    def test_admission_timeout_removes_queued_job_and_allows_retry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_BlockedAllocator(""),
            )
            service.start()
            try:
                with self.assertRaises(TimeoutError):
                    submit_and_wait(
                        [sys.executable, "-c", "pass"],
                        peer_id="gen0_peer0",
                        experiment_id="queue-timeout",
                        run_dir=run_dir,
                        wait_timeout_seconds=0.1,
                    )
                status = service.status()
                service.allocator = _GPUAllocator("")
                retry_code = submit_and_wait(
                    [sys.executable, "-c", "pass"],
                    peer_id="gen0_peer0",
                    experiment_id="queue-timeout",
                    run_dir=run_dir,
                )
            finally:
                service.stop()
            self.assertEqual(status["queued"], 0)
            self.assertEqual(status["rejected"], 1)
            self.assertEqual(retry_code, 0)

    def test_expired_pending_reservation_never_releases_launch_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _BindingMissingAllocator("")
            settings = _settings(maximum=1)
            settings.infrastructure_retries = 0
            marker = Path(td) / "started"
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=settings, allocator=allocator
            )
            job = service.submit(
                {
                    "command": [
                        sys.executable,
                        "-c",
                        "import pathlib,sys;pathlib.Path(sys.argv[1]).write_text('bad')",
                        str(marker),
                    ],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "binding-row-missing",
                }
            )
            self.assertTrue(service._launch(job))
            self.assertFalse(marker.exists())
            self.assertEqual(job.state, "failed")

    def test_retry_environment_preserves_legitimate_key_names_and_unsets(self) -> None:
        with (
            tempfile.TemporaryDirectory() as td,
            patch.dict(
                os.environ,
                {"KEYFRAME_INTERVAL": "host", "REMOVE_FROM_RETRY": "host"},
                clear=False,
            ),
        ):
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=_settings(maximum=1)
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "environment-roundtrip",
                    "environment": {
                        **{
                            key: value
                            for key, value in os.environ.items()
                            if key != "REMOVE_FROM_RETRY"
                        },
                        "KEYFRAME_INTERVAL": "task",
                        "OPENAI_API_KEY": "must-not-persist",
                    },
                }
            )
            payload = service._event_identity(job, include_request=True)
            with patch.dict(
                os.environ,
                {"NEW_CONTROLLER_ONLY": "new", "OPENAI_API_KEY": "different"},
                clear=False,
            ):
                recovered = recover_environment(payload)
            self.assertEqual(recovered["KEYFRAME_INTERVAL"], "task")
            self.assertNotIn("REMOVE_FROM_RETRY", recovered)
            self.assertNotIn("NEW_CONTROLLER_ONLY", recovered)
            self.assertNotIn("OPENAI_API_KEY", recovered)
            self.assertFalse(sensitive_environment_matches(payload))
            self.assertNotIn("OPENAI_API_KEY", payload["environment_values"])
            self.assertIn("OPENAI_API_KEY", payload["environment_sensitive_hashes"])

    def test_recovery_before_popen_releases_pending_reservation_before_requeue(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _GPUAllocator("")
            allocator.active.add("pending-allocation")
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=allocator,
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "intent-only",
                    "environment": {},
                }
            )
            service._queue.remove(job.job_id)
            job.attempts = 1
            attempt_dir = service.state_dir / "attempts" / f"{job.job_id}-a1"
            service._append_event(
                {
                    "event": "launch_intent",
                    **service._event_identity(job, include_request=True),
                    "allocation_id": "pending-allocation",
                    "attempt_dir": str(attempt_dir),
                },
                required=True,
            )
            service._jobs.clear()
            service._semantic_jobs.clear()
            service._recover_terminal_events()
            self.assertIn("pending-allocation", allocator.released)
            self.assertEqual(service._queue, [job.job_id])


if __name__ == "__main__":
    unittest.main()
