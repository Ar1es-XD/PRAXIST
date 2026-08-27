from __future__ import annotations

import time
import unittest

from praxist.plugins.agent_runtimes.claude_sdk.liveness import ClaudeSessionLiveness


class ClaudeSessionLivenessTest(unittest.TestCase):
    def test_foreground_work_owns_state_and_heartbeats(self) -> None:
        state = ClaudeSessionLiveness()
        initial = state.observation()
        self.assertEqual(initial["session_state"], "model_waiting")
        self.assertEqual(initial["active_foreground_tools"], ())
        self.assertEqual(initial["active_background_tasks"], 0)

        idle_progress = state.tool_progress_at
        state.record_active_work_heartbeat()
        self.assertEqual(state.tool_progress_at, idle_progress)

        state.start_foreground_tool(" Bash\ncommand ", "tool-1")
        state.start_foreground_tool("ignored duplicate", "tool-1")
        state.start_foreground_tool("TaskOutput")
        time.sleep(0.001)
        before_heartbeat = state.tool_progress_at
        state.record_active_work_heartbeat()
        active = state.observation()
        self.assertEqual(active["session_state"], "foreground_tool_running")
        self.assertEqual(active["active_foreground_tools"], ("Bash command", "TaskOutput"))
        self.assertGreater(state.tool_progress_at, before_heartbeat)
        self.assertTrue(state.foreground_work_active())

        state.finish_foreground_tool("unknown-id")
        self.assertTrue(state.foreground_work_active())
        state.finish_foreground_tool("tool-1")
        self.assertFalse(state.foreground_work_active())

        state.start_foreground_tool("wait_for_file", "only-id")
        state.finish_foreground_tool("mismatched-id")
        self.assertFalse(state.foreground_work_active())
        self.assertEqual(state.observation()["session_state"], "model_waiting")

    def test_background_closing_and_terminal_states_are_coherent(self) -> None:
        state = ClaudeSessionLiveness()
        state.record_background_status("task-1", "running")
        state.record_background_status("task-1", "running")
        before_heartbeat = state.tool_progress_at
        time.sleep(0.001)
        state.record_active_work_heartbeat()
        active = state.observation()
        self.assertEqual(active["session_state"], "background_work_running")
        self.assertEqual(active["active_background_tasks"], 1)
        self.assertGreater(state.tool_progress_at, before_heartbeat)
        self.assertFalse(state.all_background_terminal())
        self.assertFalse(state.any_background_failed())

        state.record_background_status("task-1", "completed")
        self.assertTrue(state.all_background_terminal())
        self.assertEqual(state.observation()["session_state"], "model_waiting")

        state.record_background_status("task-2", "failed")
        self.assertTrue(state.all_background_terminal())
        self.assertTrue(state.any_background_failed())
        state.mark_closing()
        self.assertEqual(state.observation()["session_state"], "closing")
        state.mark_terminal()
        terminal = state.observation()
        self.assertEqual(terminal["session_state"], "idle")
        self.assertGreaterEqual(
            terminal["latest_progress_at"], terminal["sdk_complete_message_progress_at"]
        )

    def test_complete_and_partial_message_clocks_advance_independently(self) -> None:
        state = ClaudeSessionLiveness()
        complete_before = state.sdk_complete_message_progress_at
        partial_before = state.model_stream_progress_at
        time.sleep(0.001)
        state.record_complete_message()
        complete_after = state.sdk_complete_message_progress_at
        self.assertGreater(complete_after, complete_before)
        self.assertEqual(state.model_stream_progress_at, partial_before)

        time.sleep(0.001)
        state.record_model_stream_progress()
        self.assertGreater(state.model_stream_progress_at, partial_before)
        self.assertEqual(state.sdk_complete_message_progress_at, complete_after)


if __name__ == "__main__":
    unittest.main()
