"""Backward-compat shim: re-exports the terminal-reward module.

The DIG C01 contract names `terminal_reward.py` as the mechanism module.
This shim keeps the variant-local harness import (`import reward_shaping
as _reward`) unchanged while the real reward logic lives in terminal_reward.py.
"""
from terminal_reward import *  # noqa: F401,F403
