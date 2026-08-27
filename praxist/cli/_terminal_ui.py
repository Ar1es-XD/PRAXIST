"""Small dependency-free terminal controls for Praxist operator setup.

The research CLI must also work in pipes and automation, so this module is
strictly opt-in: callers use it only after confirming that input and output are
interactive terminals.  It never stores configuration and never logs input.
"""

from __future__ import annotations

import os
import select
import sys
import textwrap
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import IO

_BACKSPACE = {"\x08", "\x7f"}
_ENTER = {"\r", "\n"}
_MAX_SECRET_LENGTH = 4096
_ANSI_RESET = "\x1b[0m"
_ANSI_BOLD = "\x1b[1m"
_ANSI_DIM = "\x1b[2m"
_ANSI_FOCUS = "\x1b[1;36m"
_ANSI_FOCUS_DETAIL = "\x1b[2;36m"
_ANSI_HIDE_CURSOR = "\x1b[?25l"
_ANSI_SHOW_CURSOR = "\x1b[?25h"
_ANSI_ENTER_ALT_SCREEN = "\x1b[?1049h"
_ANSI_LEAVE_ALT_SCREEN = "\x1b[?1049l"


class TerminalInteractionError(RuntimeError):
    """Raised when an interactive terminal control cannot run safely."""


class TerminalInteractionCancelled(TerminalInteractionError):
    """Raised when the operator presses Escape or closes interactive input."""


@dataclass(frozen=True)
class Choice:
    """One stable value rendered by :func:`select_choice`."""

    value: str
    label: str
    detail: str = ""


def interactive_terminal_available(
    input_stream: IO[str] | None = None,
    output_stream: IO[str] | None = None,
) -> bool:
    """Return whether the streams can support direct key interaction."""

    source = input_stream or sys.stdin
    target = output_stream or sys.stderr
    return (
        os.name == "posix"
        and bool(getattr(source, "isatty", lambda: False)())
        and bool(getattr(target, "isatty", lambda: False)())
        and callable(getattr(source, "fileno", None))
    )


def select_choice(
    prompt: str,
    choices: Sequence[Choice],
    *,
    default: int | None = 0,
    input_stream: IO[str] | None = None,
    output_stream: IO[str] | None = None,
    read_key: Callable[[], str] | None = None,
) -> str:
    """Select one value with arrows or a number, Enter, and Escape.

    ``read_key`` is injectable so the state machine can be tested without a
    pseudo-terminal.  Production calls use a raw POSIX terminal reader.
    """

    if not choices:
        raise ValueError("at least one choice is required")
    if default is not None and (default < 0 or default >= len(choices)):
        raise ValueError("default choice is out of range")

    target = output_stream or sys.stderr
    selected: int | None = default

    def interact(reader: Callable[[], str]) -> str:
        nonlocal selected
        styled = _ansi_styling_available(target)
        rendered_lines = 0
        if styled:
            target.write(_ANSI_HIDE_CURSOR)
            rendered_lines = _render_choice_menu(target, prompt, choices, selected)
        else:
            target.write(f"{prompt}\n")
            for index, choice in enumerate(choices, start=1):
                detail = f" - {choice.detail}" if choice.detail else ""
                target.write(f"  {index}. {choice.label}{detail}\n")
            _render_choice(target, choices, selected)
        try:
            while True:
                key = reader()
                if key in {"UP", "k"}:
                    selected = (
                        len(choices) - 1 if selected is None else (selected - 1) % len(choices)
                    )
                elif key in {"DOWN", "j"}:
                    selected = 0 if selected is None else (selected + 1) % len(choices)
                elif len(key) == 1 and key.isdigit() and key != "0":
                    candidate = int(key) - 1
                    if candidate < len(choices):
                        selected = candidate
                elif key in _ENTER and selected is not None:
                    target.write("\n")
                    target.flush()
                    return choices[selected].value
                elif key in {"ESC", "EOF", "INTERRUPT"}:
                    target.write("\n")
                    target.flush()
                    raise TerminalInteractionCancelled("interactive selection cancelled")
                else:
                    continue
                if styled:
                    rendered_lines = _render_choice_menu(
                        target,
                        prompt,
                        choices,
                        selected,
                        previous_line_count=rendered_lines,
                    )
                else:
                    _render_choice(target, choices, selected)
        finally:
            if styled:
                target.write(_ANSI_SHOW_CURSOR)
                target.flush()

    if read_key is not None:
        return interact(read_key)
    source = input_stream or sys.stdin
    if not interactive_terminal_available(source, target):
        raise TerminalInteractionError("an interactive terminal is required")
    with _terminal_key_reader(source) as reader:
        return interact(reader)


def confirm_action(
    prompt: str,
    *,
    input_stream: IO[str] | None = None,
    output_stream: IO[str] | None = None,
    read_key: Callable[[], str] | None = None,
) -> None:
    """Wait for Enter, or cancel on Escape, without a yes/no prompt."""

    target = output_stream or sys.stderr

    def interact(reader: Callable[[], str]) -> None:
        if _ansi_styling_available(target):
            target.write(
                f"{_ANSI_BOLD}{prompt}{_ANSI_RESET}  "
                f"{_ANSI_DIM}[Enter: continue | Esc: back]{_ANSI_RESET}"
            )
        else:
            target.write(f"{prompt}  [Enter: continue | Esc: back]")
        target.flush()
        while True:
            key = reader()
            if key in _ENTER:
                target.write("\n")
                target.flush()
                return
            if key in {"ESC", "EOF"}:
                target.write("\n")
                target.flush()
                raise TerminalInteractionCancelled("interactive action cancelled")
            if key == "INTERRUPT":
                raise TerminalInteractionCancelled("interactive action cancelled")

    if read_key is not None:
        interact(read_key)
        return
    source = input_stream or sys.stdin
    if not interactive_terminal_available(source, target):
        raise TerminalInteractionError("an interactive terminal is required")
    with _terminal_key_reader(source) as reader:
        interact(reader)


def read_masked_secret(
    prompt: str,
    *,
    input_stream: IO[str] | None = None,
    output_stream: IO[str] | None = None,
    read_key: Callable[[], str] | None = None,
    max_length: int = _MAX_SECRET_LENGTH,
) -> str:
    """Read a secret while echoing one ``*`` for every stored character.

    Paste and backspace work naturally.  Escape sequences are recognized by
    the key reader and never become part of the secret.  Left and right arrows
    are intentionally ignored: the compact editor is append/backspace only.
    """

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    target = output_stream or sys.stderr
    secret: list[str] = []
    overflow = False

    def interact(reader: Callable[[], str]) -> str:
        nonlocal overflow
        target.write(_styled_input_prompt(target, prompt))
        target.flush()
        while True:
            key = reader()
            if key in _ENTER:
                target.write("\n")
                target.flush()
                if overflow:
                    raise TerminalInteractionError(
                        f"secret exceeds the {max_length}-character safety limit"
                    )
                return "".join(secret)
            if key in {"ESC", "EOF"}:
                target.write("\n")
                target.flush()
                raise TerminalInteractionCancelled("secret input cancelled")
            if key == "INTERRUPT":
                raise TerminalInteractionCancelled("secret input cancelled")
            if key in _BACKSPACE:
                if secret:
                    secret.pop()
                    target.write("\b \b")
                    target.flush()
                continue
            if key in {"UP", "DOWN", "LEFT", "RIGHT"}:
                continue
            if len(key) != 1 or not key.isprintable():
                continue
            if len(secret) >= max_length:
                overflow = True
                continue
            secret.append(key)
            target.write("*")
            target.flush()

    if read_key is not None:
        return interact(read_key)
    source = input_stream or sys.stdin
    if not interactive_terminal_available(source, target):
        raise TerminalInteractionError("a supported interactive terminal is required")
    with _terminal_key_reader(source) as reader:
        return interact(reader)


def read_visible_text(
    prompt: str,
    *,
    input_stream: IO[str] | None = None,
    output_stream: IO[str] | None = None,
    read_key: Callable[[], str] | None = None,
    max_length: int = 4096,
) -> str:
    """Read a short visible value with backspace, Enter, and Escape support."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    target = output_stream or sys.stderr
    value: list[str] = []

    def interact(reader: Callable[[], str]) -> str:
        target.write(_styled_input_prompt(target, prompt))
        target.flush()
        while True:
            key = reader()
            if key in _ENTER:
                target.write("\n")
                target.flush()
                return "".join(value)
            if key in {"ESC", "EOF"}:
                target.write("\n")
                target.flush()
                raise TerminalInteractionCancelled("text input cancelled")
            if key == "INTERRUPT":
                raise TerminalInteractionCancelled("text input cancelled")
            if key in _BACKSPACE:
                if value:
                    value.pop()
                    target.write("\b \b")
                    target.flush()
                continue
            if key in {"UP", "DOWN", "LEFT", "RIGHT"}:
                continue
            if len(key) != 1 or not key.isprintable():
                continue
            if len(value) >= max_length:
                continue
            value.append(key)
            target.write(key)
            target.flush()

    if read_key is not None:
        return interact(read_key)
    source = input_stream or sys.stdin
    if not interactive_terminal_available(source, target):
        raise TerminalInteractionError("a supported interactive terminal is required")
    with _terminal_key_reader(source) as reader:
        return interact(reader)


def view_scrollable_text(
    title: str,
    text: str,
    *,
    input_stream: IO[str] | None = None,
    output_stream: IO[str] | None = None,
    read_key: Callable[[], str] | None = None,
    viewport_height: int | None = None,
) -> None:
    """Show long text in a temporary, scrollable full-screen terminal view.

    The alternate screen keeps long legal or diagnostic text out of the normal
    shell transcript. The injected reader and viewport are test hooks; regular
    callers use the current POSIX terminal dimensions.
    """

    target = output_stream or sys.stderr
    source = input_stream or sys.stdin
    if read_key is None and not interactive_terminal_available(source, target):
        raise TerminalInteractionError("an interactive terminal is required")
    if os.environ.get("TERM", "").lower() == "dumb" and read_key is None:
        raise TerminalInteractionError("the terminal does not support a scrollable view")

    width = max(20, _terminal_width(target))
    height = viewport_height or _terminal_height(target)
    if height < 8:
        raise TerminalInteractionError("the terminal is too short for a scrollable view")
    body_height = height - 4
    lines = _wrap_terminal_document(text, width)
    max_offset = max(0, len(lines) - body_height)
    offset = 0

    def render() -> None:
        target.write("\x1b[H\x1b[2J")
        target.write(f"{_ANSI_BOLD}{_fit_terminal_text(title, width)}{_ANSI_RESET}\r\n")
        target.write(f"{_ANSI_DIM}{'-' * width}{_ANSI_RESET}\r\n")
        visible = lines[offset : offset + body_height]
        for index in range(body_height):
            target.write("\x1b[2K")
            if index < len(visible):
                target.write(visible[index])
            target.write("\r\n")
        position = f"Lines {offset + 1}-{min(len(lines), offset + body_height)} of {len(lines)}"
        hint = "Up/Down scroll  PgUp/PgDn page  Home/End jump  Q/Esc return"
        footer = _fit_terminal_text(f"{position}  |  {hint}", width)
        target.write(f"{_ANSI_DIM}{footer}{_ANSI_RESET}")
        target.flush()

    def interact(reader: Callable[[], str]) -> None:
        nonlocal offset
        target.write(_ANSI_ENTER_ALT_SCREEN + _ANSI_HIDE_CURSOR)
        try:
            render()
            while True:
                key = reader()
                if key in {"q", "Q", "ESC", "EOF", "\r", "\n"}:
                    return
                if key == "INTERRUPT":
                    raise TerminalInteractionCancelled("document review cancelled")
                if key in {"UP", "k"}:
                    offset = max(0, offset - 1)
                elif key in {"DOWN", "j"}:
                    offset = min(max_offset, offset + 1)
                elif key in {"PAGE_UP", "b"}:
                    offset = max(0, offset - body_height)
                elif key in {"PAGE_DOWN", " "}:
                    offset = min(max_offset, offset + body_height)
                elif key in {"HOME", "g"}:
                    offset = 0
                elif key in {"END", "G"}:
                    offset = max_offset
                else:
                    continue
                render()
        finally:
            target.write(_ANSI_SHOW_CURSOR + _ANSI_LEAVE_ALT_SCREEN)
            target.flush()

    if read_key is not None:
        interact(read_key)
        return
    with _terminal_key_reader(source) as reader:
        interact(reader)


def _render_choice(target: IO[str], choices: Sequence[Choice], selected: int | None) -> None:
    if selected is None:
        target.write("\r\x1b[2K  No selection  [Up/Down, Enter, Esc]")
        target.flush()
        return
    choice = choices[selected]
    detail = f" - {choice.detail}" if choice.detail else ""
    target.write(
        f"\r\x1b[2K  {selected + 1}/{len(choices)}  {choice.label}{detail}  [Up/Down, Enter, Esc]"
    )
    target.flush()


def _render_choice_menu(
    target: IO[str],
    prompt: str,
    choices: Sequence[Choice],
    selected: int | None,
    *,
    previous_line_count: int = 0,
) -> int:
    """Render and update the complete ANSI choice menu without scrolling."""

    lines = _choice_menu_lines(target, prompt, choices, selected)
    if previous_line_count:
        target.write("\r")
        if previous_line_count > 1:
            target.write(f"\x1b[{previous_line_count - 1}A")
    else:
        target.write("\r")
    canvas_height = max(previous_line_count, len(lines))
    for index in range(canvas_height):
        target.write("\x1b[2K")
        if index < len(lines):
            target.write(lines[index])
        if index + 1 < canvas_height:
            target.write("\r\n")
    target.flush()
    return len(lines)


def _choice_menu_lines(
    target: IO[str],
    prompt: str,
    choices: Sequence[Choice],
    selected: int | None,
) -> list[str]:
    width = _terminal_width(target)
    number_width = len(str(len(choices)))
    lines = [f"{_ANSI_BOLD}{_fit_terminal_text(prompt, width)}{_ANSI_RESET}", ""]
    for index, choice in enumerate(choices):
        focused = index == selected
        marker = ">" if focused else " "
        prefix = f"  {marker} {index + 1:>{number_width}}. "
        label = _fit_terminal_text(choice.label, max(1, width - len(prefix)))
        if focused:
            lines.append(f"{_ANSI_FOCUS}{prefix}{label}{_ANSI_RESET}")
        else:
            lines.append(f"{prefix}{label}")
        if choice.detail:
            detail_prefix = " " * len(prefix)
            detail = _fit_terminal_text(choice.detail, max(1, width - len(detail_prefix)))
            detail_style = _ANSI_FOCUS_DETAIL if focused else _ANSI_DIM
            lines.append(f"{detail_style}{detail_prefix}{detail}{_ANSI_RESET}")
    hint = "Up/Down move  Enter select  Esc back"
    lines.extend(("", f"{_ANSI_DIM}{_fit_terminal_text(hint, width)}{_ANSI_RESET}"))
    return lines


def _ansi_styling_available(target: IO[str]) -> bool:
    """Respect terminal capabilities and the conventional color opt-outs."""

    return (
        bool(getattr(target, "isatty", lambda: False)())
        and _terminal_width(target) >= 20
        and os.environ.get("TERM", "").lower() != "dumb"
        and "NO_COLOR" not in os.environ
        and os.environ.get("CLICOLOR") != "0"
    )


def _styled_input_prompt(target: IO[str], prompt: str) -> str:
    if not _ansi_styling_available(target):
        return prompt
    return f"{_ANSI_BOLD}{prompt}{_ANSI_RESET}"


def _terminal_width(target: IO[str]) -> int:
    try:
        return max(1, os.get_terminal_size(target.fileno()).columns)
    except (AttributeError, OSError, ValueError):
        return 80


def _terminal_height(target: IO[str]) -> int:
    try:
        return max(1, os.get_terminal_size(target.fileno()).lines)
    except (AttributeError, OSError, ValueError):
        return 24


def _fit_terminal_text(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return f"{value[: width - 3].rstrip()}..."


def _wrap_terminal_document(value: str, width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in value.expandtabs(4).splitlines():
        if not raw_line:
            lines.append("")
            continue
        indent = raw_line[: len(raw_line) - len(raw_line.lstrip())]
        wrapped = textwrap.wrap(
            raw_line,
            width=width,
            subsequent_indent=indent,
            replace_whitespace=False,
            drop_whitespace=True,
            break_long_words=True,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [""])
    return lines or [""]


@contextmanager
def _terminal_key_reader(stream: IO[str]) -> Iterator[Callable[[], str]]:
    """Yield a key reader while restoring terminal attributes on every exit."""

    try:
        import termios
        import tty
    except ImportError as exc:  # pragma: no cover - native Windows is unsupported
        raise TerminalInteractionError("POSIX terminal controls are unavailable") from exc

    try:
        descriptor = stream.fileno()
        original = termios.tcgetattr(descriptor)
    except (AttributeError, OSError, termios.error) as exc:
        raise TerminalInteractionError("could not enter interactive terminal mode") from exc

    try:
        tty.setcbreak(descriptor)
        yield lambda: _read_terminal_key(descriptor)
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, original)


def _read_terminal_key(descriptor: int) -> str:
    raw = os.read(descriptor, 1)
    if not raw or raw == b"\x04":
        return "EOF"
    if raw == b"\x03":
        return "INTERRUPT"
    if raw != b"\x1b":
        first = raw[0]
        expected = 1
        if 0xC2 <= first <= 0xDF:
            expected = 2
        elif 0xE0 <= first <= 0xEF:
            expected = 3
        elif 0xF0 <= first <= 0xF4:
            expected = 4
        if expected == 1:
            return raw.decode("utf-8", errors="ignore")
        encoded = bytearray(raw)
        while len(encoded) < expected:
            readable, _, _ = select.select([descriptor], [], [], 0.03)
            if not readable:
                return "UNKNOWN"
            encoded.extend(os.read(descriptor, expected - len(encoded)))
        return bytes(encoded).decode("utf-8", errors="ignore") or "UNKNOWN"

    sequence = bytearray(raw)
    while len(sequence) < 8:
        readable, _, _ = select.select([descriptor], [], [], 0.03)
        if not readable:
            break
        sequence.extend(os.read(descriptor, 1))
        if sequence[-1:] == b"~" or (len(sequence) >= 3 and chr(sequence[-1]).isalpha()):
            break
    return {
        b"\x1b[A": "UP",
        b"\x1b[B": "DOWN",
        b"\x1b[C": "RIGHT",
        b"\x1b[D": "LEFT",
        b"\x1b[5~": "PAGE_UP",
        b"\x1b[6~": "PAGE_DOWN",
        b"\x1b[H": "HOME",
        b"\x1b[1~": "HOME",
        b"\x1bOH": "HOME",
        b"\x1b[F": "END",
        b"\x1b[4~": "END",
        b"\x1bOF": "END",
        b"\x1b": "ESC",
    }.get(bytes(sequence), "UNKNOWN")
