"""Maps jupyter_rfb (browser W3C KeyboardEvent.key) strings and app-model key types.

This module has no scenex dependencies — only app_model.types — so it can be
upstreamed to app-model's backends in the future.

Key values follow the W3C ``KeyboardEvent.key`` specification:
https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/key/Key_Values
"""

from __future__ import annotations

from typing import Any

from app_model.types import KeyCode, KeyCombo, KeyMod

# ---------------------------------------------------------------------------
# Static mapping: W3C key name → KeyCode
# ---------------------------------------------------------------------------

_KEY_FROM_JUPYTER_STATIC: dict[str, KeyCode | KeyCombo] = {
    # Functional keys
    "Backspace": KeyCode.Backspace,
    "Tab": KeyCode.Tab,
    "Enter": KeyCode.Enter,
    "Escape": KeyCode.Escape,
    " ": KeyCode.Space,
    "Delete": KeyCode.Delete,
    "Insert": KeyCode.Insert,
    # Arrow keys
    "ArrowLeft": KeyCode.LeftArrow,
    "ArrowRight": KeyCode.RightArrow,
    "ArrowUp": KeyCode.UpArrow,
    "ArrowDown": KeyCode.DownArrow,
    # Navigation
    "Home": KeyCode.Home,
    "End": KeyCode.End,
    "PageUp": KeyCode.PageUp,
    "PageDown": KeyCode.PageDown,
    # Lock keys
    "NumLock": KeyCode.NumLock,
    "CapsLock": KeyCode.CapsLock,
    # Modifier keys (when the key itself is the modifier)
    "Control": KeyCode.Ctrl,
    "Shift": KeyCode.Shift,
    "Alt": KeyCode.Alt,
    "Meta": KeyCode.Meta,
    # Function keys
    "F1": KeyCode.F1,
    "F2": KeyCode.F2,
    "F3": KeyCode.F3,
    "F4": KeyCode.F4,
    "F5": KeyCode.F5,
    "F6": KeyCode.F6,
    "F7": KeyCode.F7,
    "F8": KeyCode.F8,
    "F9": KeyCode.F9,
    "F10": KeyCode.F10,
    "F11": KeyCode.F11,
    "F12": KeyCode.F12,
    # Punctuation / symbol keys (unshifted value)
    "`": KeyCode.Backquote,
    "\\": KeyCode.Backslash,
    "[": KeyCode.BracketLeft,
    "]": KeyCode.BracketRight,
    ",": KeyCode.Comma,
    "=": KeyCode.Equal,
    "-": KeyCode.Minus,
    ".": KeyCode.Period,
    "'": KeyCode.Quote,
    ";": KeyCode.Semicolon,
    "/": KeyCode.Slash,
}

# ---------------------------------------------------------------------------
# Build the full table by adding letters and digits dynamically.
# The browser sends the *actual* character, so "a" and "A" are distinct, but
# both map to the same physical key (KeyCode.KeyA).
# ---------------------------------------------------------------------------

KEY_FROM_JUPYTER: dict[str, KeyCode | KeyCombo] = dict(_KEY_FROM_JUPYTER_STATIC)

# Letters: lowercase and uppercase both map to the same KeyCode
for _c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    KEY_FROM_JUPYTER[_c.lower()] = getattr(KeyCode, f"Key{_c}")
    KEY_FROM_JUPYTER[_c] = getattr(KeyCode, f"Key{_c}")

# Digits
for _d in "0123456789":
    KEY_FROM_JUPYTER[_d] = getattr(KeyCode, f"Digit{_d}")


def jupyterkey2modelkey(ev: dict[str, Any]) -> KeyCode | KeyCombo:
    """Convert a jupyter_rfb key event dict to an app-model KeyCode or KeyCombo.

    ``ev`` is expected to have a ``"key"`` field (W3C KeyboardEvent.key string)
    and an optional ``"modifiers"`` field (iterable of strings such as
    ``"Control"``, ``"Shift"``, ``"Alt"``, ``"Meta"``).

    The returned value encodes both the base key and any held modifiers so it
    can be passed directly to ``SimpleKeyBinding.from_int()``.
    """
    key: KeyCode | KeyCombo = KEY_FROM_JUPYTER.get(ev["key"], KeyCode.UNKNOWN)

    modifiers = ev.get("modifiers", ())
    mods = KeyMod.NONE
    if "Control" in modifiers:
        mods |= KeyMod.CtrlCmd
    if "Shift" in modifiers:
        mods |= KeyMod.Shift
    if "Alt" in modifiers:
        mods |= KeyMod.Alt
    if "Meta" in modifiers:
        mods |= KeyMod.WinCtrl

    if mods:
        return mods | key  # type: ignore[return-value]
    return key
