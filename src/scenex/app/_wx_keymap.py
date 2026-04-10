"""Mapping between wxPython key codes and app-model key types.

This module has no scenex dependencies — only wx and app_model.types — so it
can be upstreamed to app-model's backends in the future.
"""

from __future__ import annotations

import wx
from app_model.types import KeyCode, KeyCombo, KeyMod

# ---------------------------------------------------------------------------
# Special-key mapping: wx.WXK_* → KeyCode
# ---------------------------------------------------------------------------

_KEY_FROM_WX_SPECIAL: dict[int, KeyCode | KeyCombo] = {
    wx.WXK_BACK: KeyCode.Backspace,
    wx.WXK_TAB: KeyCode.Tab,
    wx.WXK_RETURN: KeyCode.Enter,
    wx.WXK_ESCAPE: KeyCode.Escape,
    wx.WXK_SPACE: KeyCode.Space,
    wx.WXK_DELETE: KeyCode.Delete,
    wx.WXK_INSERT: KeyCode.Insert,
    wx.WXK_LEFT: KeyCode.LeftArrow,
    wx.WXK_RIGHT: KeyCode.RightArrow,
    wx.WXK_UP: KeyCode.UpArrow,
    wx.WXK_DOWN: KeyCode.DownArrow,
    wx.WXK_HOME: KeyCode.Home,
    wx.WXK_END: KeyCode.End,
    wx.WXK_PAGEUP: KeyCode.PageUp,
    wx.WXK_PAGEDOWN: KeyCode.PageDown,
    wx.WXK_NUMLOCK: KeyCode.NumLock,
    wx.WXK_CAPITAL: KeyCode.CapsLock,
    wx.WXK_SHIFT: KeyCode.Shift,
    wx.WXK_ALT: KeyCode.Alt,
    wx.WXK_CONTROL: KeyCode.Ctrl,
    # Function keys
    wx.WXK_F1: KeyCode.F1,
    wx.WXK_F2: KeyCode.F2,
    wx.WXK_F3: KeyCode.F3,
    wx.WXK_F4: KeyCode.F4,
    wx.WXK_F5: KeyCode.F5,
    wx.WXK_F6: KeyCode.F6,
    wx.WXK_F7: KeyCode.F7,
    wx.WXK_F8: KeyCode.F8,
    wx.WXK_F9: KeyCode.F9,
    wx.WXK_F10: KeyCode.F10,
    wx.WXK_F11: KeyCode.F11,
    wx.WXK_F12: KeyCode.F12,
    # Numpad digits
    wx.WXK_NUMPAD0: KeyCode.Numpad0,
    wx.WXK_NUMPAD1: KeyCode.Numpad1,
    wx.WXK_NUMPAD2: KeyCode.Numpad2,
    wx.WXK_NUMPAD3: KeyCode.Numpad3,
    wx.WXK_NUMPAD4: KeyCode.Numpad4,
    wx.WXK_NUMPAD5: KeyCode.Numpad5,
    wx.WXK_NUMPAD6: KeyCode.Numpad6,
    wx.WXK_NUMPAD7: KeyCode.Numpad7,
    wx.WXK_NUMPAD8: KeyCode.Numpad8,
    wx.WXK_NUMPAD9: KeyCode.Numpad9,
    # Numpad operators
    wx.WXK_NUMPAD_ADD: KeyCode.NumpadAdd,
    wx.WXK_NUMPAD_SUBTRACT: KeyCode.NumpadSubtract,
    wx.WXK_NUMPAD_MULTIPLY: KeyCode.NumpadMultiply,
    wx.WXK_NUMPAD_DIVIDE: KeyCode.NumpadDivide,
    wx.WXK_NUMPAD_DECIMAL: KeyCode.NumpadDecimal,
    wx.WXK_NUMPAD_ENTER: KeyCode.Enter,
}

# ---------------------------------------------------------------------------
# Build the full lookup table by adding ASCII-range keys dynamically.
# wx returns the *uppercase* ASCII value for letter keys regardless of Shift.
# Digit keys return their ASCII value (48-57).
# Punctuation keys return their ASCII value directly.
# ---------------------------------------------------------------------------

KEY_FROM_WX: dict[int, KeyCode | KeyCombo] = dict(_KEY_FROM_WX_SPECIAL)

# Letters: ord('A')=65 … ord('Z')=90
for _c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    KEY_FROM_WX[ord(_c)] = getattr(KeyCode, f"Key{_c}")

# Digits: ord('0')=48 … ord('9')=57
for _d in "0123456789":
    KEY_FROM_WX[ord(_d)] = getattr(KeyCode, f"Digit{_d}")

# Punctuation (ASCII values wx reports for these keys)
KEY_FROM_WX.update(
    {
        ord("`"): KeyCode.Backquote,
        ord("-"): KeyCode.Minus,
        ord("="): KeyCode.Equal,
        ord("["): KeyCode.BracketLeft,
        ord("]"): KeyCode.BracketRight,
        ord("\\"): KeyCode.Backslash,
        ord(";"): KeyCode.Semicolon,
        ord("'"): KeyCode.Quote,
        ord(","): KeyCode.Comma,
        ord("."): KeyCode.Period,
        ord("/"): KeyCode.Slash,
    }
)


def wxevent2modelkey(event: wx.KeyEvent) -> KeyCode | KeyCombo:
    """Convert a wx.KeyEvent to an app-model KeyCode or KeyCombo.

    The returned value encodes both the base key and any held modifiers so it
    can be passed directly to ``SimpleKeyBinding.from_int()``.
    """
    key: KeyCode | KeyCombo = KEY_FROM_WX.get(event.GetKeyCode(), KeyCode.UNKNOWN)

    mods = KeyMod.NONE
    if event.ControlDown():
        mods |= KeyMod.CtrlCmd
    if event.ShiftDown():
        mods |= KeyMod.Shift
    if event.AltDown():
        mods |= KeyMod.Alt
    if event.MetaDown():
        mods |= KeyMod.WinCtrl

    if mods:
        return mods | key  # type: ignore[return-value]
    return key
