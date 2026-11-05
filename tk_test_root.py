"""The one Tk root the window tests share, and a watchdog for when a test hangs.

On GitHub's macOS runner, the test run hung with no output at the first test to make a
new tk.Tk() after destroying an earlier one. Each test still gets its own new window
contents; only the root stays open between them.
"""

from __future__ import annotations

import faulthandler
import tkinter as tk
import unittest

WATCHDOG_SECONDS = 120      # far longer than any window test takes
_root = None


def open_root(test: unittest.TestCase, seconds: float = WATCHDOG_SECONDS) -> tk.Tk:
    """The shared root, emptied after the test. Skips the test if there's no display.

    If the test takes longer than `seconds`, every thread's stack is printed and the
    whole test run exits with a failure, so a hang names itself.
    """
    global _root
    faulthandler.dump_traceback_later(seconds, exit=True)
    test.addCleanup(faulthandler.cancel_dump_traceback_later)
    if _root is None:
        try:
            _root = tk.Tk()
        except tk.TclError as exc:
            test.skipTest(f"no display: {exc}")
        _root.withdraw()
    test.addCleanup(_empty, _root)
    return _root


def _empty(root: tk.Tk) -> None:
    for child in root.winfo_children():
        child.destroy()
    root.unbind_all("<MouseWheel>")      # the app's scrolling areas add to this each time
    root.update_idletasks()
