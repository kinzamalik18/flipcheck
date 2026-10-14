"""Test configuration for FlipCheck.

This conftest.py applies platform-specific workarounds needed for tests to
run correctly on all supported operating systems.
"""

from __future__ import annotations

import sys
import tempfile


def pytest_configure(config):
    """Apply platform-specific patches before the test session starts.

    On Windows, SQLite connections hold exclusive file locks until the
    connection is explicitly closed or the object is garbage collected.
    tempfile.TemporaryDirectory raises PermissionError on cleanup when any
    SQLite file inside the directory is still locked. Forcing a GC sweep
    before cleanup, or ignoring cleanup errors, prevents spurious failures
    that are unrelated to the test logic.
    """
    if sys.platform == "win32":
        _patch_temporary_directory_for_windows()


def _patch_temporary_directory_for_windows() -> None:
    """Patch TemporaryDirectory to ignore cleanup errors on Windows.

    SQLiteStore (and other modules that hold open file handles) can prevent
    Windows from deleting temporary files immediately. The test assertions
    still run correctly; only the directory removal may fail.
    """
    original_init = tempfile.TemporaryDirectory.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("ignore_cleanup_errors", True)
        original_init(self, *args, **kwargs)

    tempfile.TemporaryDirectory.__init__ = patched_init  # type: ignore[method-assign]
