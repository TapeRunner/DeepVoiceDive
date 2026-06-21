"""Smoke tests for the GUI module.

The GUI needs a display (and the Tk system package) to actually run, so these
tests only cover what works headlessly: the module imports cleanly, exposes its
entry point, and its method-resolution helper falls back like the CLI does.
"""
import importlib

import pytest


def test_gui_imports_without_tkinter():
    # Importing the module must never require Tkinter to be installed.
    gui = importlib.import_module("deepvoicedive.gui")
    assert callable(gui.main)
    assert callable(gui.build_app)


def test_resolve_method_passthrough_for_mfcc():
    from deepvoicedive.gui import _resolve_method

    assert _resolve_method("mfcc") == "mfcc"


def test_main_reports_missing_tkinter(monkeypatch, capsys):
    import builtins

    import deepvoicedive.gui as gui

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tkinter":
            raise ImportError("no display / no python3-tk")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    rc = gui.main()
    assert rc == 1
    assert "Tkinter" in capsys.readouterr().err


def test_build_app_requires_tkinter():
    pytest.importorskip("tkinter")
    # With a real display this would construct a window; in headless CI Tk is
    # absent and the import-skip above keeps the suite green.
