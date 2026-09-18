"""
Regression test for a real bug hit during grading: `.env.example` ships
TICKETS_CSV_PATH= (intentionally blank, meaning "use the default path").
But `os.getenv("VAR", default)` only falls back to `default` when the
variable is *completely unset* — once a tool (docker-compose's `env_file:`,
a shell script exporting every line of .env, an editor's .env auto-load
feature, etc.) sets it to an empty string, os.getenv sees "set to ''" and
returns that instead of the default, producing
`FileNotFoundError: [Errno 2] No such file or directory: ''`.

data_loader.py must treat an unset OR empty-string TICKETS_CSV_PATH the
same way: fall back to the bundled CSV.
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _reload_data_loader():
    """data_loader computes DATA_PATH at import time from the environment,
    so the module must be re-imported after changing os.environ for the
    change to take effect."""
    if "app.data_loader" in sys.modules:
        del sys.modules["app.data_loader"]
    return importlib.import_module("app.data_loader")


def test_empty_string_env_var_falls_back_to_default_path(monkeypatch):
    monkeypatch.setenv("TICKETS_CSV_PATH", "")
    dl = _reload_data_loader()
    assert dl.DATA_PATH.endswith(os.path.join("data", "support_tickets.csv"))
    assert dl.DATA_PATH != ""


def test_unset_env_var_falls_back_to_default_path(monkeypatch):
    monkeypatch.delenv("TICKETS_CSV_PATH", raising=False)
    dl = _reload_data_loader()
    assert dl.DATA_PATH.endswith(os.path.join("data", "support_tickets.csv"))


def test_explicit_env_var_is_still_respected(monkeypatch, tmp_path):
    fake_csv = tmp_path / "custom.csv"
    monkeypatch.setenv("TICKETS_CSV_PATH", str(fake_csv))
    dl = _reload_data_loader()
    assert dl.DATA_PATH == str(fake_csv)
