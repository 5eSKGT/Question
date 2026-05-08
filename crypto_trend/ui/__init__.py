"""UI package — `charts` is dash-independent so it stays importable in tests."""
from .charts import SIGNAL_STYLE, build_signal_chart

__all__ = ["SIGNAL_STYLE", "build_signal_chart"]


def build_app(*args, **kwargs):  # lazy proxy — avoids hard dash import at collection time
    from .app import build_app as _impl
    return _impl(*args, **kwargs)


def run_ui(*args, **kwargs):
    from .app import run_ui as _impl
    return _impl(*args, **kwargs)
