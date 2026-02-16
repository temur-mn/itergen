"""Backward-compatible wrapper for ``vorongen.api.cli``."""

from .api import cli as _impl

_SKIP_NAMES = {"main"}
for _name in dir(_impl):
    if _name.startswith("__") or _name in _SKIP_NAMES:
        continue
    globals()[_name] = getattr(_impl, _name)


def _sync_impl_symbols() -> None:
    _impl.available_sample_configs = globals()["available_sample_configs"]
    _impl.get_sample_config = globals()["get_sample_config"]
    _impl.load_config = globals()["load_config"]
    _impl.VorongenSynthesizer = globals()["VorongenSynthesizer"]


def main(argv=None):
    _sync_impl_symbols()
    return _impl.main(argv)


__all__ = [name for name in dir(_impl) if not name.startswith("__")]
