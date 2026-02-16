"""Backward-compatible wrapper for ``vorongen.api.synthesizer``."""

from .api import synthesizer as _impl

_ORIGINAL_IS_TORCH_AVAILABLE = _impl.is_torch_available
_SKIP_NAMES = {
    "VorongenSynthesizer",
    "compare_torch_vs_classic",
    "generate",
    "is_torch_available",
}
for _name in dir(_impl):
    if _name.startswith("__") or _name in _SKIP_NAMES:
        continue
    globals()[_name] = getattr(_impl, _name)


def _sync_impl_symbols() -> None:
    _impl.is_torch_available = globals()["is_torch_available"]


def is_torch_available() -> bool:
    return _ORIGINAL_IS_TORCH_AVAILABLE()


class VorongenSynthesizer(_impl.VorongenSynthesizer):
    def generate(self):
        _sync_impl_symbols()
        return super().generate()


def generate(config, run_config=None):
    _sync_impl_symbols()
    return _impl.generate(config, run_config=run_config)


def compare_torch_vs_classic(config, run_config=None):
    _sync_impl_symbols()
    return _impl.compare_torch_vs_classic(config, run_config=run_config)


__all__ = [name for name in dir(_impl) if not name.startswith("__")]
