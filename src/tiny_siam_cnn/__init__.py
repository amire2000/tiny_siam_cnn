"""Tiny Siamese CNN tracking library."""
__all__ = ["TinySiamese", "Tracker"]


def __getattr__(name):
    if name == "TinySiamese":
        from .demo import TinySiamese
        return TinySiamese
    if name == "Tracker":
        from .tracker import Tracker
        return Tracker
    raise AttributeError(name)
