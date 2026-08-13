"""Frame sources. Imports are lazy so the Aria ones stay optional."""

from __future__ import annotations

from .base import FrameSource

__all__ = ["FrameSource", "VideoFrameSource"]


def __getattr__(name: str):  # noqa: D105 - lazy re-export
    if name == "VideoFrameSource":
        from .video import VideoFrameSource

        return VideoFrameSource
    if name == "VrsFrameSource":
        from .vrs import VrsFrameSource

        return VrsFrameSource
    if name == "LiveFrameSource":
        from .live import LiveFrameSource

        return LiveFrameSource
    raise AttributeError(name)
