"""FrameSource ABC.

A source's job is to hand the rest of the pipeline uniform ``Frame`` objects:
upright, rectified, RGB, scaled. Everything Aria-specific — fisheye warping,
the 90-degree rotation, gaze, on-device hand tracking — is resolved here so no
downstream module needs to know whether the pixels came from glasses, a VRS
file, or a webcam.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator

from ..frames import Frame


class FrameSource(abc.ABC):
    """An iterable source of frames."""

    #: Nominal frame rate, used for MP4 output timing.
    fps: float = 30.0

    #: Pinhole focal length in pixels for the frames this source emits, or None
    #: when unknown. The physical size prior needs this; without it the size
    #: term is dropped rather than guessed.
    focal_px: float | None = None

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Frame]:
        ...

    def __len__(self) -> int:
        raise TypeError(f"{type(self).__name__} has no known length")

    def close(self) -> None:
        return None

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
