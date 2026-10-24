"""Clip-local immutable frame storage with bounded RAM and lossless disk spill."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile

import numpy as np


@dataclass(frozen=True)
class RepresentationDescriptor:
    name: str
    layout: str = "HW"
    dtype: str = "uint8"
    value_range: tuple[int, int] = (0, 255)
    preprocessing_fingerprint: str | None = None


class BoundedFrameStore:
    """Own pixels until close; disk files are uncompressed .npy, never JPEG.

    RAM accounting covers retained pixel buffers, disk accounting includes .npy
    headers. Reads of spilled images are read-only memory maps. No full-frame
    float conversion or GPU allocation occurs here.
    """

    def __init__(
        self,
        clip_id: str,
        ram_limit_bytes: int = 512 * 1024**2,
        disk_limit_bytes: int = 20 * 1024**3,
        directory: str | Path | None = None,
        representations: tuple[RepresentationDescriptor, ...] | None = None,
    ):
        if ram_limit_bytes < 0 or disk_limit_bytes < 0:
            raise ValueError("Frame store limits cannot be negative")
        self.clip_id = str(clip_id)
        self.ram_limit_bytes, self.disk_limit_bytes = int(ram_limit_bytes), int(
            disk_limit_bytes
        )
        self.ram_bytes = self.disk_bytes = 0
        self.peak_ram_bytes = self.peak_disk_bytes = 0
        self._memory = OrderedDict()
        self._disk = {}
        self._shapes = {}
        self._frame_hw = None
        descriptors = representations or (RepresentationDescriptor("bgs_signed_u8"),)
        self.representations = {r.name: r for r in descriptors}
        if directory is not None:
            Path(directory).mkdir(parents=True, exist_ok=True)
        self.directory = Path(
            tempfile.mkdtemp(prefix="fisheye-endpoints-", dir=directory)
        )
        self.closed = False

    @property
    def frame_ids(self):
        return tuple(sorted({key[0] for key in self._shapes}))

    @property
    def frame_hw(self):
        return self._frame_hw

    def frame_shape(self, frame_id: int, representation: str = "bgs_signed_u8"):
        self._check_open()
        return self._shapes[(int(frame_id), representation)]

    def _check_open(self):
        if self.closed:
            raise RuntimeError("Frame store is already closed")

    def _spill(self, key, image):
        # np.save's actual byte count is checked before publishing this entry.
        path = self.directory / f"{len(self._disk):010d}.npy"
        if image.nbytes > self.disk_limit_bytes - self.disk_bytes:
            raise OSError("Endpoint frame store disk limit exceeded")
        try:
            with path.open("xb") as handle:
                np.save(handle, image, allow_pickle=False)
            size = path.stat().st_size
            if size > self.disk_limit_bytes - self.disk_bytes:
                raise OSError(
                    "Endpoint frame store disk limit exceeded (including array headers)"
                )
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        self._disk[key] = path
        self.disk_bytes += size
        self.peak_disk_bytes = max(self.peak_disk_bytes, self.disk_bytes)

    def put(
        self, frame_id: int, image: np.ndarray, representation: str = "bgs_signed_u8"
    ):
        self._check_open()
        if representation not in self.representations:
            raise ValueError(f"Undeclared frame representation {representation!r}")
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 2 or min(array.shape) < 1:
            raise ValueError(
                "Endpoint frame representations must be nonempty HW uint8 arrays"
            )
        if self._frame_hw is not None and array.shape != self._frame_hw:
            raise ValueError(
                "All frames and representations in one clip must share original dimensions"
            )
        key = (int(frame_id), representation)
        if key in self._shapes:
            raise ValueError(f"Frame representation already stored: {key}")
        if array.nbytes > self.ram_limit_bytes:
            # Save directly from the caller's immutable-for-this-call input:
            # copying a huge frame first would defeat a small RAM budget.
            self._spill(key, array)
        else:
            while self.ram_bytes + array.nbytes > self.ram_limit_bytes:
                oldest_key, oldest = next(iter(self._memory.items()))
                self._spill(oldest_key, oldest)
                self._memory.pop(oldest_key)
                self.ram_bytes -= oldest.nbytes
            # Allocate only after eviction. Bytes-backed buffers cannot have
            # WRITEABLE restored by a caller, unlike an owned ndarray.
            owned = np.frombuffer(array.tobytes(order="C"), dtype=np.uint8).reshape(
                array.shape
            )
            self._memory[key] = owned
            self.ram_bytes += owned.nbytes
            self.peak_ram_bytes = max(self.peak_ram_bytes, self.ram_bytes)
        self._shapes[key] = array.shape
        self._frame_hw = array.shape

    def get(self, frame_id: int, representation: str = "bgs_signed_u8") -> np.ndarray:
        self._check_open()
        key = (int(frame_id), representation)
        if key in self._memory:
            return self._memory[key]
        if key in self._disk:
            return np.load(self._disk[key], mmap_mode="r", allow_pickle=False)
        raise KeyError(
            f"No {representation!r} image for frame {frame_id} in clip {self.clip_id!r}"
        )

    def metadata(self):
        return {
            "clip_id": self.clip_id,
            "frame_hw": self.frame_hw,
            "frame_ids": list(self.frame_ids),
            "ram_bytes": self.ram_bytes,
            "disk_bytes": self.disk_bytes,
            "peak_ram_bytes": self.peak_ram_bytes,
            "peak_disk_bytes": self.peak_disk_bytes,
            "representations": {
                name: vars(desc) for name, desc in self.representations.items()
            },
        }

    def close(self):
        if not self.closed:
            self._memory.clear()
            self._disk.clear()
            self._shapes.clear()
            self.ram_bytes = self.disk_bytes = 0
            shutil.rmtree(self.directory)
            self.closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


FrameStore = BoundedFrameStore
