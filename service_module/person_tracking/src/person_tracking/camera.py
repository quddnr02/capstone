from __future__ import annotations

import numpy as np
from primesense import openni2


class AstraCamera:
    def __init__(self, openni_path: str):
        self.openni_path = openni_path
        self.dev = None
        self.color_stream = None
        self.depth_stream = None

    def start(self) -> None:
        openni2.initialize(self.openni_path)
        self.dev = openni2.Device.open_any()
        print("Device opened:", self.dev.get_device_info())

        self.color_stream = self.dev.create_color_stream()
        self.depth_stream = self.dev.create_depth_stream()

        self.color_stream.start()
        self.depth_stream.start()

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        if self.color_stream is None or self.depth_stream is None:
            raise RuntimeError("Camera streams are not started")

        color_frame = self.color_stream.read_frame()
        depth_frame = self.depth_stream.read_frame()

        cw, ch = color_frame.width, color_frame.height
        dw, dh = depth_frame.width, depth_frame.height

        color_data = color_frame.get_buffer_as_uint8()
        depth_data = depth_frame.get_buffer_as_uint16()

        rgb = np.frombuffer(color_data, dtype=np.uint8).reshape((ch, cw, 3)).copy()
        depth = np.frombuffer(depth_data, dtype=np.uint16).reshape((dh, dw)).copy()

        return rgb, depth

    def close(self) -> None:
        if self.color_stream is not None:
            self.color_stream.stop()
        if self.depth_stream is not None:
            self.depth_stream.stop()
        openni2.unload()
