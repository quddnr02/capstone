from __future__ import annotations

import struct
from typing import Any

import can


class CanTargetSender:
    TARGET_VALID = 1 << 0
    DEPTH_VALID = 1 << 1

    def __init__(self, channel: str = "can0", interface: str = "socketcan", arbitration_id: int = 0x101):
        self.channel = channel
        self.interface = interface
        self.arbitration_id = arbitration_id
        self.sequence = 0
        self.bus: can.BusABC | None = can.interface.Bus(channel=channel, interface=interface)
        self._send_error_logged = False

    @staticmethod
    def _clamp(value: int, minimum: int, maximum: int) -> int:
        return max(minimum, min(maximum, value))

    @classmethod
    def _to_track_id(cls, raw: Any) -> int:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return -1
        return cls._clamp(value, -32768, 32767)

    @classmethod
    def _to_angle_cdeg(cls, angle_x: Any) -> int:
        try:
            value = float(angle_x)
        except (TypeError, ValueError):
            value = 0.0
        return cls._clamp(int(round(value * 100.0)), -32768, 32767)

    @classmethod
    def _to_distance_mm(cls, depth_m: Any) -> int:
        try:
            value = float(depth_m)
        except (TypeError, ValueError):
            return 0
        return cls._clamp(int(round(value * 1000.0)), 0, 65535)

    def _build_payload(self, target: dict[str, Any] | None) -> bytes:
        status = 0
        distance_mm = 0
        angle_cdeg = 0
        track_id = -1

        if isinstance(target, dict):
            status |= self.TARGET_VALID
            track_id = self._to_track_id(target.get("track_id"))
            angle_cdeg = self._to_angle_cdeg(target.get("angle_x", 0.0))
            depth_m = target.get("depth_m")
            if depth_m is not None:
                status |= self.DEPTH_VALID
                distance_mm = self._to_distance_mm(depth_m)

        payload = struct.pack("<BBHhh", status, self.sequence, distance_mm, angle_cdeg, track_id)
        if len(payload) != 8:
            raise RuntimeError(f"Invalid payload size: {len(payload)}")
        self.sequence = (self.sequence + 1) & 0xFF
        return payload

    def send_target(self, target: dict[str, Any] | None) -> bool:
        if self.bus is None:
            return False

        payload = self._build_payload(target)
        msg = can.Message(
            arbitration_id=self.arbitration_id,
            is_extended_id=False,
            data=payload,
        )
        try:
            self.bus.send(msg)
            return True
        except can.CanError as exc:
            if not self._send_error_logged:
                print(f"[WARN] CAN send failed: {exc}")
                self._send_error_logged = True
            return False

    def close(self) -> None:
        if self.bus is not None:
            self.bus.shutdown()
            self.bus = None
