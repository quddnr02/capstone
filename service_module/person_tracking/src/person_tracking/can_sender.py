from __future__ import annotations

import struct
from typing import Any

import can


class CanTargetSender:
    TARGET_VALID = 1 << 0
    DEPTH_VALID = 1 << 1
    TARGET_REACHED = 1 << 2

    def __init__(
        self,
        channel: str = "can0",
        interface: str = "socketcan",
        arbitration_id: int = 0x101,
        target_follow_distance_m: float = 1.0,
        distance_deadband_m: float = 0.10,
        angle_deadband_deg: float = 3.0,
    ):
        self.channel = channel
        self.interface = interface
        self.arbitration_id = arbitration_id
        self.target_follow_distance_m = target_follow_distance_m
        self.distance_deadband_m = distance_deadband_m
        self.angle_deadband_deg = angle_deadband_deg
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
    def _to_target_move_distance_mm(cls, target_move_distance_m: Any) -> int:
        try:
            value = float(target_move_distance_m)
        except (TypeError, ValueError):
            return 0
        return cls._clamp(int(round(value * 1000.0)), -32768, 32767)

    def _build_payload(self, target: dict[str, Any] | None) -> bytes:
        status = 0
        target_move_distance_mm = 0
        angle_cdeg = 0
        track_id = -1

        if isinstance(target, dict):
            status |= self.TARGET_VALID
            track_id = self._to_track_id(target.get("track_id"))
            angle_deg = float(target.get("angle_x", 0.0))
            if abs(angle_deg) <= self.angle_deadband_deg:
                angle_deg = 0.0
            angle_cdeg = self._to_angle_cdeg(angle_deg)
            depth_m = target.get("depth_m")
            if depth_m is not None:
                status |= self.DEPTH_VALID
                measured_distance_m = float(depth_m)
                target_move_distance_m = measured_distance_m - self.target_follow_distance_m
                if abs(target_move_distance_m) <= self.distance_deadband_m:
                    target_move_distance_m = 0.0
                    status |= self.TARGET_REACHED
                target_move_distance_mm = self._to_target_move_distance_mm(target_move_distance_m)

        payload = struct.pack("<BBhhh", status, self.sequence, target_move_distance_mm, angle_cdeg, track_id)
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
