from __future__ import annotations

import struct
import time
from typing import Optional


class TargetCanSender:
    def __init__(
        self,
        channel: str = "can0",
        interface: str = "socketcan",
        bitrate: int = 500000,
        arbitration_id: int = 0x101,
        send_hz: float = 10.0,
        enabled: bool = True,
        debug: bool = False,
    ):
        self.channel = channel
        self.interface = interface
        self.bitrate = bitrate
        self.arbitration_id = arbitration_id
        self.send_hz = send_hz
        self.enabled = enabled
        self.debug = debug

        self._bus = None
        self._can_module = None
        self._min_interval = 0.0 if send_hz <= 0 else 1.0 / send_hz
        self._last_send_time = 0.0
        self._sequence = 0
        self._error_logged = False

        self._connect()

    def _connect(self) -> None:
        if not self.enabled:
            print("[CAN] disabled: --no-can option")
            return

        try:
            import can  # type: ignore

            self._can_module = can
        except Exception as exc:
            print(f"[CAN] disabled: python-can import failed ({exc})")
            self.enabled = False
            return

        try:
            self._bus = self._can_module.interface.Bus(
                channel=self.channel,
                bustype=self.interface,
                bitrate=self.bitrate,
            )
            print(
                f"[CAN] connected: interface={self.interface} channel={self.channel} "
                f"id=0x{self.arbitration_id:03X} send_hz={self.send_hz}"
            )
        except Exception as exc:
            print(f"[CAN] disabled: failed to connect CAN bus ({exc})")
            self._bus = None
            self.enabled = False

    def _clamp_u16(self, value: int) -> int:
        return max(0, min(65534, value))

    def _clamp_i16(self, value: int) -> int:
        return max(-32768, min(32767, value))

    def _build_payload(
        self,
        track_id: Optional[int],
        depth_m: Optional[float],
        angle_x_deg: Optional[float],
        confidence: Optional[float],
        reconnected: bool,
    ) -> tuple[bytes, dict]:
        flags = 0

        if track_id is None:
            distance_mm = 0xFFFF
            angle_cdeg = 0
            track_id_u8 = 0xFF
            confidence_pct = 0
        else:
            flags |= 0x01
            if depth_m is None:
                distance_mm = 0xFFFF
            else:
                distance_mm = self._clamp_u16(int(round(depth_m * 1000.0)))
                flags |= 0x02

            if angle_x_deg is None:
                angle_cdeg = 0
            else:
                angle_cdeg = self._clamp_i16(int(round(angle_x_deg * 100.0)))
                flags |= 0x04

            track_id_u8 = max(0, min(254, int(track_id)))
            conf_raw = 0.0 if confidence is None else float(confidence)
            confidence_pct = max(0, min(100, int(round(conf_raw * 100.0))))

            if reconnected:
                flags |= 0x08

        seq = self._sequence
        payload = struct.pack("<HhBBBB", distance_mm, angle_cdeg, track_id_u8, flags, confidence_pct, seq)
        fields = {
            "distance_mm": distance_mm,
            "angle_cdeg": angle_cdeg,
            "track_id": track_id_u8,
            "flags": flags,
            "confidence_pct": confidence_pct,
            "sequence": seq,
        }
        return payload, fields

    def send_target_state(
        self,
        track_id: int | None,
        depth_m: float | None,
        angle_x_deg: float | None,
        confidence: float | None,
        reconnected: bool = False,
        force: bool = False,
    ) -> bool:
        now = time.time()
        if not force and (now - self._last_send_time) < self._min_interval:
            return False

        payload, fields = self._build_payload(track_id, depth_m, angle_x_deg, confidence, reconnected)

        if self._bus is None or self._can_module is None:
            self._sequence = (self._sequence + 1) & 0xFF
            self._last_send_time = now
            return False

        try:
            msg = self._can_module.Message(
                arbitration_id=self.arbitration_id,
                data=payload,
                is_extended_id=False,
            )
            self._bus.send(msg)
            self._sequence = (self._sequence + 1) & 0xFF
            self._last_send_time = now
            if self.debug:
                print(
                    f"[CAN TX] id=0x{self.arbitration_id:03X} data={payload.hex()} "
                    f"depth_mm={fields['distance_mm']} angle_cdeg={fields['angle_cdeg']} "
                    f"track_id={fields['track_id']} flags=0x{fields['flags']:02X} "
                    f"conf={fields['confidence_pct']} seq={fields['sequence']}"
                )
            return True
        except Exception as exc:
            self._sequence = (self._sequence + 1) & 0xFF
            self._last_send_time = now
            if not self._error_logged:
                print(f"[CAN] send error: {exc}")
                self._error_logged = True
            return False

    def close(self) -> None:
        if self._bus is not None:
            try:
                self._bus.shutdown()
            except Exception:
                pass
            self._bus = None
