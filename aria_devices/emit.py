"""Integration outputs: push results to another process or machine.

The Aria v1 study already forwarded frames over UDP to a downstream consumer,
so this keeps that shape but sends the structured result instead of pixels —
one JSON datagram per frame, which is far smaller and directly usable.

Three sinks, all optional and all fire-and-forget:

``UdpEmitter``      one JSON datagram per frame to host:port
``StdoutEmitter``   line-delimited JSON on stdout, for piping into another tool
``JsonlWriter``     (in pipeline.py) the on-disk record

All of them attach via ``DevicePipeline.add_result_hook``, so adding a new
transport means writing one callable and nothing else.
"""

from __future__ import annotations

import json
import logging
import socket
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import FrameResult

log = logging.getLogger(__name__)

#: Datagrams above this are likely to be dropped or fragmented on the wire.
SAFE_DATAGRAM_BYTES = 8192


class UdpEmitter:
    """Sends one JSON datagram per frame.

    UDP is the right choice here and the tradeoff is deliberate: a dropped
    frame of detections is harmless because another arrives ~30 ms later,
    whereas a TCP stall would back-pressure the whole detection pipeline.

    Oversized payloads are trimmed rather than sent and silently lost: the
    per-signal debug breakdown is dropped first, since it is the bulky part and
    the least useful to a downstream consumer.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8899, include_signals: bool = False) -> None:
        self.host = host
        self.port = port
        self.include_signals = include_signals
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent = 0
        self.dropped = 0
        log.info("UDP emitter -> %s:%d", host, port)

    def __call__(self, result: "FrameResult") -> None:
        record = result.to_record()
        if not self.include_signals:
            for det in record.get("detections", []):
                det.pop("signals", None)
        payload = json.dumps(record, separators=(",", ":")).encode("utf-8")

        if len(payload) > SAFE_DATAGRAM_BYTES:
            for det in record.get("detections", []):
                det.pop("signals", None)
            payload = json.dumps(record, separators=(",", ":")).encode("utf-8")
        if len(payload) > SAFE_DATAGRAM_BYTES:
            self.dropped += 1
            log.warning("dropping oversized datagram (%d bytes)", len(payload))
            return

        try:
            self._sock.sendto(payload, (self.host, self.port))
            self.sent += 1
        except OSError as exc:  # pragma: no cover - network dependent
            self.dropped += 1
            log.warning("UDP send failed: %s", exc)

    def close(self) -> None:
        self._sock.close()


class EventEmitter:
    """Sends only hand-on-device *events*, one datagram each.

    This is the integration surface for the interaction system: a device does
    not need bounding boxes or scores, it needs "a hand entered my screen at
    (0.42, 0.66) and it is grabbing". Sending only transitions keeps the wire
    quiet — nothing is sent at all while the hands are away from every device.

    Payload per datagram::

        {"state":"enter","device":"tablet","hand":"right","pose":"grab",
         "x":0.42,"y":0.66,"distance":0.0,"timestamp_ns":...}

    ``x`` / ``y`` are normalised screen coordinates, origin top-left. Multiply
    by the target device's own resolution to place a cursor.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8900) -> None:
        self.host = host
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent = 0
        log.info("event emitter -> %s:%d", host, port)

    def __call__(self, result: "FrameResult") -> None:
        for event in result.events:
            payload = json.dumps(event.to_record(), separators=(",", ":")).encode("utf-8")
            try:
                self._sock.sendto(payload, (self.host, self.port))
                self.sent += 1
            except OSError as exc:  # pragma: no cover - network dependent
                log.warning("event send failed: %s", exc)

    def close(self) -> None:
        self._sock.close()


def listen(port: int = 8900, host: str = "0.0.0.0", pretty: bool = True) -> None:
    """Print datagrams as they arrive — a stand-in for your receiver.

    Lets you confirm the wire format and that events are actually reaching the
    other side, before wiring anything into the real system.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    # Unbuffered: this is meant to be watched live or piped into a file.
    print(f"listening on {host}:{port} — Ctrl+C to stop", flush=True)
    try:
        while True:
            data, _addr = sock.recvfrom(65535)
            try:
                record = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                print(f"<{len(data)} bytes of non-JSON>", flush=True)
                continue
            if pretty and "state" in record:
                xy = (
                    f"({record['x']:.3f}, {record['y']:.3f})"
                    if record.get("x") is not None
                    else "-"
                )
                print(
                    f"{record['state']:<8} {record.get('device','?'):<7} "
                    f"{record.get('hand','?'):<5} {record.get('pose','?'):<7} {xy}",
                    flush=True,
                )
            else:
                print(json.dumps(record), flush=True)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        sock.close()


class StdoutEmitter:
    """Line-delimited JSON on stdout, for `aria-devices ... | your-tool`.

    Writes to stdout only; every log line in this package goes to stderr, so
    the stream stays machine-readable.
    """

    def __init__(self, include_signals: bool = False) -> None:
        self.include_signals = include_signals

    def __call__(self, result: "FrameResult") -> None:
        record = result.to_record()
        if not self.include_signals:
            for det in record.get("detections", []):
                det.pop("signals", None)
        sys.stdout.write(json.dumps(record, separators=(",", ":")) + "\n")
        sys.stdout.flush()
