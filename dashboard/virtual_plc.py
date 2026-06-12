"""
virtual_plc.py
──────────────
Modbus TCP virtual PLC for gate control with non-blocking writes.

Architecture:
  - In-memory dict is the authoritative gate state (instant, always works).
  - Modbus TCP server runs in a daemon thread (best-effort for hardware demo).
  - A persistent writer thread drains a command queue and sends coil writes
    to the Modbus server without blocking the caller.

Coil map:
  Coil 0 → Zone A gate  (True = OPEN)
  Coil 1 → Zone B gate  (True = OPEN)
  Coil 2 → Zone C gate  (True = OPEN)
  Coil 3 → Emergency    (True = ALL GATES OPEN)
"""
from __future__ import annotations

import logging
import queue
import threading
import time

logger = logging.getLogger(__name__)

PLC_HOST = "127.0.0.1"
PLC_PORT = 5020

# ── In-memory authoritative gate state ────────────────────────────────────────
_gate_states: dict[int, bool] = {0: False, 1: False, 2: False, 3: False}
_lock = threading.Lock()

# ── Server state ──────────────────────────────────────────────────────────────
_server_running = False
_server_thread: threading.Thread | None = None

# ── Async writer state ────────────────────────────────────────────────────────
_write_queue: queue.Queue[tuple[int, bool]] = queue.Queue(maxsize=32)
_writer_thread: threading.Thread | None = None
_writer_running = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Modbus TCP Server (daemon thread)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_server() -> None:
    """Run Modbus TCP server in a daemon thread."""
    global _server_running
    try:
        from pymodbus.datastore.context import (
            ModbusDeviceContext,
            ModbusSequentialDataBlock,
            ModbusServerContext,
        )
        from pymodbus.server import StartTcpServer

        coil_block = ModbusSequentialDataBlock(1, [False] * 16)
        device = ModbusDeviceContext(co=coil_block)
        context = ModbusServerContext(devices=device, single=True)

        logger.info("[PLC] Virtual PLC starting on %s:%d", PLC_HOST, PLC_PORT)
        _server_running = True
        StartTcpServer(context=context, address=(PLC_HOST, PLC_PORT))

    except Exception as exc:
        logger.warning("[PLC] Server unavailable (%s). Memory-only mode.", exc)
    finally:
        _server_running = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Async Writer Thread (persistent Modbus client)
# ═══════════════════════════════════════════════════════════════════════════════

def _writer_loop() -> None:
    """Background thread: drain the write queue and send to Modbus server."""
    global _writer_running
    _writer_running = True
    client = None

    while _writer_running:
        try:
            coil_index, value = _write_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if not _server_running:
            continue

        # Lazy-connect the persistent client
        if client is None:
            try:
                from pymodbus.client import ModbusTcpClient
                client = ModbusTcpClient(PLC_HOST, port=PLC_PORT, timeout=2)
                client.connect()
            except Exception as exc:
                logger.debug("[PLC-Writer] Connect failed: %s", exc)
                client = None
                continue

        try:
            client.write_coil(coil_index + 1, value)  # 1-based addressing
        except Exception as exc:
            logger.debug("[PLC-Writer] Write failed, reconnecting: %s", exc)
            try:
                client.close()
            except Exception:
                pass
            client = None  # will reconnect on next iteration


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def start_plc_server() -> None:
    """Start the virtual PLC server and async writer in background threads."""
    global _server_thread, _writer_thread

    # Start Modbus server
    if not _server_running and not (_server_thread and _server_thread.is_alive()):
        _server_thread = threading.Thread(
            target=_run_server, daemon=True, name="VirtualPLC-Server"
        )
        _server_thread.start()
        time.sleep(0.4)

    # Start async writer
    if not (_writer_thread and _writer_thread.is_alive()):
        _writer_thread = threading.Thread(
            target=_writer_loop, daemon=True, name="VirtualPLC-Writer"
        )
        _writer_thread.start()


def write_gate(coil_index: int, value: bool) -> bool:
    """
    Set a gate state. Returns immediately (non-blocking).
    In-memory state updates instantly; Modbus write is queued.
    """
    with _lock:
        _gate_states[coil_index] = value
    logger.info("[PLC] Gate %d → %s", coil_index, "OPEN" if value else "CLOSED")

    # Fire-and-forget to the async writer
    try:
        _write_queue.put_nowait((coil_index, value))
    except queue.Full:
        logger.debug("[PLC] Write queue full, dropping oldest")
        try:
            _write_queue.get_nowait()
        except queue.Empty:
            pass
        _write_queue.put_nowait((coil_index, value))

    return True


def write_emergency_open() -> bool:
    """Emergency lockdown: open ALL gates (coils 0-3)."""
    for i in range(4):
        write_gate(i, True)
    logger.warning("[PLC] 🚨 EMERGENCY: All gates OPENED")
    return True


def read_all_gates() -> dict[int, bool]:
    """Return current gate states (from authoritative in-memory dict)."""
    with _lock:
        return dict(_gate_states)


def is_plc_running() -> bool:
    """Returns True if the Modbus TCP server thread is alive."""
    return _server_running
