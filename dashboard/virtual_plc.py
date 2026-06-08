"""
virtual_plc.py
──────────────
Asynchronous Modbus TCP server acting as a virtual PLC for gate control.
Compatible with pymodbus >= 3.10 (ModbusDeviceContext API).

Coil map:
  Coil 0 → Zone A gate  (True = OPEN)
  Coil 1 → Zone B gate  (True = OPEN)
  Coil 2 → Zone C gate  (True = OPEN)
  Coil 3 → Emergency    (True = ALL GATES OPEN)

Design: In-memory dict is always the authoritative gate state.
        Modbus server/client is a best-effort layer for the hardware demo.
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

PLC_HOST = "127.0.0.1"
PLC_PORT = 5020

# In-memory gate state — always authoritative.
_gate_states: dict[int, bool] = {0: False, 1: False, 2: False, 3: False}
_server_running  = False
_server_thread   = None


def _run_server() -> None:
    """Run Modbus TCP server in a daemon thread (pymodbus >= 3.10)."""
    global _server_running
    try:
        # pymodbus 3.10+ imports
        from pymodbus.datastore.context import (
            ModbusDeviceContext,
            ModbusSequentialDataBlock,
            ModbusServerContext,
        )
        from pymodbus.server import StartTcpServer

        # Address must be 1-based for coils in pymodbus 3.x
        coil_block = ModbusSequentialDataBlock(1, [False] * 16)
        device     = ModbusDeviceContext(co=coil_block)
        context    = ModbusServerContext(slaves=device, single=True)

        logger.info("[PLC] Virtual PLC starting on %s:%d", PLC_HOST, PLC_PORT)
        _server_running = True
        StartTcpServer(context=context, address=(PLC_HOST, PLC_PORT))

    except Exception as exc:
        logger.warning("[PLC] Server unavailable (%s). Using memory-only mode.", exc)
    finally:
        _server_running = False


def start_plc_server() -> None:
    global _server_thread, _server_running
    if _server_running or (_server_thread and _server_thread.is_alive()):
        return
    _server_thread = threading.Thread(target=_run_server, daemon=True, name="VirtualPLC")
    _server_thread.start()
    time.sleep(0.4)


def write_gate(coil_index: int, value: bool) -> bool:
    """Update gate state. Always succeeds (in-memory). Attempts Modbus write."""
    _gate_states[coil_index] = value
    logger.info("[PLC] Gate %d → %s", coil_index, "OPEN" if value else "CLOSED")

    if _server_running:
        try:
            from pymodbus.client import ModbusTcpClient
            client = ModbusTcpClient(PLC_HOST, port=PLC_PORT, timeout=1)
            if client.connect():
                client.write_coil(coil_index + 1, value)   # 1-based address
                client.close()
        except Exception as exc:
            logger.debug("[PLC] Modbus write (non-fatal): %s", exc)

    return True


def read_all_gates() -> dict[int, bool]:
    """Return current gate states."""
    return dict(_gate_states)


def is_plc_running() -> bool:
    return bool(_server_thread and _server_thread.is_alive() and _server_running)
