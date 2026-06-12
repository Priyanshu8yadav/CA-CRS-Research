"""
threaded_stream.py
──────────────────
Daemon-threaded video frame reader for both local files and RTSP streams.

Design:
  - A background thread continuously reads frames from OpenCV VideoCapture.
  - Frames are placed into a bounded queue (maxsize=2).
  - The main thread calls `read()` to get the latest frame instantly (non-blocking).
  - For file sources, the video loops on EOF.
  - For RTSP sources, automatic reconnection is attempted on dropout.
"""
from __future__ import annotations

import logging
import queue
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_RTSP_PREFIXES = ("rtsp://", "rtsp_tcp://", "rtsps://", "http://", "https://")


class ThreadedStream:
    """Non-blocking video stream reader running in a daemon thread."""

    def __init__(
        self,
        source: str,
        queue_size: int = 2,
        reconnect_delay: float = 2.0,
        max_reconnect_attempts: int = 10,
    ):
        self.source = source
        self._is_rtsp = any(source.lower().startswith(p) for p in _RTSP_PREFIXES)
        self._reconnect_delay = reconnect_delay
        self._max_reconnect = max_reconnect_attempts

        self._cap = self._open_capture()
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_size)
        self._running = True
        self._lock = threading.Lock()

        self._thread = threading.Thread(
            target=self._reader_loop, daemon=True, name=f"Stream-{source[-30:]}"
        )
        self._thread.start()

    # ── Public API ────────────────────────────────────────────────────────────

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Non-blocking read. Returns (True, frame) or (False, None).
        Always returns the *most recent* frame, discarding stale ones.
        """
        frame = None
        try:
            # Drain queue to get latest frame
            while not self._queue.empty():
                frame = self._queue.get_nowait()
        except queue.Empty:
            pass
        if frame is not None:
            return True, frame
        return False, None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_rtsp(self) -> bool:
        return self._is_rtsp

    def stop(self) -> None:
        """Cleanly stop the reader thread and release the capture."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self._lock:
            if self._cap and self._cap.isOpened():
                self._cap.release()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _open_capture(self) -> cv2.VideoCapture:
        """Open a VideoCapture with FFmpeg backend and minimal buffering."""
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self._is_rtsp:
            # Prefer TCP transport for reliability over UDP
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        return cap

    def _reader_loop(self) -> None:
        """Background: continuously read frames into the bounded queue."""
        consecutive_failures = 0

        while self._running:
            with self._lock:
                cap = self._cap

            try:
                ret, frame = cap.read()
            except Exception:
                ret, frame = False, None

            if not ret or frame is None or frame.size == 0:
                consecutive_failures += 1

                if self._is_rtsp:
                    # RTSP dropout — attempt reconnect
                    if consecutive_failures <= self._max_reconnect:
                        logger.warning(
                            "[Stream] RTSP dropout (%s), reconnecting (%d/%d)...",
                            self.source[-40:], consecutive_failures, self._max_reconnect,
                        )
                        time.sleep(self._reconnect_delay)
                        self._reconnect()
                    else:
                        logger.error("[Stream] Max reconnects reached for %s", self.source)
                        break
                else:
                    # File mode — rewind to loop
                    with self._lock:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    consecutive_failures = 0
                continue

            # Success — reset failure counter
            consecutive_failures = 0

            # Put frame into queue, dropping oldest if full
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)

    def _reconnect(self) -> None:
        """Release and re-open the capture."""
        with self._lock:
            if self._cap and self._cap.isOpened():
                self._cap.release()
            self._cap = self._open_capture()

    def __del__(self):
        self.stop()
