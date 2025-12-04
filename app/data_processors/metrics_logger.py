# metrics_logger.py
import json
import gzip
import time
from typing import Any, Dict, Optional, Iterable
import os


class MetricsLogger:
    """
    Simple generic logger for simulation/metrics data.

    - Writes one JSON object per line (NDJSON).
    - Optionally gzip-compressed.
    """

    def __init__(
        self,
        path: str,
        *,
        compress: bool = False,
        buffer_size: int = 100,
        auto_timestamp: bool = True,
    ) -> None:
        """
        Args:
            path: File path for the log (e.g. "run1.jsonl" or "run1.jsonl.gz").
            compress: If True, use gzip compression.
            buffer_size: How many records to buffer in memory before flushing.
            auto_timestamp: If True, add a 'timestamp' field (time.time())
                            to records that don't already provide it.
        """
        self.path = path
        self.compress = compress
        self.buffer_size = buffer_size
        self.auto_timestamp = auto_timestamp

        self._file = self._open_file()
        self._buffer: list[Dict[str, Any]] = []

    def _open_file(self):
        if self.compress:
            # "at" = append text mode
            return gzip.open(self.path, mode="at", encoding="utf-8")
        else:
            # newline="" to avoid extra newline translation issues
            return open(self.path, mode="a", encoding="utf-8", newline="")

    def log(self, **fields: Any) -> None:
        """
        Log a single record. Fields must be JSON-serializable.
        """
        if self.auto_timestamp and "timestamp" not in fields:
            fields["timestamp"] = time.time()

        self._buffer.append(fields)
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """
        Flush buffered records to disk.
        """
        if not self._buffer:
            return

        # Use compact separators to reduce size: no spaces after commas/colons
        for rec in self._buffer:
            line = json.dumps(rec, separators=(",", ":"))
            self._file.write(line + "\n")

        self._file.flush()
        self._buffer.clear()

    def close(self) -> None:
        """
        Flush and close the underlying file.
        """
        self.flush()
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def iter_metrics(path):
    """
    Iterate over JSONL(-gz) metrics file.

    Robust to:
    - truncated gzip files (EOFError)
    - bad JSON lines
    """
    if not os.path.exists(path):
        return

    if path.endswith(".gz"):
        f_open = lambda p: gzip.open(p, "rt")
    else:
        f_open = lambda p: open(p, "rt")

    try:
        with f_open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping bad JSON line in %s: %r", path, line[:100])
                    continue
    except EOFError:
        # Truncated gzip – ignore the rest of the file
        print(
            "Truncated metrics file %s (EOFError while reading gzip). "
            "Using partial data up to the truncated point.",
            path,
        )
        return

def clear_metrics_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
