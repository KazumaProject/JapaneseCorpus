from __future__ import annotations

import io
import subprocess
from pathlib import Path
from types import TracebackType
from typing import TextIO


class ZstdTextWriter:
    """Stream UTF-8 text into a zstd file using the portable zstd CLI."""

    def __init__(self, output_path: Path, level: int = 10) -> None:
        self.output_path = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(
            [
                "zstd",
                "--quiet",
                "--force",
                f"-{level}",
                "--threads=0",
                "-o",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
        )
        if self._process.stdin is None:
            raise RuntimeError("zstd did not expose stdin")
        self.stream: TextIO = io.TextIOWrapper(
            self._process.stdin, encoding="utf-8", newline="\n"
        )

    def __enter__(self) -> TextIO:
        return self.stream

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            try:
                self.stream.close()
            finally:
                self._process.terminate()
                self._process.wait()
                self.output_path.unlink(missing_ok=True)
            return

        self.stream.close()
        return_code = self._process.wait()
        if return_code != 0:
            self.output_path.unlink(missing_ok=True)
            raise RuntimeError(f"zstd failed with exit code {return_code}")
