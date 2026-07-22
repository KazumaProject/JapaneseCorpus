from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def read_zstd_jsonl(path: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["zstd", "--quiet", "--decompress", "--stdout", str(path)],
        check=True,
        capture_output=True,
    )
    return [json.loads(line) for line in result.stdout.decode("utf-8").splitlines()]
