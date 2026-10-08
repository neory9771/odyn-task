"""Small argument/progress helpers shared by dataset commands."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def progress(event: dict[str, Any]) -> None:
    print(json.dumps(event), file=sys.stderr, flush=True)
