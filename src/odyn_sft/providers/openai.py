"""Explicit OpenAI calls with durable request fingerprints and no implicit retries."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from ..common.storage import digest, read_json, write_json

T = TypeVar("T", bound=BaseModel)


class ResponsesProvider:
    """Reuse completed requests; never silently replay failed or ambiguous calls."""

    def __init__(
        self,
        model: str,
        ledger: Path,
        max_output_tokens: int = 6000,
        reasoning_effort: str | None = "low",
        timeout: float = 120,
        credentials_file: Path | None = None,
    ):
        self.model, self.ledger = model, ledger
        self.max_tokens, self.reasoning = max_output_tokens, reasoning_effort
        self.timeout, self.credentials_file = timeout, credentials_file

    def _client(self):
        from openai import OpenAI

        key = os.environ.get("OPENAI_API_KEY")
        if not key and self.credentials_file is not None:
            if self.credentials_file.stat().st_mode & 0o077:
                raise ValueError("Credentials file must have mode 0600")
            key = read_json(self.credentials_file).get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set OPENAI_API_KEY or supply --credentials-file")
        return OpenAI(api_key=key, max_retries=0, timeout=self.timeout)

    def complete(
        self, instruction: str, payload: str, schema: type[T], request_id: str
    ) -> T:
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = read_json(self.ledger) if self.ledger.exists() else {}
            fingerprint = digest(
                {
                    "model": self.model,
                    "instruction": instruction,
                    "payload": payload,
                    "schema": schema.model_json_schema(),
                    "max_tokens": self.max_tokens,
                    "reasoning": self.reasoning,
                }
            )
            if request_id in state:
                previous = state[request_id]
                if previous["fingerprint"] != fingerprint:
                    raise ValueError("Frozen provider request changed")
                if previous["status"] != "completed":
                    raise ValueError(
                        "Unresolved/failed provider call; reconcile the ledger before retrying"
                    )
                return schema.model_validate(previous["value"])
            # Check credentials before creating a started request; never save keys.
            client = self._client()
            state[request_id] = {
                "fingerprint": fingerprint,
                "status": "started",
                "model": self.model,
            }
            write_json(self.ledger, state)
            try:
                kwargs = {
                    "model": self.model,
                    "input": [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": payload},
                    ],
                    "text_format": schema,
                    "max_output_tokens": self.max_tokens,
                    "store": False,
                }
                if self.reasoning:
                    kwargs["reasoning"] = {"effort": self.reasoning}
                response = client.responses.parse(**kwargs)
                if response.status != "completed" or response.output_parsed is None:
                    raise ValueError("Model response incomplete/refused/unparseable")
                value = schema.model_validate(response.output_parsed.model_dump())
                state[request_id].update(
                    status="completed",
                    value=value.model_dump(),
                    response_id=response.id,
                    usage=response.usage.model_dump() if response.usage else None,
                )
                write_json(self.ledger, state)
                return value
            except Exception as exc:
                # Exception messages can contain provider context: store only types.
                state[request_id].update(status="failed", error_type=type(exc).__name__)
                write_json(self.ledger, state)
                raise
