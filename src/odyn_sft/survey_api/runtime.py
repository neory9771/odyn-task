"""Execute ordinary Python with Survey API bindings in a separate process."""

from __future__ import annotations

import ast
import json
import math
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import stats

from .errors import AnalysisError
from .service import SurveyAPI
from .types import EvidencePack, Metadata

CALLS = frozenset(
    {
        "var",
        "find_variables",
        "describe",
        "values",
        "eligible",
        "where",
        "group",
        "subclaim",
        "proportion",
        "distribution",
        "rank_options",
        "top_box",
        "mean_score",
        "compare",
        "trend",
        "association",
        "adjusted_compare",
        "not_measured",
        "proxy",
        "caveat",
        "segment_sizes",
    }
)
PROGRAM_FILENAME = "analysis.py"


@dataclass(frozen=True)
class ProcessLimits:
    """OS/process limits, without restricting Python syntax or API call counts."""

    timeout_seconds: float = 30
    cpu_seconds: int = 20
    memory_mb: int = 2048
    max_output_bytes: int = 2_000_000
    max_program_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("Timeout must be finite and positive")
        if any(
            type(v) is not int or v < 1
            for v in (
                self.cpu_seconds,
                self.memory_mb,
                self.max_output_bytes,
                self.max_program_bytes,
            )
        ):
            raise ValueError("Process limits must be positive integers")


DEFAULT_LIMITS = ProcessLimits()


def static_check(program: str) -> ast.Module:
    """Check ordinary Python syntax only; no language subset or custom evaluator."""
    if not isinstance(program, str):
        raise AnalysisError("InvalidProgram: expected a string")
    try:
        tree = ast.parse(program, filename=PROGRAM_FILENAME)
        # Compilation also checks Python context errors, e.g. top-level return.
        compile(tree, PROGRAM_FILENAME, "exec")
        return tree
    except (SyntaxError, ValueError, RecursionError) as error:
        raise AnalysisError(
            "InvalidSyntax: " + getattr(error, "msg", str(error)),
            line=getattr(error, "lineno", None),
            column=getattr(error, "offset", None),
        ) from error


class _DiscardOutput:
    """Printed diagnostics are separate from the structured evidence channel."""

    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.written = 0

    def write(self, text: str) -> int:
        self.written += len(text.encode("utf-8"))
        if self.written > self.maximum:
            raise AnalysisError("OutputLimit: printed output exceeded the limit")
        return len(text)

    def flush(self) -> None:
        pass


def _execution_error(error: BaseException) -> AnalysisError:
    line = None
    trace = error.__traceback__
    while trace:
        if trace.tb_frame.f_code.co_filename == PROGRAM_FILENAME:
            line = trace.tb_lineno
        trace = trace.tb_next
    if isinstance(error, AnalysisError):
        return error.located(line)
    return AnalysisError(type(error).__name__ + ": " + str(error), line=line)


def execute_program(
    program: str,
    api: SurveyAPI,
    *,
    max_output_bytes: int = 2_000_000,
    variable_aliases: dict[str, str] | None = None,
) -> EvidencePack:
    """Trusted reference execution in the current process using native Python.

    Candidate/LLM programs use run_program, which creates a separate process.
    This helper is for deterministic reference code authored by this package.
    """
    static_check(program)
    bindings: dict[str, Any] = {
        "__name__": "__main__", "__file__": PROGRAM_FILENAME,
        "np": np, "numpy": np, "pd": pd, "pandas": pd,
        "scipy": scipy, "stats": stats, "statsmodels": statsmodels, "sm": sm,
        "data": api.data.copy(deep=True), "metadata": deepcopy(api.metadata),
    }
    bindings.update({name: getattr(api, name) for name in CALLS})
    # Numeric references are derived from this exact frozen metadata order.
    # Explicit profile aliases remain supported for historical catalogue formats.
    if variable_aliases is None:
        from ..survey_metadata.catalogue import numeric_references
        variable_aliases = numeric_references(api.metadata['variables'])['variables']
    if variable_aliases:

        aliases = variable_aliases

        def resolve(identifier: str | int) -> str:
            if type(identifier) is int:
                key = str(identifier)
                if key not in aliases:
                    raise AnalysisError(f'UnknownVariable: numeric reference {identifier}')
                return api.var(aliases[key])
            if not isinstance(identifier, str):
                raise AnalysisError('UnknownVariable: expected a variable ID or integer reference')
            # Exact string IDs remain usable by canonical generator programs.
            return api.var(identifier if identifier in api.cards else aliases.get(identifier, identifier))

        def variable_call(name: str) -> Callable[..., Any]:
            def call(v: str | int, *args: Any, **kwargs: Any) -> Any:
                return getattr(api, name)(resolve(v), *args, **kwargs)
            return call

        def group(v: Any, mapping: Any = None) -> Any:
            return api.group(v if isinstance(v, dict) else resolve(v), mapping)

        def proxy(concept: str, variables: list[str | int], rationale: str) -> dict[str, Any]:
            if not isinstance(variables, list):
                raise AnalysisError('WrongProxy: expected a list of variable references')
            return api.proxy(concept, [resolve(v) for v in variables], rationale)

        def association(x: str | int, y: str | int, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return api.association(resolve(x), resolve(y), *args, **kwargs)

        def adjusted_compare(est: Any, a: str, b: str, controls: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
            controls = [resolve(v) for v in controls] if isinstance(controls, list) else resolve(controls)
            return api.adjusted_compare(est, a, b, controls, *args, **kwargs)

        block_aliases = {str(i): block for i, block in enumerate(api.blocks, 1)}

        def rank_options(block: str | int, *args: Any, **kwargs: Any) -> Any:
            if type(block) is int:
                if str(block) not in block_aliases:
                    raise AnalysisError(f"UnknownBlock: numeric reference {block}")
                block = block_aliases[str(block)]
            if not isinstance(block, str):
                raise AnalysisError("UnknownBlock: expected a block ID or integer reference")
            return api.rank_options(block, *args, **kwargs)

        bindings.update({name: variable_call(name) for name in (
            'eligible', 'where', 'proportion', 'top_box', 'mean_score',
            'describe', 'values', 'distribution')})
        bindings.update(var=resolve, group=group, proxy=proxy, association=association,
                        adjusted_compare=adjusted_compare, rank_options=rank_options)
    output = _DiscardOutput(max_output_bytes)
    try:
        with redirect_stdout(output), redirect_stderr(output):
            exec(compile(program, PROGRAM_FILENAME, "exec"), bindings, bindings)
    except BaseException as error:
        raise _execution_error(error) from error
    return api.pack()


def run_reference(program: str, wide: pd.DataFrame, metadata: Metadata) -> EvidencePack:
    """Execute deterministic generator-authored code without process startup cost."""
    return execute_program(program, SurveyAPI(wide, metadata))


def run_program(
    program: str,
    wide: pd.DataFrame,
    metadata: Metadata,
    *,
    limits: ProcessLimits = DEFAULT_LIMITS,
    collect_trace: bool = False,
    variable_aliases: dict[str, str] | None = None,
) -> EvidencePack:
    """Run candidate Python in a fresh worker with a deadline and clean environment.

    A child process separates execution state and applies resource limits. It is
    not a filesystem/network security sandbox; callers needing that use a container.
    Only the worker's structured API evidence, never printed text, is returned.
    """
    if not isinstance(program, str):
        raise AnalysisError("InvalidProgram: expected a string")
    if len(program.encode("utf-8")) > limits.max_program_bytes:
        raise AnalysisError("ProgramTooLarge: source byte limit exceeded")
    static_check(program)
    # Validate before serialization for consistent library errors.
    from .data import normalize_data

    wide = normalize_data(wide, metadata)
    worker = Path(__file__).with_name("worker.py")
    with TemporaryDirectory(prefix="survey-python-") as scratch_name:
        scratch = Path(scratch_name)
        wide.to_parquet(scratch / "respondents.parquet", index=True)
        (scratch / "request.json").write_text(
            json.dumps(
                {
                    "program": program,
                    "metadata": metadata,
                    "collect_trace": collect_trace,
                    "variable_aliases": variable_aliases,
                    "limits": vars(limits),
                },
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        (scratch / PROGRAM_FILENAME).write_text(program)
        environment = {
            "PATH": os.defpath,
            "HOME": scratch_name,
            "LANG": "C.UTF-8",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "-1",
        }
        process = subprocess.Popen(
            [sys.executable, "-I", str(worker), str(scratch / "request.json")],
            cwd=scratch,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            process.wait(timeout=limits.timeout_seconds)
        except subprocess.TimeoutExpired as error:
            _stop_process_group(process)
            raise AnalysisError(
                "ExecutionTimeout: Python worker exceeded its deadline"
            ) from error
        finally:
            # Also clean up descendants if candidate code starts subprocesses.
            _stop_process_group(process)
        response_path = scratch / "response.json"
        if process.returncode != 0 or not response_path.exists():
            code = (
                "ResourceLimit"
                if process.returncode is not None and process.returncode < 0
                else "WorkerFailure"
            )
            raise AnalysisError(
                code + ": Python worker exited without a completed response"
            )
        if response_path.stat().st_size > limits.max_output_bytes:
            raise AnalysisError("OutputLimit: evidence exceeded the output limit")
        try:
            response = json.loads(response_path.read_text())
        except (ValueError, OSError) as error:
            raise AnalysisError("WorkerFailure: invalid worker response") from error
        if "error" in response:
            failure = response["error"]
            raise AnalysisError(
                failure["message"],
                line=failure.get("line"),
                column=failure.get("column"),
            )
        return response["evidence"]


def _stop_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
