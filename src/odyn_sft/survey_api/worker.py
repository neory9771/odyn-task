"""One-shot Python worker; no provider initialization or parent credentials."""

from __future__ import annotations

import json
import resource
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text())
    limits = request["limits"]
    resource.setrlimit(
        resource.RLIMIT_CPU, (limits["cpu_seconds"], limits["cpu_seconds"] + 1)
    )
    memory = limits["memory_mb"] * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(
        resource.RLIMIT_FSIZE, (limits["max_output_bytes"], limits["max_output_bytes"])
    )
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # -I ignores PYTHONPATH; this trusted path loads the installed/source library.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import pandas as pd

    from odyn_sft.survey_api.errors import AnalysisError
    from odyn_sft.survey_api.runtime import execute_program
    from odyn_sft.survey_api.service import SurveyAPI
    from odyn_sft.survey_api.tracing import TracedSurveyAPI

    response: dict[str, Any]

    try:
        data = pd.read_parquet(request_path.parent / "respondents.parquet")
        cls = TracedSurveyAPI if request["collect_trace"] else SurveyAPI
        api = cls(data, request["metadata"])
        evidence = execute_program(
            request["program"],
            api,
            max_output_bytes=limits["max_output_bytes"],
            variable_aliases=request.get("variable_aliases"),
        )
        if isinstance(api, TracedSurveyAPI):
            evidence["analysis_trace"] = api.trace
        response = {"evidence": evidence}
    except AnalysisError as error:
        response = {"error": error.as_dict()}
    except BaseException as error:
        response = {
            "error": {
                "message": type(error).__name__ + ": " + str(error),
                "line": None,
                "column": None,
            }
        }
    encoded = json.dumps(response, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode("utf-8")) > limits["max_output_bytes"]:
        encoded = json.dumps(
            {"error": {"message": "OutputLimit: evidence exceeded the output limit"}}
        )
    (request_path.parent / "response.json").write_text(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
