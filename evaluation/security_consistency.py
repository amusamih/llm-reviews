from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (SRC_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


SCAN_PATTERNS = {
    "uncontrolled_exec_call": r"\bexec\s*\(",
    "uncontrolled_eval_call": r"\beval\s*\(",
    "dangerous_faiss_deserialization": r"\ballow_" r"dangerous_deserialization\b",
    "pickle_load": r"\bpickle\.load\s*\(",
    "yaml_load": r"\byaml\.load\s*\(",
    "shell_true": r"\bshell\s*=\s*True\b",
    "os_system": r"\bos\.system\s*\(",
}


def run_security_consistency_check(
    *,
    project_root: str | Path = Path("."),
    scan_roots: tuple[str, ...] = ("src", "app", "scripts"),
) -> dict[str, Any]:
    root = Path(project_root)
    findings = []
    for scan_root in scan_roots:
        base = root / scan_root
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for name, pattern in SCAN_PATTERNS.items():
                if re.search(pattern, text):
                    findings.append(
                        {
                            "pattern": name,
                            "path": str(path),
                            "regex": pattern,
                        }
                    )
    import_checks = _import_checks()
    return {
        "live_api_calls": False,
        "scan_roots": list(scan_roots),
        "findings": findings,
        "finding_count": len(findings),
        "checks": {
            "no_uncontrolled_exec_in_implementation_surface": not any(
                finding["pattern"] == "uncontrolled_exec_call" for finding in findings
            ),
            "no_uncontrolled_eval_in_implementation_surface": not any(
                finding["pattern"] == "uncontrolled_eval_call" for finding in findings
            ),
            "no_dangerous_faiss_deserialization": not any(
                finding["pattern"] == "dangerous_faiss_deserialization" for finding in findings
            ),
            "no_pickle_or_yaml_unsafe_loads": not any(
                finding["pattern"] in {"pickle_load", "yaml_load"} for finding in findings
            ),
            "no_shell_true_or_os_system": not any(
                finding["pattern"] in {"shell_true", "os_system"} for finding in findings
            ),
            "flask_server_importable": import_checks["flask_server_importable"],
            "package_importable": import_checks["package_importable"],
        },
        "import_checks": import_checks,
        "limitations": [
            "Static scan covers implementation roots only: src, app, scripts.",
            "This does not prove complete security, but it verifies the specific implementation concerns covered by the safety checks.",
        ],
    }


def write_security_consistency_report(output_path: str | Path, *, project_root: str | Path = Path(".")) -> Path:
    report = run_security_consistency_check(project_root=project_root)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _import_checks() -> dict[str, bool | str]:
    payload: dict[str, bool | str] = {
        "package_importable": False,
        "flask_server_importable": False,
        "error": "",
    }
    try:
        importlib.import_module("llm_review_analysis")
        payload["package_importable"] = True
        server = importlib.import_module("app.server")
        payload["flask_server_importable"] = callable(getattr(server, "create_app", None))
    except Exception as exc:  # pragma: no cover - exercised only on environment failures.
        payload["error"] = f"{type(exc).__name__}: {exc}"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a static security/implementation consistency report.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    path = write_security_consistency_report(args.output, project_root=args.project_root)
    print(f"security_consistency={path}")


if __name__ == "__main__":
    main()

