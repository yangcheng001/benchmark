"""AgentX 逐档位 manifest、summary 归档和跨档位屏幕汇总。

产物布局::

    <work_dir>/
    ├── configs/<cfg_time>_<pid>.py
    ├── logs/<run_id>/{agentx_stdout.log, agentx_adapter.log, agentx_summary.log}
    └── performances/<run_id>/{agentx_raw/, agentx_summary.json|csv, execution_plan.redacted.json, manifest.json}
"""

import json
import os
import os.path as osp
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tabulate

from ais_bench.benchmark.tasks.custom_tasks.agentx_result import (
    atomic_write_json,
    json_safe,
)
from ais_bench.benchmark.utils.logging.logger import AISLogger

RAW_SUMMARY_NAME = "profile_export_aiperf.json"
RAW_SUMMARY_CSV_NAME = "profile_export_aiperf.csv"
RAW_CONSOLE_NAME = "profile_export_console.log"
SUMMARY_JSON_NAME = "agentx_summary.json"
SUMMARY_CSV_NAME = "agentx_summary.csv"
SUMMARY_LOG_NAME = "agentx_summary.log"
PERFORMANCES_DIR = "performances"
LOGS_DIR = "logs"

# 上游 AIPerf 控制台输出带 "NVIDIA AIPerf |" 品牌前缀，写入 summary 日志时去掉。
_BRAND_PREFIX_RE = re.compile(r"NVIDIA\s+AIPerf\s*\|\s*")

DISPLAY_COLUMNS = [
    "model", "dataset", "concurrency", "comparison_scope", "status", "benchmark_valid",
]


def _display(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def render_table(rows: List[Dict[str, Any]]) -> str:
    """屏幕与逐档位 summary 日志共用的表格渲染。"""
    display = [[_display(row.get(column)) for column in DISPLAY_COLUMNS] for row in rows]
    return tabulate.tabulate(display, headers=DISPLAY_COLUMNS, tablefmt="grid")


def _atomic_text(path: str, text: str) -> None:
    directory = osp.dirname(osp.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".agentx-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _metric(metrics: Dict[str, Any], name: str, field: str = "avg") -> Any:
    metric = (metrics or {}).get(name) or {}
    value = metric.get(field)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _coverage_passed(coverage: Any) -> Optional[bool]:
    if not isinstance(coverage, list) or not coverage:
        return None
    passed = []
    for item in coverage:
        if not isinstance(item, dict):
            passed.append(False)
            continue
        required = item.get("required_ratio")
        ratios = [item.get("ttft_ratio"), item.get("inter_token_latency_ratio")]
        if not isinstance(required, (int, float)) or any(
            not isinstance(value, (int, float)) for value in ratios
        ):
            passed.append(False)
        else:
            passed.append(all(value >= required for value in ratios))
    return all(passed)


def manifest_row(manifest: Dict[str, Any], raw: Dict[str, Any],
                 work_dir: str, run_dir: str) -> Dict[str, Any]:
    """manifest 记录运行事实，raw summary 提供指标本体，两者合成汇总行。"""
    result = manifest.get("result") or {}
    dataset = manifest.get("dataset") or {}
    model = manifest.get("model") or {}
    upstream = manifest.get("upstream") or {}
    run_params = manifest.get("run_params") or {}
    test_params = manifest.get("test_params") or {}
    metadata = raw.get("metadata") if isinstance(raw, dict) else None
    coverage = metadata.get("metric_duration_coverage") if isinstance(metadata, dict) else None

    def relative(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        try:
            return osp.relpath(path, work_dir)
        except ValueError:
            return path

    summary_json = result.get("summary_json")
    summary_csv = result.get("summary_csv")
    return {
        "schema_version": manifest.get("schema_version"),
        "run_id": manifest.get("run_id"),
        "model": model.get("name"), "model_abbr": model.get("abbr"),
        "dataset": dataset.get("registry_alias") or dataset.get("abbr"),
        "dataset_abbr": dataset.get("abbr"), "dataset_variant": dataset.get("variant"),
        "concurrency": test_params.get("batch_size"), "run_mode": run_params.get("run_mode"),
        "status": result.get("status"), "data_status": result.get("data_status"),
        "benchmark_valid": result.get("benchmark_valid"),
        "comparison_scope": result.get("comparison_scope"),
        "harness_version": upstream.get("version"),
        "harness_submission_valid": result.get("harness_submission_valid"),
        "invalid_reasons": result.get("invalid_reasons"),
        "request_count": _metric(raw, "request_count"),
        "error_request_count": _metric(raw, "error_request_count"),
        "request_error_rate": _metric(raw, "request_error_rate"),
        "request_throughput": _metric(raw, "request_throughput"),
        "output_token_throughput": _metric(raw, "output_token_throughput"),
        "output_token_throughput_per_user": _metric(raw, "output_token_throughput_per_user"),
        "time_to_first_token_p90": _metric(raw, "time_to_first_token", "p90"),
        "inter_token_latency_p90": _metric(raw, "inter_token_latency", "p90"),
        "request_latency_p90": _metric(raw, "request_latency", "p90"),
        "e2e_normalized_interactivity_p75": _metric(raw, "e2e_normalized_interactivity_p75"),
        "e2e_normalized_interactivity_p90": _metric(raw, "e2e_normalized_interactivity_p90"),
        "effective_concurrency": _metric(raw, "effective_concurrency"),
        "effective_decode_throughput": _metric(raw, "effective_decode_throughput"),
        "context_overflow_count": _metric(raw, "context_overflow_count"),
        "metric_coverage_passed": _coverage_passed(coverage),
        "raw_summary_path": relative(osp.join(run_dir, summary_json)) if summary_json else None,
        "summary_csv_path": relative(osp.join(run_dir, summary_csv)) if summary_csv else None,
        "manifest_path": relative(osp.join(run_dir, "manifest.json")),
    }


class AgentXSummarizer:
    """单档位落盘入口 + 跨档位只读汇总。"""

    def __init__(self, config) -> None:
        self.cfg = config
        self.logger = AISLogger()
        self.work_dir = config["work_dir"]

    def finalize_run(self, run_dir: str, manifest: Dict[str, Any],
                     raw_artifact_dir: str,
                     logs_dir: Optional[str] = None) -> Dict[str, Any]:
        """归档本档位产物。

        ``run_dir`` 为 ``performances/<run_id>``，直接承载 ``agentx_raw/``、上游
        summary 副本和 ``manifest.json``；summary 日志写入 ``logs/<run_id>/``。
        档位结束即落盘，中途失败时已完成档位的证据保持完整。
        """
        os.makedirs(run_dir, exist_ok=True)
        result = manifest.setdefault("result", {})
        source_json = osp.join(raw_artifact_dir, RAW_SUMMARY_NAME)
        source_csv = osp.join(raw_artifact_dir, RAW_SUMMARY_CSV_NAME)
        if osp.isfile(source_json):
            shutil.copyfile(source_json, osp.join(run_dir, SUMMARY_JSON_NAME))
            result["summary_json"] = SUMMARY_JSON_NAME
        else:
            result["summary_json"] = None
        if osp.isfile(source_csv):
            shutil.copyfile(source_csv, osp.join(run_dir, SUMMARY_CSV_NAME))
            result["summary_csv"] = SUMMARY_CSV_NAME
        else:
            result["summary_csv"] = None
        result["raw_artifacts"] = (
            osp.basename(raw_artifact_dir) if osp.isdir(raw_artifact_dir) else None
        )
        atomic_write_json(osp.join(run_dir, "manifest.json"), manifest)

        run_id = manifest.get("run_id") or osp.basename(run_dir)
        log_dir = logs_dir or osp.join(self.work_dir, LOGS_DIR, run_id)
        raw, _error = self._load_summary(run_dir, result.get("summary_json"))
        row = manifest_row(manifest, raw, self.work_dir, run_dir)
        self._write_summary_log(log_dir, row, raw_artifact_dir)
        # 每个档位跑完立即打印本档位结果，而不是等全部档位结束后统一打印。
        self.print_rows([row])
        return manifest

    def print_rows(self, rows: List[Dict[str, Any]]) -> None:
        """打印汇总表及逐档位明细；与 ``agentx_summary.log`` 使用同一份内容。"""
        print("Summarizing AgentX performance results...")
        print(render_table(rows))
        for row in rows:
            for line in self._row_detail_lines(row):
                print(line)

    def _row_detail_lines(self, row: Dict[str, Any]) -> List[str]:
        """逐档位明细：无效原因 + 关键产物路径，屏幕与 summary 日志共用。"""
        lines: List[str] = []
        if row.get("invalid_reasons"):
            lines.append(
                "invalid reasons: "
                + json.dumps(json_safe(row["invalid_reasons"]), ensure_ascii=False)
            )
        lines.append("")
        for key in ("run_id", "manifest_path", "raw_summary_path", "summary_csv_path"):
            if row.get(key):
                lines.append(f"{key}: {row[key]}")
        return lines

    def _load_summary(self, run_dir: str,
                      summary_json: Optional[str]) -> Tuple[Dict[str, Any], Optional[str]]:
        if not summary_json:
            return {}, None
        try:
            with open(osp.join(run_dir, summary_json), "r", encoding="utf-8") as file:
                raw = json.load(file)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return {}, str(exc)
        return (raw if isinstance(raw, dict) else {}), None

    def _write_summary_log(self, log_dir: str, row: Dict[str, Any],
                           raw_artifact_dir: Optional[str] = None) -> str:
        """逐档位 summary 日志：屏幕同款表格 + 明细 + 上游控制台输出。"""
        lines = [render_table([row])] + self._row_detail_lines(row)
        console_lines = self._console_log_lines(raw_artifact_dir)
        if console_lines:
            lines.append("")
            lines.extend(console_lines)
        path = osp.join(log_dir, SUMMARY_LOG_NAME)
        _atomic_text(path, "\n".join(lines) + "\n")
        return path

    def _console_log_lines(self, raw_artifact_dir: Optional[str]) -> List[str]:
        """读取 raw_artifact_dir 下的 console 日志，脱敏并去掉 NVIDIA AIPerf 品牌前缀。"""
        if not raw_artifact_dir:
            return []
        path = osp.join(raw_artifact_dir, RAW_CONSOLE_NAME)
        if not osp.isfile(path):
            return []
        # 与 agentx_stdout.log 同一脱敏函数；延后导入避免与 agentx_task 相互循环。
        from ais_bench.benchmark.tasks.custom_tasks.agentx_task import _redact_text
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as file:
                return [
                    _BRAND_PREFIX_RE.sub("", _redact_text(line.rstrip("\r\n"), []))
                    for line in file
                ]
        except OSError:
            return []

    def _discover(self) -> Tuple[List[str], List[str]]:
        """返回 (manifest 路径, 有产物但缺 manifest 的档位目录)。"""
        perf_root = Path(self.work_dir, PERFORMANCES_DIR)
        paths = sorted(str(path) for path in perf_root.glob("*/manifest.json"))
        with_manifest = {osp.dirname(path) for path in paths}
        missing: List[str] = []
        for run_dir in sorted(perf_root.glob("agentx_*")):
            if not run_dir.is_dir() or str(run_dir) in with_manifest:
                continue
            raw_dir = run_dir / "agentx_raw"
            if raw_dir.is_dir() and any(raw_dir.iterdir()):
                missing.append(str(run_dir))
        return paths, missing

    def summarize(self, time_str=None, print_table: bool = True) -> Dict[str, Any]:
        paths, missing_dirs = self._discover()
        if not paths and not missing_dirs:
            raise RuntimeError(
                "AGENTX_RESULT_MISSING: no manifest.json under "
                f"{self.work_dir}/{PERFORMANCES_DIR}/agentx_<variant>_batchsize_<N>"
            )
        manifests: List[Dict[str, Any]] = []
        rows: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        for run_dir in missing_dirs:
            # 档位目录有产物但 manifest 未写成（被 kill、磁盘满等）：显式降级，不静默丢档。
            manifest_path = osp.join(run_dir, "manifest.json")
            errors.append({"manifest_path": manifest_path, "error": "manifest_missing"})
            row = {
                "schema_version": None, "run_id": Path(run_dir).name,
                "status": "failed", "data_status": "unavailable",
                "benchmark_valid": False, "comparison_scope": "invalid",
                "invalid_reasons": ["manifest_missing"],
                "manifest_path": osp.relpath(manifest_path, self.work_dir),
            }
            rows.append(row)
            self._write_summary_log(osp.join(self.work_dir, LOGS_DIR, row["run_id"]), row)
        for path in paths:
            run_dir = osp.dirname(path)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    manifest = json.load(file)
                if not isinstance(manifest, dict) or not manifest.get("run_id"):
                    raise ValueError("unsupported AISBench AgentX manifest schema")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append({"manifest_path": path, "error": str(exc)})
                rows.append({
                    "schema_version": None, "run_id": Path(run_dir).name,
                    "status": "failed", "data_status": "unavailable",
                    "benchmark_valid": False, "comparison_scope": "invalid",
                    "invalid_reasons": [f"manifest_load_failed:{exc}"],
                    "manifest_path": osp.relpath(path, self.work_dir),
                })
                continue
            result = manifest.get("result") or {}
            raw, summary_error = self._load_summary(run_dir, result.get("summary_json"))
            summary_reason = None
            if summary_error:
                summary_reason = f"summary_load_failed:{summary_error}"
                errors.append({"manifest_path": path, "summary_error": summary_error})
            elif not raw:
                # summary 缺失或内容为空时降级该行，不能显示 benchmark_valid=True 但指标全 N/A。
                summary_reason = "summary_missing" if not result.get("summary_json") else "summary_empty"
                errors.append({"manifest_path": path, "summary_error": summary_reason})
            manifests.append(manifest)
            row = manifest_row(manifest, raw, self.work_dir, run_dir)
            if summary_reason:
                row["data_status"] = "unavailable"
                row["benchmark_valid"] = False
                if row.get("comparison_scope") == "official_agentx":
                    row["comparison_scope"] = "invalid"
                if not row.get("invalid_reasons"):
                    row["invalid_reasons"] = [summary_reason]
            rows.append(row)
            # viz 重建时同步刷新各档位 summary 日志，保证屏幕与落盘口径一致。
            raw_artifact_dir = (
                osp.join(run_dir, result["raw_artifacts"]) if result.get("raw_artifacts") else None
            )
            self._write_summary_log(osp.join(self.work_dir, LOGS_DIR, row["run_id"]), row, raw_artifact_dir)

        if print_table:
            self.print_rows(rows)
        counts = {
            "runs": len(rows),
            "completed": sum(row.get("status") == "completed" for row in rows),
            "failed": sum(row.get("status") == "failed" for row in rows),
            "cancelled": sum(row.get("status") == "cancelled" for row in rows),
            "benchmark_valid": sum(row.get("benchmark_valid") is True for row in rows),
            "result_load_errors": len(errors),
        }
        payload = {
            "schema_version": "aisbench_agentx_summary/v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "counts": counts, "input_files": paths, "runs": manifests, "errors": errors,
        }
        self.logger.info("AgentX summary printed for %s run(s)", len(rows))
        if errors:
            raise RuntimeError(
                f"AGENTX_RESULT_INVALID: {len(errors)} manifest/summary file(s) could not be loaded; "
                "partial reports were retained."
            )
        return payload
