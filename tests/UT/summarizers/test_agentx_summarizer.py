import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mmengine.config import ConfigDict

from ais_bench.benchmark.summarizers.agentx import (
    AgentXSummarizer,
    manifest_row,
)
from ais_bench.benchmark.tasks.custom_tasks.agentx_result import atomic_write_json


def make_raw():
    return {
        "schema_version": "1.4",
        "request_count": {"unit": "requests", "avg": 10},
        "error_request_count": {"unit": "requests", "avg": 0},
        "request_throughput": {"unit": "requests/s", "avg": 8},
        "output_token_throughput": {"unit": "tokens/s", "avg": 80},
        "time_to_first_token": {"unit": "s", "avg": 0.1, "p90": 0.2},
        "inter_token_latency": {"unit": "ms", "avg": 12},
        "request_latency": {"unit": "us", "avg": 1000, "p90": 2000},
        "metadata": {
            "submission_valid": True,
            "submission_invalid_reasons": [],
            "metric_duration_coverage": [{
                "required_ratio": 0.9,
                "ttft_ratio": 1.0,
                "inter_token_latency_ratio": 0.95,
            }],
        },
    }


def make_manifest(run_id="agentx_full_batchsize_8", *, model="qwen3.8",
                  dataset="weka"):
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": "2026-01-01T00:00:00+00:00",
        "upstream": {"framework": "agentx-harness", "version": "1.0.13"},
        "run_params": {"run_mode": "benchmark", "benchmark_id": f"{model}/{dataset}"},
        "dataset": {"abbr": dataset, "variant": "full", "source": "public"},
        "model": {"name": "=1+1", "abbr": "qwen,\"x\"\nnext", "path": "/models/qwen"},
        "engine": {"type": "vllm", "version": "0.8.5"},
        "test_params": {"batch_size": 8, "random_seed": 42},
        "result": {
            "status": "completed", "parse_status": "success", "data_status": "complete",
            "benchmark_valid": True, "comparison_scope": "official_agentx",
            "harness_submission_valid": True, "invalid_reasons": [],
            "aisbench_invalid_reasons": [], "errors": [],
            "summary_json": "agentx_summary.json",
            "summary_csv": "agentx_summary.csv",
            "raw_artifacts": "agentx_raw",
        },
    }


def write_run(work_dir, manifest=None):
    manifest = manifest or make_manifest()
    run_dir = Path(work_dir) / "performances" / manifest["run_id"]
    (run_dir / "agentx_raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "agentx_summary.json").write_text(
        json.dumps(make_raw()), encoding="utf-8"
    )
    (run_dir / "agentx_summary.csv").write_text(
        "metric,avg\nrequest_throughput,8\n", encoding="utf-8"
    )
    atomic_write_json(str(run_dir / "manifest.json"), manifest)
    return run_dir


class TestAgentXSummaryMapping(unittest.TestCase):
    def test_manifest_row_passes_through_raw_metric_values(self):
        with tempfile.TemporaryDirectory() as work_dir:
            manifest = make_manifest()
            run_dir = str(Path(work_dir) / "performances" / manifest["run_id"])
            row = manifest_row(manifest, make_raw(), work_dir, run_dir)
            self.assertEqual(row["run_id"], "agentx_full_batchsize_8")
            self.assertEqual(row["request_throughput"], 8)
            self.assertEqual(row["error_request_count"], 0)
            # 时延列直接透传上游 p90 原始值和原始单位，不做换算。
            self.assertEqual(row["time_to_first_token_p90"], 0.2)
            self.assertEqual(row["request_latency_p90"], 2000)
            self.assertIsNone(row["inter_token_latency_p90"])
            self.assertTrue(row["metric_coverage_passed"])
            self.assertEqual(
                row["manifest_path"], "performances/agentx_full_batchsize_8/manifest.json"
            )
            self.assertEqual(
                row["raw_summary_path"],
                "performances/agentx_full_batchsize_8/agentx_summary.json",
            )
            self.assertEqual(
                row["summary_csv_path"],
                "performances/agentx_full_batchsize_8/agentx_summary.csv",
            )


class TestAgentXSummarizer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.work_dir = Path(self.temp_dir.name)
        self.summarizer = AgentXSummarizer(ConfigDict({"work_dir": str(self.work_dir)}))

    def test_finalize_run_writes_slot_artifacts_and_summary_log(self):
        run_dir = self.work_dir / "performances" / "agentx_full_batchsize_8"
        raw_dir = run_dir / "agentx_raw"
        raw_dir.mkdir(parents=True)
        (raw_dir / "profile_export_aiperf.json").write_text(
            json.dumps(make_raw()), encoding="utf-8"
        )
        (raw_dir / "profile_export_aiperf.csv").write_text(
            "metric,avg\nrequest_throughput,8\n", encoding="utf-8"
        )
        logs_dir = self.work_dir / "logs" / "agentx_full_batchsize_8"
        manifest = make_manifest()
        manifest["result"]["summary_json"] = None
        manifest["result"]["summary_csv"] = None
        manifest["result"]["raw_artifacts"] = None
        self.summarizer.finalize_run(str(run_dir), manifest, str(raw_dir), str(logs_dir))

        self.assertEqual(
            (run_dir / "agentx_summary.json").read_bytes(),
            (raw_dir / "profile_export_aiperf.json").read_bytes(),
        )
        self.assertEqual(
            (run_dir / "agentx_summary.csv").read_bytes(),
            (raw_dir / "profile_export_aiperf.csv").read_bytes(),
        )
        written = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(written["result"]["summary_json"], "agentx_summary.json")
        self.assertEqual(written["result"]["summary_csv"], "agentx_summary.csv")
        self.assertEqual(written["result"]["raw_artifacts"], "agentx_raw")

        # 档位结束即写本档位 summary 日志，且不产生额外目录。
        summary_log = (logs_dir / "agentx_summary.log").read_text(encoding="utf-8")
        self.assertIn("agentx_full_batchsize_8", summary_log)
        self.assertIn(
            "performances/agentx_full_batchsize_8/agentx_summary.csv", summary_log
        )
        self.assertFalse((run_dir / "performances").exists())
        self.assertFalse((run_dir / "configs").exists())

    def test_summarize_reads_manifests_and_refreshes_slot_logs(self):
        write_run(self.work_dir)
        with mock.patch("builtins.print") as output:
            payload = self.summarizer.summarize()
        self.assertEqual(payload["counts"]["runs"], 1)
        self.assertEqual(payload["counts"]["benchmark_valid"], 1)
        self.assertTrue(output.called)

        # 根目录不再生成跨档位汇总文件，档位信息由 manifest 和 summary 日志承载。
        self.assertFalse((self.work_dir / "agentx_summary.csv").exists())
        self.assertFalse((self.work_dir / "agentx_summary.json").exists())
        self.assertFalse((self.work_dir / "summary").exists())
        summary_log = (
            self.work_dir / "logs" / "agentx_full_batchsize_8" / "agentx_summary.log"
        ).read_text(encoding="utf-8")
        self.assertIn("agentx_full_batchsize_8", summary_log)
        self.assertIn("performances/agentx_full_batchsize_8/agentx_summary.json", summary_log)

    def test_partial_corrupt_manifest_retains_other_reports(self):
        good_dir = write_run(self.work_dir)
        bad_dir = self.work_dir / "performances" / "agentx_full_batchsize_16"
        bad_dir.mkdir(parents=True)
        (bad_dir / "manifest.json").write_text("{bad", encoding="utf-8")
        with mock.patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "AGENTX_RESULT_INVALID"):
                self.summarizer.summarize()
        self.assertTrue((bad_dir / "manifest.json").exists())
        self.assertTrue((good_dir / "manifest.json").exists())
        self.assertTrue(
            (
                self.work_dir / "logs" / "agentx_full_batchsize_8" / "agentx_summary.log"
            ).is_file()
        )

    def test_no_manifests_is_explicit_error(self):
        with self.assertRaisesRegex(RuntimeError, "AGENTX_RESULT_MISSING"):
            self.summarizer.summarize()


if __name__ == "__main__":
    unittest.main()
