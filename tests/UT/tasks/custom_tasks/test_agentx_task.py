import http.client
import json
import os
import os.path as osp
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from mmengine.config import ConfigDict

from ais_bench.benchmark.tasks.custom_tasks.agentx_task import (
    AGENTX_SUMMARY_NAME,
    DATASETS,
    IMAGE_HARNESS_PYTHON,
    PROCESS_TIMEOUT_ENV,
    AgentXTask,
    _extra_arg_secrets,
    _http_json,
    _process_timeout,
    build_agentx_command,
    probe_harness_environment,
    redact_command,
    resolve_agentx_bin,
    resolve_cooldown,
    resolve_harness_python,
    resolve_smoke_overrides,
    resolve_variant,
    sha256_file,
    validate_agentx_args,
)


class AgentXTaskTestBase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        self.work_dir = self.root / "work"
        self.work_dir.mkdir()

    def model_cfg(self, **overrides):
        model = {
            "abbr": "qwen",
            "model": "qwen3.8",
            "url": "http://127.0.0.1:8000",
            "path": str(self.model_dir),
        }
        model.update(overrides)
        return model

    def dataset_args(self, **overrides):
        args = {"concurrency": 8, "variant": "full"}
        args.update(overrides)
        return args

    def task(self, nested=False, **dataset_overrides):
        dataset = {"abbr": "weka", "args": self.dataset_args(**dataset_overrides)}
        datasets = [[dataset]] if nested else [dataset]
        return AgentXTask(ConfigDict({
            "work_dir": str(self.work_dir),
            "models": [self.model_cfg()],
            "datasets": datasets,
            "cli_args": {"debug": False},
        }))

    def harness_metadata(self):
        return {
            "distribution": "agentx-harness",
            "version": "1.0.13",
            "agentx_version": "1.0.13",
            "python_version": "3.11.9",
            "python": "python",
            "entry": ["agentx"],
            "platform": "test",
        }


class TestAgentXValidation(AgentXTaskTestBase):
    def test_variant_must_be_explicit(self):
        self.assertEqual(resolve_variant("full"), "full")
        self.assertEqual(resolve_variant("256k"), "256k")
        for value in (None, "", "  ", "other"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    resolve_variant(value)
        with self.assertRaisesRegex(ValueError, "explicit dataset variant"):
            validate_agentx_args(self.model_cfg(), {"concurrency": 8})
        with self.assertRaisesRegex(ValueError, "Unknown AgentX dataset variant"):
            validate_agentx_args(self.model_cfg(), self.dataset_args(variant="other"))

    def test_concurrency_levels(self):
        self.assertEqual(validate_agentx_args(self.model_cfg(), self.dataset_args(concurrency=8)), [8])
        self.assertEqual(validate_agentx_args(self.model_cfg(), self.dataset_args(concurrency=[8])), [8])
        self.assertEqual(
            validate_agentx_args(
                self.model_cfg(), self.dataset_args(concurrency=[1, 8, 32])
            ),
            [1, 8, 32],
        )

    def test_invalid_concurrency_rejected_before_execution(self):
        for value in (0, -1, 1.5, True, "8", [], [8, 8], [1, 0]):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_agentx_args(self.model_cfg(), self.dataset_args(concurrency=value))

    def test_cooldown_default_and_validation(self):
        for value in (None, 0, 5, 2.5):
            with self.subTest(value=value):
                validate_agentx_args(self.model_cfg(), self.dataset_args(cooldown=value))
        self.assertEqual(resolve_cooldown(None), 5.0)
        self.assertEqual(resolve_cooldown(2), 2.0)
        for value in (-1, True, "5", float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "cooldown must be"):
                    validate_agentx_args(self.model_cfg(), self.dataset_args(cooldown=value))

    def test_required_model_url_and_tokenizer(self):
        with self.assertRaisesRegex(ValueError, "non-empty model name"):
            validate_agentx_args(self.model_cfg(model=""), self.dataset_args())
        with self.assertRaisesRegex(ValueError, "base URL"):
            validate_agentx_args(self.model_cfg(url=None), self.dataset_args())
        with self.assertRaisesRegex(ValueError, "base URL without"):
            validate_agentx_args(
                self.model_cfg(url="http://127.0.0.1:8000/v1/chat/completions"),
                self.dataset_args(),
            )
        for endpoint in ("", "v1/chat/completions", "/v1/chat?x=1", "/v1/chat#frag", "/v1 chat"):
            with self.subTest(endpoint=endpoint):
                with self.assertRaisesRegex(ValueError, "endpoint must be"):
                    validate_agentx_args(
                        self.model_cfg(endpoint=endpoint), self.dataset_args()
                    )
        validate_agentx_args(
            self.model_cfg(endpoint="/custom/v2/chat"), self.dataset_args()
        )
        with self.assertRaisesRegex(ValueError, "base URL without /custom/v2/chat"):
            validate_agentx_args(
                self.model_cfg(
                    url="http://127.0.0.1:8000/custom/v2/chat",
                    endpoint="/custom/v2/chat",
                ),
                self.dataset_args(),
            )
        with self.assertRaisesRegex(ValueError, r"models\[0\].path"):
            validate_agentx_args(self.model_cfg(path=None), self.dataset_args())
        with self.assertRaises(FileNotFoundError):
            validate_agentx_args(
                self.model_cfg(path=str(self.root / "missing")), self.dataset_args()
            )

    def test_smoke_duration_and_trace_limit_policy(self):
        args = self.dataset_args(run_mode="smoke", duration=2.5, trace_limit=3)
        self.assertEqual(validate_agentx_args(self.model_cfg(), args), [8])
        validate_agentx_args(
            self.model_cfg(), self.dataset_args(run_mode="benchmark", duration=1800)
        )
        for key, value, message in (
            ("duration", 2, ">= 900"),
            ("trace_limit", 3, "only valid in smoke"),
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, message):
                    validate_agentx_args(self.model_cfg(), self.dataset_args(**{key: value}))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            validate_agentx_args(
                self.model_cfg(), self.dataset_args(run_mode="smoke", trace_limit=1.5)
            )

    def test_extra_args_passthrough(self):
        extra_args = (
            "--log-level INFO --verbose --benchmark-duration 901 "
            "--random-seed 7 --concurrency 2 --scenario agentx --model Qwen"
        )
        validate_agentx_args(self.model_cfg(), self.dataset_args(extra_args=extra_args))
        command = build_agentx_command(
            ["agentx"],
            url=self.model_cfg()["url"],
            model="qwen3.8",
            concurrency=8,
            dataset_alias=DATASETS["full"]["alias"],
            artifact_dir="/out/run/artifacts",
            extra_args=extra_args,
        )
        # 重复参数以 extra args 为准，基础命令中的同名选项被移除
        self.assertEqual(command.count("--model"), 1)
        self.assertEqual(command[command.index("--model") + 1], "Qwen")
        self.assertEqual(command.count("--concurrency"), 1)
        self.assertEqual(command[command.index("--concurrency") + 1], "2")
        self.assertEqual(command.count("--scenario"), 1)
        self.assertNotIn("qwen3.8", command)
        # 未被覆盖的参数保持原值，extra args 整体追加在末尾
        self.assertEqual(command[command.index("--url") + 1], self.model_cfg()["url"])
        self.assertEqual(command[-13:], [
            "--log-level", "INFO", "--verbose", "--benchmark-duration", "901",
            "--random-seed", "7", "--concurrency", "2", "--scenario", "agentx",
            "--model", "Qwen",
        ])
        braced = build_agentx_command(
            ["agentx"],
            url=self.model_cfg()["url"],
            model="qwen3.8",
            concurrency=8,
            dataset_alias=DATASETS["full"]["alias"],
            artifact_dir="/out/run/artifacts",
            extra_args="{ --concurrency 1 --model Qwen }",
        )
        self.assertEqual(braced[-4:], ["--concurrency", "1", "--model", "Qwen"])
        for item in ("", "  ", [], ["--log-level=INFO"]):
            with self.subTest(item=item):
                with self.assertRaises(ValueError):
                    validate_agentx_args(self.model_cfg(), self.dataset_args(extra_args=item))

    def test_environment_validation(self):
        with self.assertRaisesRegex(ValueError, "KEY=VALUE|env variable"):
            validate_agentx_args(self.model_cfg(), self.dataset_args(env={"BAD-KEY": "x"}))
        with self.assertRaisesRegex(ValueError, "must not override"):
            validate_agentx_args(self.model_cfg(), self.dataset_args(env={"AIPERF_SCENARIO": "x"}))


class TestAgentXCommand(AgentXTaskTestBase):
    def test_public_benchmark_command_golden(self):
        command = build_agentx_command(
            ["agentx"],
            url=self.model_cfg()["url"] + "/",
            model="qwen3.8",
            concurrency=8,
            dataset_alias=DATASETS["full"]["alias"],
            artifact_dir="/out/run/artifacts",
            tokenizer_path=str(self.model_dir),
            tokenizer_trust_remote_code=True,
            api_key="secret",
            extra_args="--log-level=INFO",
        )
        self.assertEqual(command[:3], ["agentx", "profile", "--scenario"])
        self.assertEqual(command[3], "agentx")
        self.assertEqual(command[command.index("--url") + 1], "http://127.0.0.1:8000")
        self.assertEqual(command[command.index("--endpoint") + 1], "/v1/chat/completions")
        self.assertEqual(command[command.index("--concurrency") + 1], "8")
        self.assertEqual(command[command.index("--tokenizer") + 1], str(self.model_dir))
        self.assertIn("--tokenizer-trust-remote-code", command)
        self.assertIn("--public-dataset", command)
        self.assertNotIn("--hf-weka-dataset", command)
        self.assertNotIn("--input-file", command)
        for forbidden in ("--benchmark-duration", "--num-dataset-entries", "--seed", "--random-seed"):
            self.assertNotIn(forbidden, command)

    def test_custom_endpoint_is_forwarded(self):
        command = build_agentx_command(
            ["agentx"],
            url=self.model_cfg()["url"],
            model="qwen3.8",
            concurrency=8,
            dataset_alias=DATASETS["full"]["alias"],
            artifact_dir="/out/custom",
            endpoint="/custom/v2/chat",
        )
        self.assertEqual(command[command.index("--endpoint") + 1], "/custom/v2/chat")
        self.assertEqual(command[command.index("--url") + 1], "http://127.0.0.1:8000")

    def test_local_and_smoke_command_golden(self):
        dataset_dir = self.root / "dataset"
        dataset_dir.mkdir()
        command = build_agentx_command(
            ["python", "-m", "agentx_cli"],
            url=self.model_cfg()["url"],
            model="qwen3.8",
            concurrency=1,
            dataset_alias=None,
            dataset_dir=str(dataset_dir),
            artifact_dir="/out/smoke",
            run_mode="smoke",
            duration=2,
            trace_limit=4,
        )
        self.assertEqual(command[command.index("--hf-weka-dataset") + 1], str(dataset_dir))
        self.assertIn("--unsafe-override", command)
        self.assertNotIn("--public-dataset", command)
        self.assertEqual(command[command.index("--benchmark-duration") + 1], "2")
        self.assertEqual(command[command.index("--num-dataset-entries") + 1], "4")

    def test_benchmark_duration_override_is_forwarded(self):
        command = build_agentx_command(
            ["agentx"],
            url=self.model_cfg()["url"],
            model="qwen3.8",
            concurrency=8,
            dataset_alias=DATASETS["full"]["alias"],
            artifact_dir="/out/benchmark",
            run_mode="benchmark",
            duration=1800.0,
        )
        self.assertEqual(command[command.index("--benchmark-duration") + 1], "1800.0")
        self.assertNotIn("--unsafe-override", command)
        self.assertNotIn("--num-dataset-entries", command)

    def test_command_redaction(self):
        command = ["agentx", "--api-key", "secret", "--api-key=other", "--model", "m"]
        redacted = redact_command(command)
        self.assertNotIn("secret", redacted)
        self.assertNotIn("other", redacted)
        self.assertIn("--api-key=<redacted>", redacted)
        self.assertEqual(redacted[-1], "m")

    def test_command_redaction_does_not_forge_tail_value(self):
        # --api-key 位于末尾时没有取值，不能凭空补一个 <redacted>。
        self.assertEqual(redact_command(["agentx", "--api-key"]), ["agentx", "--api-key"])

    def test_tokenizer_path_is_expanded(self):
        command = build_agentx_command(
            ["agentx"], url="http://127.0.0.1:8000", model="m", concurrency=1,
            dataset_alias=DATASETS["full"]["alias"], artifact_dir="/out/benchmark",
            tokenizer_path="~/tokenizer",
        )
        self.assertEqual(command[command.index("--tokenizer") + 1], osp.expanduser("~/tokenizer"))

    def test_process_timeout_env(self):
        self.assertIsNone(_process_timeout({}))
        self.assertEqual(_process_timeout({PROCESS_TIMEOUT_ENV: "1800"}), 1800.0)
        for bad in ("0", "-5", "abc", "nan"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    _process_timeout({PROCESS_TIMEOUT_ENV: bad})

    def test_extra_arg_secrets_collect_known_sensitive_values(self):
        self.assertEqual(
            _extra_arg_secrets("--api-key k1 --api-key=k2 --log-level INFO"),
            ["k1", "k2"],
        )
        self.assertEqual(_extra_arg_secrets(None), [])
        self.assertEqual(_extra_arg_secrets("--log-level INFO"), [])

    def test_http_json_degrades_http_exceptions_to_error(self):
        # IncompleteRead 等 HTTPException 不是 URLError/OSError/ValueError，
        # 必须同样降级为探测错误，不能逃逸吞掉整档结果。
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=http.client.IncompleteRead(b"partial"),
        ):
            payload, error = _http_json("http://127.0.0.1:8000/version")
        self.assertIsNone(payload)
        self.assertIn("IncompleteRead", error)


class TestAgentXDatasetAndEnvironment(AgentXTaskTestBase):
    def test_local_hash_match_and_mismatch_policy(self):
        dataset_dir = self.root / "dataset"
        dataset_dir.mkdir()
        trace = dataset_dir / "traces.jsonl"
        trace.write_bytes(b'{"session": 1}\n')
        actual = sha256_file(str(trace))
        task = self.task(dataset_path=str(dataset_dir))
        with mock.patch.dict(DATASETS["full"], {"sha256": actual}):
            dataset = task._resolve_dataset("benchmark")
        self.assertEqual(dataset["actual_sha256"], actual)
        self.assertEqual(dataset["verification"], ["local_content_hash_verified"])
        self.assertFalse(dataset["sha256_mismatch"])

        with self.assertRaisesRegex(ValueError, "AGENTX_DATASET_HASH_MISMATCH"):
            task._resolve_dataset("benchmark")
        smoke = self.task(dataset_path=str(dataset_dir), run_mode="smoke")
        dataset = smoke._resolve_dataset("smoke")
        self.assertTrue(dataset["sha256_mismatch"])
        self.assertEqual(dataset["actual_sha256"], actual)
        self.assertEqual(dataset["expected_sha256"], DATASETS["full"]["sha256"])

    def test_probe_accepts_supported_environment(self):
        probe = SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "python_version": "3.11.9",
                "agentx_version": "1.0.13",
                "upstream_aiperf": False,
            }),
            stderr="",
        )
        help_result = SimpleNamespace(returncode=0, stdout="help", stderr="")
        with mock.patch("subprocess.run", side_effect=[probe, help_result]) as run:
            metadata = probe_harness_environment("python", {}, agentx_bin="agentx")
        self.assertEqual(metadata["version"], "1.0.13")
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            # 子进程一律按 UTF-8 解码，非 UTF-8 locale 下上游输出不会中断整档测评。
            self.assertEqual(call.kwargs.get("encoding"), "utf-8")
            self.assertEqual(call.kwargs.get("errors"), "replace")

    def test_probe_rejects_python_version_harness_version_and_aiperf(self):
        cases = (
            ({"python_version": "3.10.14", "agentx_version": "1.0.13", "upstream_aiperf": False},
             "Python 3.11~3.13"),
            ({"python_version": "3.11.9", "agentx_version": "1.0.12", "upstream_aiperf": False},
             "AGENTX_VERSION_UNSUPPORTED"),
            ({"python_version": "3.11.9", "agentx_version": "1.0.13", "upstream_aiperf": True},
             "upstream PyPI aiperf"),
        )
        for metadata, message in cases:
            with self.subTest(message=message):
                completed = SimpleNamespace(
                    returncode=0, stdout=json.dumps(metadata), stderr=""
                )
                with mock.patch("subprocess.run", return_value=completed):
                    with self.assertRaisesRegex(RuntimeError, message):
                        probe_harness_environment("python", {}, agentx_bin="agentx")

    def test_explicit_python_and_environment_variable_resolution(self):
        environ = {"AISBENCH_AGENTX_PYTHON": "/env/agentx/bin/python"}
        self.assertEqual(
            resolve_harness_python({"agentx_python": "~/agentx/bin/python"}, environ),
            osp.expanduser("~/agentx/bin/python"),
        )
        self.assertEqual(resolve_harness_python({}, environ), "/env/agentx/bin/python")

    def test_image_and_current_interpreter_fallbacks(self):
        with mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.osp.isfile", return_value=True
        ):
            self.assertEqual(resolve_harness_python({}, {}), IMAGE_HARNESS_PYTHON)

        with mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.osp.isfile", return_value=False
        ):
            self.assertEqual(resolve_harness_python({}, {}), sys.executable)

    def test_agentx_entry_stays_with_selected_python(self):
        bin_dir = self.root / "venv" / "bin"
        bin_dir.mkdir(parents=True)
        python_exe = bin_dir / "python"
        python_exe.touch()
        entry = bin_dir / "agentx"
        entry.touch()
        self.assertEqual(
            resolve_agentx_bin(None, str(python_exe)), [str(entry.resolve())]
        )
        entry.unlink()
        self.assertEqual(
            resolve_agentx_bin(None, str(python_exe)),
            [str(python_exe), "-m", "agentx_cli"],
        )
        self.assertEqual(
            resolve_agentx_bin("/custom/agentx", str(python_exe)), ["/custom/agentx"]
        )
        # 校验阶段按 expanduser 判定存在，执行阶段必须同样展开。
        self.assertEqual(
            resolve_agentx_bin("~/agentx", str(python_exe)), [osp.expanduser("~/agentx")]
        )


class TestAgentXTaskLifecycle(AgentXTaskTestBase):
    def test_direct_and_legacy_nested_dataset_shapes(self):
        direct = self.task(nested=False)
        legacy = self.task(nested=True)
        self.assertEqual(direct.dataset_cfg["abbr"], "weka")
        self.assertEqual(legacy.dataset_cfg["abbr"], "weka")
        self.assertIsNone(direct.run_id)

    def test_dry_run_writes_redacted_plan_without_popen(self):
        task = self.task(api_key="secret")
        with mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.resolve_harness_python",
            return_value="python",
        ), mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.probe_harness_environment",
            return_value=self.harness_metadata(),
        ), mock.patch("subprocess.Popen") as popen, mock.patch.object(
            task, "_execute_plan"
        ) as execute:
            plan = task.dry_run()
        popen.assert_not_called()
        execute.assert_not_called()
        run_id = "agentx_full_batchsize_8"
        plan_path = self.work_dir / "performances" / run_id / "execution_plan.redacted.json"
        self.assertTrue(plan_path.is_file())
        self.assertFalse((self.work_dir / "configs" / run_id).exists())
        content = plan_path.read_text(encoding="utf-8")
        self.assertNotIn("secret", content)
        self.assertIn("<redacted>", content)
        self.assertEqual(plan["runs"][0]["run_id"], "agentx_full_batchsize_8")

    def test_reuse_clears_previous_run_artifacts(self):
        """--reuse 只复用时间戳目录：清空该档位旧产物后重跑，不因目录非空失败"""
        run_id = "agentx_full_batchsize_8"
        artifact_dir = self.work_dir / "performances" / run_id / "agentx_raw"
        artifact_dir.mkdir(parents=True)
        stale = artifact_dir / "summary_export.json"
        stale.write_text("{}", encoding="utf-8")
        logs_dir = self.work_dir / "logs" / run_id
        logs_dir.mkdir(parents=True)
        (logs_dir / "agentx_summary.log").write_text("stale", encoding="utf-8")

        task = AgentXTask(ConfigDict({
            "work_dir": str(self.work_dir),
            "models": [self.model_cfg()],
            "datasets": [{"abbr": "weka", "args": self.dataset_args()}],
            "cli_args": {"debug": False, "reuse": "latest"},
        }))
        completed = {
            "run": {
                "status": "completed", "parse_status": "success",
                "benchmark_valid": True, "comparison_scope": "official_agentx",
            },
            "artifacts": {"harness_log": "agentx_stdout.log"},
        }
        with mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.resolve_harness_python",
            return_value="python",
        ), mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.probe_harness_environment",
            return_value=self.harness_metadata(),
        ), mock.patch.object(task, "_execute_plan", return_value=completed):
            task.run()

        self.assertFalse(stale.exists())
        self.assertTrue(artifact_dir.is_dir())
        self.assertFalse((logs_dir / "agentx_summary.log").exists())

    def test_run_rejects_existing_artifacts_without_reuse(self):
        run_id = "agentx_full_batchsize_8"
        artifact_dir = self.work_dir / "performances" / run_id / "agentx_raw"
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "summary_export.json").write_text("{}", encoding="utf-8")
        task = self.task()
        with mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.resolve_harness_python",
            return_value="python",
        ), mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.probe_harness_environment",
            return_value=self.harness_metadata(),
        ):
            with self.assertRaisesRegex(RuntimeError, "AGENTX_OUTPUT_WRITE_FAILED"):
                task.run()

    def test_cancelled_run_does_not_record_process_failure(self):
        """取消时退出码来自我方 kill，不应同时记 AGENTX_PROCESS_FAILED。"""
        task = self.task()
        with mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.resolve_harness_python",
            return_value="python",
        ), mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.probe_harness_environment",
            return_value=self.harness_metadata(),
        ):
            plan = task._prepare()[0]
        with mock.patch.object(task, "_execute", return_value=(-9, True, None, None)), \
                mock.patch.object(
                    task, "_read_summary", return_value=({}, "missing", "no summary")), \
                mock.patch.object(task, "_probe_engine", return_value=(None, [])):
            result = task._execute_plan(plan)

        codes = [item["code"] for item in result["errors"]]
        self.assertNotIn("AGENTX_PROCESS_FAILED", codes)
        self.assertIn("AGENTX_CANCELLED", codes)

    def test_smoke_defaults_are_applied_when_unset(self):
        self.assertEqual(resolve_smoke_overrides("smoke", None, None), (60.0, 4))
        self.assertEqual(resolve_smoke_overrides("smoke", 2.5, 3), (2.5, 3))
        self.assertEqual(resolve_smoke_overrides("benchmark", None, None), (None, None))

        def dry_run_plan(**dataset_overrides):
            task = self.task(**dataset_overrides)
            with mock.patch(
                "ais_bench.benchmark.tasks.custom_tasks.agentx_task.resolve_harness_python",
                return_value="python",
            ), mock.patch(
                "ais_bench.benchmark.tasks.custom_tasks.agentx_task.probe_harness_environment",
                return_value=self.harness_metadata(),
            ):
                return task.dry_run()["runs"][0]

        smoke = dry_run_plan(run_mode="smoke")
        self.assertEqual(smoke["duration"], 60.0)
        self.assertEqual(smoke["trace_limit"], 4)
        self.assertEqual(
            smoke["argv"][smoke["argv"].index("--benchmark-duration") + 1], "60.0"
        )
        self.assertEqual(
            smoke["argv"][smoke["argv"].index("--num-dataset-entries") + 1], "4"
        )

        benchmark = dry_run_plan(run_mode="benchmark")
        self.assertIsNone(benchmark["duration"])
        self.assertIsNone(benchmark["trace_limit"])
        self.assertNotIn("--benchmark-duration", benchmark["argv"])
        self.assertNotIn("--num-dataset-entries", benchmark["argv"])

    def test_multiple_concurrency_levels_run_serially_with_fresh_dirs(self):
        task = self.task(concurrency=[1, 8, 32], cooldown=7)
        calls = []

        def fake_execute(plan):
            calls.append(plan)
            return {
                "run": {"status": "completed", "parse_status": "success",
                        "benchmark_valid": True, "comparison_scope": "official_agentx"},
                "artifacts": {"manifest": plan["manifest_path"],
                              "harness_log": plan["log_path"]},
            }

        with mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.resolve_harness_python",
            return_value="python",
        ), mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.probe_harness_environment",
            return_value=self.harness_metadata(),
        ), mock.patch.object(task, "_execute_plan", side_effect=fake_execute), mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.time.sleep"
        ) as sleep:
            result = task.run()

        self.assertEqual([plan["concurrency"] for plan in calls], [1, 8, 32])
        self.assertEqual(len({plan["run_id"] for plan in calls}), 3)
        self.assertEqual(len({plan["artifact_dir"] for plan in calls}), 3)
        for plan in calls:
            argv = plan["argv"]
            self.assertEqual(argv[argv.index("--concurrency") + 1], str(plan["concurrency"]))
            self.assertEqual(Path(plan["plan_path"]).is_file(), True)
            self.assertNotIn("run_config_path", plan)
        self.assertEqual(
            [plan["run_id"] for plan in calls],
            ["agentx_full_batchsize_1", "agentx_full_batchsize_8", "agentx_full_batchsize_32"],
        )
        self.assertEqual(sleep.call_args_list, [mock.call(7.0), mock.call(7.0)])
        self.assertEqual([item["run"]["status"] for item in result["runs"]], ["completed"] * 3)

    def test_start_failure_writes_structured_redacted_result(self):
        task = self.task(api_key="secret")
        with mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.resolve_harness_python",
            return_value="python",
        ), mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.probe_harness_environment",
            return_value=self.harness_metadata(),
        ), mock.patch.object(
            task, "_probe_engine", return_value=({"type": None}, [])
        ), mock.patch.object(task, "_execute", return_value=(None, False, "spawn secret failed")):
            with self.assertRaisesRegex(RuntimeError, "start_failed/missing"):
                task.run()
        manifest_path = (
            self.work_dir / "performances" / "agentx_full_batchsize_8" / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_id"], "agentx_full_batchsize_8")
        self.assertEqual(manifest["result"]["status"], "start_failed")
        self.assertEqual(manifest["result"]["parse_status"], "missing")
        self.assertFalse(manifest["result"]["benchmark_valid"])
        self.assertEqual(manifest["result"]["errors"][0]["code"], "AGENTX_PROCESS_START_FAILED")
        self.assertIsNone(manifest["result"]["summary_json"])
        self.assertEqual(manifest["engine"], {"type": None})
        self.assertNotIn("secret", manifest_path.read_text(encoding="utf-8"))

    def test_summary_missing_corrupt_schema_and_ambiguity(self):
        task = self.task()
        artifact = self.root / "artifacts"
        artifact.mkdir()
        expected = artifact / AGENTX_SUMMARY_NAME
        raw, status, error = task._read_summary(str(artifact), str(expected))
        self.assertEqual((raw, status), ({}, "missing"))
        self.assertIn("missing", error)

        expected.write_text("{bad", encoding="utf-8")
        _, status, _ = task._read_summary(str(artifact), str(expected))
        self.assertEqual(status, "invalid_json")

        expected.write_text(json.dumps({"schema_version": "1.3"}), encoding="utf-8")
        _, status, _ = task._read_summary(str(artifact), str(expected))
        self.assertEqual(status, "unsupported_schema")

        expected.write_text(json.dumps({"schema_version": "1.4"}), encoding="utf-8")
        nested = artifact / "nested"
        nested.mkdir()
        (nested / AGENTX_SUMMARY_NAME).write_text("{}", encoding="utf-8")
        _, status, error = task._read_summary(str(artifact), str(expected))
        self.assertEqual(status, "invalid_json")
        self.assertIn("Multiple", error)


if __name__ == "__main__":
    unittest.main()
