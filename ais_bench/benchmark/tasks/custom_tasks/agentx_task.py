"""AgentX (InferenceX AgentX 1.0) 性能测评任务。

Trace DAG、会话回放、warmup 和 profiling 全部由独立环境中的
``agentx-harness==1.0.13`` 完成；本模块只负责配置、数据溯源、进程和结果。
"""

import argparse
import hashlib
import json
import math
import os
import os.path as osp
import platform
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

from mmengine.config import Config
from mmengine.utils import mkdir_or_exist

from ais_bench.benchmark.registry import TASKS
from ais_bench.benchmark.cli.utils import SWEEP_COOLDOWN_SECONDS
from ais_bench.benchmark.tasks.base import BaseTask, TaskStateManager
from ais_bench.benchmark.tasks.custom_tasks.agentx_result import (
    SUPPORTED_HARNESS_VERSION,
    atomic_write_json,
    normalize_agentx_result,
)
from ais_bench.benchmark.utils.core.abbr import task_abbr_from_cfg
from ais_bench.benchmark.utils.logging import AISLogger

AGENTX_PACKAGE = "agentx-harness"
AGENTX_SUMMARY_NAME = "profile_export_aiperf.json"
AGENTX_SUMMARY_CSV_NAME = "profile_export_aiperf.csv"
AGENTX_SCENARIO = "agentx"
DEFAULT_ENDPOINT = "/v1/chat/completions"
DEFAULT_HARNESS_VERSION = SUPPORTED_HARNESS_VERSION
IMAGE_HARNESS_PYTHON = "/opt/venvs/agentx113/bin/python"
SMOKE_DEFAULT_DURATION = 60.0
SMOKE_DEFAULT_TRACE_LIMIT = 4
# agentx 场景要求 profiling 时长不低于 900 秒才能进入稳态并触发 KV offload。
MIN_BENCHMARK_DURATION = 900.0
# agentx scenario preset 的官方默认口径；manifest 用于记录未显式覆盖时的生效值。
PRESET_BENCHMARK_DURATION = 3600.0
PRESET_WARMUP_REQUESTS_PER_LANE = 10
PRESET_RANDOM_SEED = 42
ENGINE_PROBE_TIMEOUT = 5.0
MANIFEST_SCHEMA_VERSION = "1.0"

DATASETS: Dict[str, Dict[str, str]] = {
    "full": {
        "alias": "semianalysis_cc_traces_weka_062126",
        "hf_repo": "semianalysisai/cc-traces-weka-062126",
        "hf_revision": "23f152f6f0f9399a85901b89a6458def0ef16729",
        "sha256": "29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276",
    },
    "256k": {
        "alias": "semianalysis_cc_traces_weka_062126_256k",
        "hf_repo": "semianalysisai/cc-traces-weka-062126-256k",
        "hf_revision": "8fecd2fc56694469f758f0afbbb6335ad3043740",
        "sha256": "e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef",
    },
}

_REDACT_NEXT_FLAGS = ("--api-key",)
_REDACT_HEADER_NAMES = ("authorization", "proxy-authorization", "x-api-key", "x-auth-token")
_SECRET_ENV_RE = re.compile(r"(?:key|token|secret|password|credential|authorization)", re.I)
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")
# 上游 AIPerf 的结果表标题和帮助头带 "NVIDIA AIPerf |" 品牌前缀，
# 控制台回显与 agentx_stdout 日志均去掉；上游原始产物（summary/CSV 等）保持原样。
_SCREEN_BRAND_RE = re.compile(r"NVIDIA\s+AIPerf\s*\|\s*")


def resolve_variant(variant: Optional[str]) -> str:
    """Require an explicit corpus variant; never infer it from the model name."""
    if not isinstance(variant, str) or not variant.strip():
        raise ValueError(
            "AgentX requires an explicit dataset variant: 'full' or '256k' "
            "(--agentx-dataset-variant or datasets[0].args.variant)."
        )
    variant = variant.strip()
    if variant not in DATASETS:
        raise ValueError(f"Unknown AgentX dataset variant {variant!r}; expected {sorted(DATASETS)}.")
    return variant


def _sha256_and_line_count(path: str) -> Tuple[str, int]:
    digest = hashlib.sha256()
    lines = 0
    last_byte = b""
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
            lines += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if last_byte and last_byte != b"\n":
        lines += 1
    return digest.hexdigest(), lines


def sha256_file(path: str) -> str:
    return _sha256_and_line_count(path)[0]


def resolve_endpoint(value: Any) -> str:
    if value is None:
        return DEFAULT_ENDPOINT
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"AgentX endpoint must be a non-empty path such as {DEFAULT_ENDPOINT}."
        )
    endpoint = value.strip()
    if (
        not endpoint.startswith("/")
        or any(char in endpoint for char in ("?", "#"))
        or any(char.isspace() for char in endpoint)
    ):
        raise ValueError(
            f"AgentX endpoint must be an absolute path without query, fragment or spaces: {endpoint!r}."
        )
    return endpoint


def normalize_base_url(url: str, endpoint: str = DEFAULT_ENDPOINT) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"Invalid model url {url!r}; expected http(s)://host:port[/path].")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("AgentX model url must not contain userinfo, query or fragment.")
    path = parts.path.rstrip("/")
    if path.endswith(endpoint.rstrip("/")):
        raise ValueError(
            f"AgentX model url must be a base URL without {endpoint}; "
            "configure the path with --endpoint instead."
        )
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def resolve_cooldown(value: Any) -> float:
    """Seconds to wait between consecutive performance runs; default 5."""
    if value is None:
        return float(SWEEP_COOLDOWN_SECONDS)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(
            f"AgentX cooldown must be a non-negative finite number of seconds, got {value!r}."
        )
    return float(value)


def resolve_smoke_overrides(
    run_mode: str, duration: Optional[float], trace_limit: Optional[int]
) -> Tuple[Optional[float], Optional[int]]:
    """smoke 未显式传值时补短跑默认值；benchmark 只透传显式值，否则用 harness 预设。"""
    if run_mode != "smoke":
        return duration, trace_limit
    return (
        SMOKE_DEFAULT_DURATION if duration is None else duration,
        SMOKE_DEFAULT_TRACE_LIMIT if trace_limit is None else trace_limit,
    )


def _concurrency_levels(value: Any) -> List[int]:
    """Normalize ``--batch-size`` into ordered, unique positive levels."""
    if isinstance(value, bool):
        raise ValueError(f"AgentX concurrency must be an integer, got {value!r}.")
    if isinstance(value, int):
        levels = [value]
    elif isinstance(value, (list, tuple)):
        levels = list(value)
    else:
        raise ValueError(f"AgentX concurrency must be an integer or a list of integers, got {value!r}.")
    if not levels:
        raise ValueError("AgentX concurrency requires at least one level.")
    seen = set()
    for level in levels:
        if isinstance(level, bool) or not isinstance(level, int) or level < 1:
            raise ValueError(f"AgentX concurrency must be a positive integer, got {level!r}.")
        if level in seen:
            raise ValueError(f"AgentX concurrency level {level} is duplicated.")
        seen.add(level)
    return levels


def _looks_like_path(value: str) -> bool:
    return osp.dirname(value) != ""


def _explicit_local_path(value: str) -> bool:
    return value.startswith((".", "~", "/", "\\")) or osp.isabs(value) or bool(re.match(r"^[A-Za-z]:[\\/]", value))


def validate_agentx_args(model_cfg: Dict[str, Any], args: Dict[str, Any]) -> List[int]:
    """CLI/配置共用校验；在下载和 Harness 启动前拒绝非法配置。"""
    model = model_cfg.get("model") or args.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("AgentX requires a non-empty model name (models[0].model).")
    url = model_cfg.get("url") or args.get("url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("AgentX requires models[0].url with the service base URL.")
    endpoint = resolve_endpoint(model_cfg.get("endpoint"))
    normalize_base_url(url.strip(), endpoint)
    levels = _concurrency_levels(args.get("concurrency", 1))
    resolve_cooldown(args.get("cooldown"))

    model_path = model_cfg.get("path")
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("AgentX requires models[0].path for the model/tokenizer.")
    model_path = model_path.strip()
    if _explicit_local_path(model_path) and not osp.isdir(osp.expanduser(model_path)):
        raise FileNotFoundError(f"AgentX local model/tokenizer directory does not exist: {model_path}")
    resolve_variant(args.get("variant"))
    run_mode = args.get("run_mode") or "benchmark"
    if run_mode not in ("benchmark", "smoke"):
        raise ValueError("AgentX run_mode must be 'benchmark' or 'smoke'.")

    dataset_path = args.get("dataset_path")
    if dataset_path is not None:
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            raise ValueError("AgentX dataset_path must be a non-empty string.")
        if not osp.isdir(dataset_path):
            raise FileNotFoundError(f"AgentX dataset_path is not a directory: {dataset_path}")
        if not osp.isfile(osp.join(dataset_path, "traces.jsonl")):
            raise FileNotFoundError(f"dataset_path must contain traces.jsonl: {dataset_path}")

    for key in ("duration", "trace_limit"):
        value = args.get(key)
        if value is None:
            continue
        if key == "trace_limit" and run_mode != "smoke":
            raise ValueError("AgentX trace_limit is only valid in smoke mode.")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"AgentX {key} must be a positive finite number.")
        if key == "trace_limit" and not isinstance(value, int):
            raise ValueError("AgentX trace_limit must be a positive integer.")
        if key == "duration" and run_mode == "benchmark" and value < MIN_BENCHMARK_DURATION:
            raise ValueError(
                f"AgentX benchmark duration must be >= {MIN_BENCHMARK_DURATION:g} seconds; "
                f"got {value}. Shorter runs are only allowed in smoke mode."
            )

    api_key = args.get("api_key")
    if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
        raise ValueError("AgentX api_key must be a non-empty string.")

    env = args.get("env") or {}
    if not isinstance(env, dict):
        raise ValueError("AgentX env must be a mapping of KEY=VALUE.")
    for key, value in env.items():
        if not isinstance(key, str) or not _ENV_NAME_RE.fullmatch(key):
            raise ValueError(f"Invalid AgentX env variable name: {key!r}.")
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError(f"AgentX env value for {key} must be a string without NUL characters.")
        if run_mode == "benchmark" and key in {"AIPERF_CONFIG", "AIPERF_SCENARIO"}:
            raise ValueError(f"AgentX benchmark env must not override the workload via {key}.")

    # extra_args 是通用参数入口：用户直接写出 agentx 原始参数，原样追加到命令末尾。
    extra_args = args.get("extra_args")
    if extra_args is not None and (not isinstance(extra_args, str) or not extra_args.strip()):
        raise ValueError("AgentX extra_args must be a non-empty string.")

    for name in ("agentx_python", "agentx_bin"):
        value = args.get(name)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"AgentX {name} must be a non-empty string.")
        if _looks_like_path(value) and not osp.isfile(osp.expanduser(value)):
            raise FileNotFoundError(f"AgentX {name} does not exist: {value}")
    return levels


def _split_extra_args(extra_args: Optional[str]) -> List[str]:
    """把用户写的一段 agentx 原始参数拆成 argv 片段；只做分词，不解释参数含义。"""
    if not extra_args:
        return []
    text = extra_args.strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()
    try:
        return shlex.split(text)
    except ValueError as exc:
        raise ValueError(f"Invalid AgentX extra_args: {exc}") from exc


# 生成的命令中不带值的开关选项，去重时需要连同其后的值一起保留/丢弃。
_AGENTX_FLAG_OPTIONS = {"--tokenizer-trust-remote-code", "--unsafe-override"}


def _option_name(token: str) -> Optional[str]:
    if not token.startswith("-") or token in ("-", "--"):
        return None
    return token.split("=", 1)[0]


def _apply_extra_args(command: List[str], extra_tokens: List[str]) -> List[str]:
    """extra args 优先级最高：移除基础命令中的同名选项后，再整体追加到末尾。"""
    overridden = {name for name in map(_option_name, extra_tokens) if name}
    if not overridden:
        return command + extra_tokens
    merged: List[str] = []
    index = 0
    while index < len(command):
        token = command[index]
        name = _option_name(token)
        if name in overridden:
            index += 1
            if "=" not in token and name not in _AGENTX_FLAG_OPTIONS and index < len(command):
                index += 1
            continue
        merged.append(token)
        index += 1
    return merged + extra_tokens


def _argv_parameters(argv: List[str]) -> Dict[str, Any]:
    """从最终 argv 还原各选项的实际取值，写入 manifest 供审计。"""
    parameters: Dict[str, Any] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        name = _option_name(token)
        if name is None:
            index += 1
            continue
        if "=" in token:
            parameters[name] = token.split("=", 1)[1]
            index += 1
            continue
        if index + 1 < len(argv) and _option_name(argv[index + 1]) is None:
            parameters[name] = argv[index + 1]
            index += 2
            continue
        parameters[name] = True
        index += 1
    return parameters


def build_agentx_command(
    agentx_bin: List[str], *, url: str, model: str, concurrency: int,
    dataset_alias: Optional[str], artifact_dir: str, dataset_dir: Optional[str] = None,
    endpoint: Optional[str] = None, tokenizer_path: Optional[str] = None,
    tokenizer_trust_remote_code: bool = False,
    api_key: Optional[str] = None, extra_args: Optional[str] = None,
    run_mode: str = "benchmark", duration: Optional[float] = None,
    trace_limit: Optional[int] = None,
) -> List[str]:
    """唯一的真实/dry-run 命令生成器；benchmark 默认沿用 preset，显式 duration 透传。"""
    endpoint = resolve_endpoint(endpoint)
    command = list(agentx_bin) + [
        "profile", "--scenario", "agentx", "--url", normalize_base_url(url, endpoint),
        "--endpoint", endpoint, "--model", model, "--concurrency", str(concurrency),
        "--output-artifact-dir", artifact_dir,
    ]
    if dataset_dir:
        command += ["--hf-weka-dataset", dataset_dir, "--unsafe-override"]
    else:
        if not dataset_alias:
            raise ValueError("AgentX public dataset alias is required when dataset_dir is absent.")
        command += ["--public-dataset", dataset_alias]
    if tokenizer_path:
        command += ["--tokenizer", tokenizer_path]
        if tokenizer_trust_remote_code:
            command += ["--tokenizer-trust-remote-code"]
    if api_key:
        command += ["--api-key", api_key]
    if duration is not None:
        command += ["--benchmark-duration", str(duration)]
    if run_mode == "smoke":
        if "--unsafe-override" not in command:
            command.append("--unsafe-override")
        if trace_limit is not None:
            command += ["--num-dataset-entries", str(trace_limit)]
    if extra_args:
        command = _apply_extra_args(command, _split_extra_args(extra_args))
    return command


def redact_command(command: List[str]) -> List[str]:
    redacted: List[str] = []
    redact_next = False
    for token in command:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if token in _REDACT_NEXT_FLAGS:
            redacted.extend([token, "<redacted>"] if token == command[-1] else [token])
            redact_next = True
            continue
        matched_secret = False
        for flag in _REDACT_NEXT_FLAGS:
            if token.startswith(flag + "="):
                redacted.append(flag + "=<redacted>")
                matched_secret = True
                break
        if matched_secret:
            continue
        if ":" in token:
            name, _, _ = token.partition(":")
            if name.strip().lower() in _REDACT_HEADER_NAMES:
                redacted.append(f"{name}:<redacted>")
                continue
        redacted.append(re.sub(r"\$\{[^}]*\}", "<redacted>", token))
    return redacted


def resolve_harness_python(args: Dict[str, Any], environ: Optional[Dict[str, str]] = None) -> str:
    """Resolve the explicitly supplied interpreter or the image/current interpreter."""
    if environ is None:
        environ = os.environ
    explicit_python = args.get("agentx_python") or environ.get("AISBENCH_AGENTX_PYTHON")
    if explicit_python:
        return osp.expanduser(explicit_python)
    if osp.isfile(IMAGE_HARNESS_PYTHON):
        return IMAGE_HARNESS_PYTHON
    return sys.executable


def probe_harness_environment(python_exe: str, env: Dict[str, str],
                              agentx_bin: Optional[str] = None) -> Dict[str, Any]:
    """在目标解释器内检查 Python、distribution、版本及 aiperf 冲突。"""
    code = (
        "import json,sys,importlib.metadata as m;"
        "print(json.dumps({'python_version':sys.version.split()[0],"
        "'agentx_version':m.version('agentx-harness'),"
        "'upstream_aiperf':any((d.metadata.get('Name') or '').lower()=='aiperf' "
        "for d in m.distributions())}))"
    )
    try:
        completed = subprocess.run([python_exe, "-c", code], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, timeout=120, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"AGENTX_ENV_INVALID: cannot query {python_exe!r}: {exc}") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"AGENTX_ENV_INVALID: agentx-harness=={DEFAULT_HARNESS_VERSION} is unavailable "
            f"in {python_exe!r}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        metadata = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"AGENTX_ENV_INVALID: invalid environment probe output: {completed.stdout!r}") from exc
    major, minor = tuple(int(item) for item in metadata["python_version"].split(".")[:2])
    if (major, minor) not in ((3, 11), (3, 12), (3, 13)):
        raise RuntimeError(f"AGENTX_ENV_INVALID: Harness requires Python 3.11~3.13, found {metadata['python_version']}.")
    if metadata["agentx_version"] != DEFAULT_HARNESS_VERSION:
        raise RuntimeError(
            f"AGENTX_VERSION_UNSUPPORTED: required agentx-harness=={DEFAULT_HARNESS_VERSION}, "
            f"found {metadata['agentx_version']}. Interpreter: {python_exe}. No benchmark process was started."
        )
    if metadata.get("upstream_aiperf"):
        raise RuntimeError("AGENTX_ENV_INVALID: upstream PyPI aiperf conflicts with agentx-harness; use an isolated environment.")
    command = resolve_agentx_bin(agentx_bin, python_exe)
    try:
        help_result = subprocess.run(command + ["--help"], stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, timeout=30, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"AGENTX_ENV_INVALID: agentx entry self-check failed: {exc}") from exc
    if help_result.returncode != 0:
        raise RuntimeError(f"AGENTX_ENV_INVALID: agentx entry self-check failed: {help_result.stdout[-2000:]}")
    return dict(metadata, distribution=AGENTX_PACKAGE, version=metadata["agentx_version"],
                python=python_exe, entry=command, platform=platform.platform())


def probe_harness_version(python_exe: Optional[str], env: Optional[Dict[str, str]] = None) -> str:
    """兼容旧调用；新任务使用完整 ``probe_harness_environment``。"""
    return probe_harness_environment(python_exe or sys.executable, env or os.environ.copy())["version"]


def resolve_agentx_bin(agentx_bin: Optional[str], python_exe: Optional[str]) -> List[str]:
    if agentx_bin:
        return [agentx_bin]
    candidates: List[str] = []
    if python_exe:
        bin_dir = Path(python_exe).resolve().parent
        candidates.append(str(bin_dir / "agentx"))
    for candidate in candidates:
        if osp.isfile(candidate):
            return [candidate]
    if python_exe:
        return [python_exe, "-m", "agentx_cli"]
    raise RuntimeError("Cannot locate agentx; set agentx_bin or agentx_python.")


def _safe_component(value: Any) -> str:
    component = _SAFE_COMPONENT_RE.sub("_", str(value or "unknown")).strip("._-")
    return component[:80] or "unknown"


def _redact_text(text: str, secrets: List[str]) -> str:
    result = text
    for secret in secrets:
        if secret:
            result = result.replace(secret, "<redacted>")
    result = re.sub(r"(?i)(authorization|proxy-authorization|x-api-key|x-auth-token)\s*[:=]\s*\S+",
                    r"\1:<redacted>", result)
    result = re.sub(r"(--api-key(?:=|\s+))\S+", r"\1<redacted>", result)
    return result


def _screen_line(text: str) -> str:
    """控制台回显与 agentx_stdout 日志落盘前去掉上游品牌前缀；原始产物不受影响。"""
    return _SCREEN_BRAND_RE.sub("", text)


def run_dir_name(variant: str, concurrency: int) -> str:
    """每个 batch 档位一个全新目录：``agentx_<variant>_batchsize_<N>``。"""
    return f"agentx_{_safe_component(variant)}_batchsize_{int(concurrency)}"


def _http_json(url: str, timeout: float = ENGINE_PROBE_TIMEOUT) -> Tuple[Optional[Any], Optional[str]]:
    """读取引擎探测端点；失败只记录错误，不中断测评。"""
    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8")), None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return None, str(exc)


def _extra_arg_value(extra_args: Optional[str], flag: str) -> Optional[str]:
    """从 ``--agentx-extra-arg`` 中读取显式覆盖值；仅用于 manifest 记录。"""
    args = _split_extra_args(extra_args)
    for index, token in enumerate(args):
        if token == flag and index + 1 < len(args):
            return args[index + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


@TASKS.register_module()
class AgentXTask(BaseTask):
    """一次 Task 执行一个 model × dataset，可包含多个并发档位并逐档串行。"""

    name_prefix = "AgentXTask"
    log_subdir = "logs"
    _PROGRESS_INTERVAL = 2.0

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # 支持配置文件的单层 datasets，以及 NaivePartitioner 历史嵌套形状。
        if isinstance(self.dataset_cfgs, dict):
            self.dataset_cfgs = [self.dataset_cfgs]
        self.dataset_cfg = self.dataset_cfgs[0]
        self.args: Dict[str, Any] = dict(self.dataset_cfg.get("args") or {})
        self.task_state_manager: Optional[TaskStateManager] = None
        self._last_state_ts = 0.0
        self.run_id: Optional[str] = None

    def get_command(self, cfg_path, template) -> str:
        sys.path.append(os.getcwd())
        return f"{sys.executable} {__file__} {cfg_path}"

    def run(self, task_state_manager: Optional[TaskStateManager] = None) -> Dict[str, Any]:
        self.task_state_manager = task_state_manager
        plans = self._prepare()
        results: List[Dict[str, Any]] = []
        for index, plan in enumerate(plans):
            self._update_state({"status": "start", "stage": "agentx", "run_id": plan["run_id"]})
            result = self._execute_plan(plan)
            self.logger.info(
                "AgentX run_id=%s concurrency=%s status=%s parse=%s valid=%s scope=%s manifest=%s",
                plan["run_id"], plan["concurrency"], result["run"]["status"],
                result["run"]["parse_status"], result["run"]["benchmark_valid"],
                result["run"]["comparison_scope"], plan["manifest_path"],
            )
            results.append(result)
            if result["run"]["status"] != "completed" or result["run"]["parse_status"] != "success":
                self._update_state({"status": "error", "run_id": plan["run_id"]})
                message = (
                    f"AgentX run {plan['run_id']} failed: {result['run']['status']}/"
                    f"{result['run']['parse_status']}; see {result['artifacts']['harness_log']}"
                )
                skipped = [item["concurrency"] for item in plans[index + 1:]]
                if skipped:
                    message += f"; skipped concurrency levels {skipped} after failure"
                raise RuntimeError(message)
            if index + 1 < len(plans) and plan["cooldown"] > 0:
                self.logger.info(
                    "AgentX cooldown %.3fs before next concurrency level", plan["cooldown"]
                )
                self._adapter_log(
                    plan, f"cooldown {plan['cooldown']:g}s before next concurrency level"
                )
                time.sleep(plan["cooldown"])
        self._update_state({"status": "finish", "run_id": plans[-1]["run_id"]})
        return results[0] if len(results) == 1 else {"runs": results}

    def dry_run(self) -> Dict[str, Any]:
        """完成离线校验并写脱敏计划；不启动 Harness、不访问模型服务。"""
        plans = self._prepare()
        payload = {"runs": [self._redacted_plan(plan) for plan in plans]}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        self._adapter_log(
            plans[0],
            f"dry-run completed for {len(plans)} concurrency level(s); "
            "no Harness process or model request was started",
        )
        return payload

    def _prepare(self) -> List[Dict[str, Any]]:
        levels = validate_agentx_args(self.model_cfg, self.args)
        cooldown = resolve_cooldown(self.args.get("cooldown"))
        run_mode = self.args.get("run_mode") or "benchmark"
        duration, trace_limit = resolve_smoke_overrides(
            run_mode, self.args.get("duration"), self.args.get("trace_limit")
        )
        dataset = self._resolve_dataset(run_mode)
        if dataset.get("num_dataset_entries") is None and trace_limit is not None:
            dataset["num_dataset_entries"] = trace_limit
        harness_env = self._harness_env()
        harness_python = resolve_harness_python(self.args, harness_env)
        try:
            harness = probe_harness_environment(harness_python, harness_env, self.args.get("agentx_bin"))
        except RuntimeError as exc:
            raise RuntimeError(
                f"{exc}\nAgentX environment search order: agentx_python/AISBENCH_AGENTX_PYTHON, "
                f"image interpreter {IMAGE_HARNESS_PYTHON}, current interpreter. "
                "Install agentx-harness==1.0.13 in an isolated Python 3.11~3.13 "
                "environment and set --agentx-python to its interpreter."
            ) from exc
        api_key = (self.args.get("api_key") or self.model_cfg.get("api_key")
                   or os.environ.get("AISBENCH_AGENTX_API_KEY"))
        secrets = [api_key] if api_key else []
        secrets.extend(
            value for key, value in (self.args.get("env") or {}).items()
            if _SECRET_ENV_RE.search(key) and value
        )
        plans = []
        for concurrency in levels:
            run_id = run_dir_name(dataset["variant"], concurrency)
            # 根目录固定 configs/、logs/、performances/，档位以 run_id 分子目录。
            run_dir = osp.abspath(osp.join(self.work_dir, "performances", run_id))
            logs_dir = osp.abspath(osp.join(self.work_dir, "logs", run_id))
            artifact_dir = osp.join(run_dir, "agentx_raw")
            if osp.exists(artifact_dir) and os.listdir(artifact_dir):
                raise RuntimeError(
                    f"AGENTX_OUTPUT_WRITE_FAILED: artifact directory is not empty: {artifact_dir}"
                )
            for directory in (logs_dir, run_dir, artifact_dir):
                mkdir_or_exist(directory)
            command = build_agentx_command(
                harness["entry"], url=self.model_cfg.get("url") or self.args.get("url"),
                model=self.model_cfg["model"], endpoint=self.model_cfg.get("endpoint"),
                concurrency=concurrency, dataset_alias=dataset.get("alias"),
                dataset_dir=dataset.get("path"), artifact_dir=osp.abspath(artifact_dir),
                tokenizer_path=self.model_cfg.get("path"),
                tokenizer_trust_remote_code=self.model_cfg.get("trust_remote_code", False),
                api_key=api_key, extra_args=self.args.get("extra_args"), run_mode=run_mode,
                duration=duration, trace_limit=trace_limit,
            )
            plan = {
                "run_id": run_id, "run_mode": run_mode, "concurrency": concurrency,
                "duration": duration, "trace_limit": trace_limit, "cooldown": cooldown,
                "argv": command, "env": harness_env, "cwd": osp.abspath(self.work_dir),
                "run_dir": run_dir, "artifact_dir": osp.abspath(artifact_dir),
                "performance_dir": run_dir, "logs_dir": logs_dir,
                "summary_path": osp.join(osp.abspath(artifact_dir), AGENTX_SUMMARY_NAME),
                "summary_csv_path": osp.join(osp.abspath(artifact_dir), AGENTX_SUMMARY_CSV_NAME),
                "log_path": osp.join(logs_dir, "agentx_stdout.log"),
                "adapter_log": osp.join(logs_dir, "agentx_adapter.log"),
                "plan_path": osp.join(run_dir, "execution_plan.redacted.json"),
                "manifest_path": osp.join(run_dir, "manifest.json"),
                "harness": harness, "dataset": dataset, "api_key": api_key,
                "secrets": secrets,
            }
            atomic_write_json(plan["plan_path"], self._redacted_plan(plan))
            self._adapter_log(
                plan,
                f"prepared run_id={run_id} concurrency={concurrency} "
                f"command={' '.join(redact_command(command))}",
            )
            plans.append(plan)
        self.run_id = plans[0]["run_id"] if len(plans) == 1 else None
        return plans

    def _redacted_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        # 只记录用户显式传入的环境，不把宿主全部环境落盘。
        public_keys = set((self.args.get("env") or {}).keys())
        public_env = {}
        for key in public_keys:
            if key in plan["env"]:
                public_env[key] = plan["env"][key] if not _SECRET_ENV_RE.search(key) else "<redacted>"
        harness = {key: value for key, value in plan["harness"].items() if key != "entry"}
        harness["entry"] = redact_command(plan["harness"]["entry"])
        return {
            "run_id": plan["run_id"], "run_mode": plan["run_mode"],
            "concurrency": plan["concurrency"], "cooldown": plan["cooldown"],
            "duration": plan["duration"], "trace_limit": plan["trace_limit"],
            "argv": redact_command(plan["argv"]),
            "env": public_env, "cwd": plan["cwd"], "run_dir": plan["run_dir"],
            "artifact_dir": plan["artifact_dir"],
            "summary_path": plan["summary_path"], "summary_csv_path": plan["summary_csv_path"],
            "log_path": plan["log_path"], "adapter_log": plan["adapter_log"],
            "plan_path": plan["plan_path"],
            "manifest_path": plan["manifest_path"],
            "harness": harness, "dataset": plan["dataset"],
        }

    def _resolve_dataset(self, run_mode: str) -> Dict[str, Any]:
        variant = resolve_variant(self.args.get("variant"))
        spec = DATASETS[variant]
        dataset_path = self.args.get("dataset_path")
        expected = spec["sha256"].lower()
        base = {
            "variant": variant, "registry_alias": spec["alias"], "repo": spec["hf_repo"],
            "revision": spec["hf_revision"], "expected_sha256": expected,
            "actual_sha256": None, "source": "public", "alias": spec["alias"],
            "verification": ["not_verified"], "sha256_mismatch": False,
            "num_dataset_entries": None,
        }
        if not dataset_path:
            return base
        started = time.monotonic()
        trace_file = osp.realpath(osp.join(dataset_path, "traces.jsonl"))
        actual, num_entries = _sha256_and_line_count(trace_file)
        mismatch = actual.lower() != expected
        if mismatch and run_mode != "smoke":
            raise ValueError(
                f"AGENTX_DATASET_HASH_MISMATCH: expected {expected}, got {actual}. "
                "Only smoke mode may continue with a mismatched local corpus."
            )
        stat = os.stat(trace_file)
        base.update({
            "source": "local", "alias": None, "path": osp.realpath(dataset_path),
            "trace_file": trace_file, "actual_sha256": actual, "sha256_mismatch": mismatch,
            "verification": ["not_verified"] if mismatch else ["local_content_hash_verified"],
            "file_size": stat.st_size, "file_mtime": stat.st_mtime,
            "num_dataset_entries": num_entries,
            "hash_elapsed_seconds": round(time.monotonic() - started, 3),
        })
        return base

    def _harness_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env.update(self.args.get("env") or {})
        return env

    def _execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        started_wall = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        return_code, cancelled, start_error = self._execute(
            plan["argv"], plan["log_path"], plan["env"], plan["run_id"],
            plan.get("secrets") or [], plan["adapter_log"],
        )
        raw, parse_status, parse_error = self._read_summary(plan["artifact_dir"], plan["summary_path"])
        dataset = dict(plan["dataset"])
        entries = self._dataset_entries(dataset, raw)
        if entries is not None:
            dataset["num_dataset_entries"] = entries
        engine, probe_errors = self._probe_engine(plan)
        was_cancelled = cancelled or bool(raw.get("was_cancelled"))
        if start_error:
            status = "start_failed"
        elif was_cancelled:
            status = "cancelled"
        elif return_code != 0 or parse_status != "success":
            status = "failed"
        else:
            status = "completed"
        errors = list(probe_errors)
        if start_error:
            errors.append({"code": "AGENTX_PROCESS_START_FAILED", "message": start_error})
        if return_code not in (0, None):
            errors.append({"code": "AGENTX_PROCESS_FAILED", "message": f"Harness exited with {return_code}."})
        if parse_error:
            code = "AGENTX_RESULT_SCHEMA_UNSUPPORTED" if parse_status == "unsupported_schema" else (
                "AGENTX_RESULT_MISSING" if parse_status == "missing" else "AGENTX_RESULT_INVALID")
            errors.append({"code": code, "message": parse_error})
        if was_cancelled:
            errors.append({"code": "AGENTX_CANCELLED", "message": "Run was cancelled; partial artifacts retained."})
        secrets = plan.get("secrets") or []
        errors = [{**item, "message": _redact_text(str(item["message"]), secrets)} for item in errors]
        identity = {
            "model": self.model_cfg["model"], "model_abbr": self.model_cfg["abbr"],
            "dataset_abbr": self.dataset_cfg["abbr"], "variant": plan["dataset"]["variant"],
            "concurrency": plan["concurrency"],
        }
        run = {
            "run_id": plan["run_id"], "mode": plan["run_mode"], "status": status,
            "parse_status": parse_status, "start_time": started_wall,
            "end_time": datetime.now(timezone.utc).isoformat(), "return_code": return_code,
            "was_cancelled": was_cancelled, "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        artifacts = {
            "raw_summary": plan["summary_path"],
            "raw_csv": plan["summary_csv_path"],
            "raw_records": osp.join(plan["artifact_dir"], "profile_export.jsonl"),
            "raw_artifacts": plan["artifact_dir"],
            "harness_log": plan["log_path"], "adapter_log": plan["adapter_log"],
            "execution_plan": plan["plan_path"],
            "manifest": plan["manifest_path"],
        }
        harness = dict(plan["harness"])
        harness.pop("entry", None)
        result = normalize_agentx_result(
            raw, run=run, identity=identity, harness=harness, dataset=dataset,
            command=redact_command(plan["argv"]), artifacts=artifacts,
            errors=errors,
        )
        manifest = self._build_manifest(plan, result, raw, engine)
        manifest = self._finalize_run(plan, result, manifest)
        self._adapter_log(
            plan,
            f"finished status={status} parse={parse_status} return_code={return_code} "
            f"manifest={plan['manifest_path']}",
        )
        return result

    def _finalize_run(self, plan: Dict[str, Any], result: Dict[str, Any],
                      manifest: Dict[str, Any]) -> Dict[str, Any]:
        """复制上游 summary 到档位目录并写入 manifest（单档位即落盘）。"""
        from ais_bench.benchmark.summarizers.agentx import AgentXSummarizer

        summarizer = AgentXSummarizer({"work_dir": self.work_dir})
        manifest = summarizer.finalize_run(
            plan["run_dir"], manifest, plan["artifact_dir"], plan["logs_dir"]
        )
        manifest_result = manifest.get("result") or {}
        result["artifacts"]["manifest"] = plan["manifest_path"]
        for key in ("summary_json", "summary_csv"):
            relative = manifest_result.get(key)
            result["artifacts"][key] = (
                osp.join(plan["run_dir"], relative) if relative else None
            )
        return manifest

    def _dataset_entries(self, dataset: Dict[str, Any], raw: Dict[str, Any]) -> Optional[int]:
        if dataset.get("num_dataset_entries") is not None:
            return dataset["num_dataset_entries"]
        metadata = raw.get("metadata") if isinstance(raw, dict) else None
        harness_dataset = metadata.get("dataset") if isinstance(metadata, dict) else None
        if isinstance(harness_dataset, dict):
            for key in ("num_entries", "num_dataset_entries", "total_entries", "records"):
                value = harness_dataset.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    return value
        return None

    def _probe_engine(self, plan: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """只读探测引擎版本与模型信息；失败写 null 并记录错误，不中断测评。"""
        endpoint = resolve_endpoint(self.model_cfg.get("endpoint"))
        base_url = normalize_base_url(
            self.model_cfg.get("url") or self.args.get("url"), endpoint
        )
        version_url = base_url + "/version"
        models_url = base_url + "/v1/models"
        version_payload, version_error = _http_json(version_url)
        models_payload, models_error = _http_json(models_url)
        version = version_payload.get("version") if isinstance(version_payload, dict) else None
        served_model = max_model_len = None
        entries = models_payload.get("data") if isinstance(models_payload, dict) else None
        if isinstance(entries, list):
            candidates = [item for item in entries if isinstance(item, dict)]
            chosen = next(
                (item for item in candidates if item.get("id") == self.model_cfg.get("model")),
                candidates[0] if candidates else None,
            )
            if chosen is not None:
                served_model = chosen.get("id")
                value = chosen.get("max_model_len")
                if isinstance(value, int) and not isinstance(value, bool):
                    max_model_len = value
        errors: List[Dict[str, Any]] = []
        if version_error:
            errors.append({
                "code": "AGENTX_ENGINE_PROBE_FAILED",
                "message": f"GET {version_url} failed: {version_error}",
            })
        if models_error:
            errors.append({
                "code": "AGENTX_ENGINE_PROBE_FAILED",
                "message": f"GET {models_url} failed: {models_error}",
            })
        engine = {
            "type": "vllm" if version is not None else None,
            "version": version,
            "served_model": served_model,
            "max_model_len": max_model_len,
            "base_url": base_url,
            "endpoint": endpoint,
            "probe": {
                "version": {
                    "url": version_url,
                    "status": "ok" if version_error is None else "error",
                    "error": version_error,
                },
                "models": {
                    "url": models_url,
                    "status": "ok" if models_error is None else "error",
                    "error": models_error,
                },
            },
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }
        return engine, errors

    def _build_manifest(self, plan: Dict[str, Any], result: Dict[str, Any],
                        raw: Dict[str, Any], engine: Dict[str, Any]) -> Dict[str, Any]:
        run = result.get("run") or {}
        identity = result.get("identity") or {}
        harness = result.get("harness") or {}
        dataset = result.get("dataset") or {}
        validity = result.get("validity") or {}
        primary_reason = validity.get("primary_invalid_reason")
        invalid_reasons = [primary_reason] if primary_reason else []
        all_invalid_reasons = list(validity.get("all_invalid_reasons") or [])
        redacted_argv = redact_command(plan["argv"])
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": plan["run_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "upstream": {
                "framework": AGENTX_PACKAGE,
                "version": harness.get("version"),
                "aiperf_version": harness.get("aiperf_version"),
                "python_version": harness.get("python_version"),
                "platform": harness.get("platform"),
            },
            "run_params": {
                "run_mode": plan["run_mode"],
                "benchmark_id": f"{identity.get('model_abbr')}/{identity.get('dataset_abbr')}",
                "command": harness.get("command") or redacted_argv,
                "command_note": (
                    "duration/seed/warmup 未显式覆盖时沿用 agentx scenario preset 默认口径"
                ),
                "started_at": run.get("start_time"),
                "ended_at": run.get("end_time"),
            },
            # 实际执行的 agentx 最终生效参数；extra args 覆盖后的值以这里为准。
            "agentx_command": {
                "parameters": _argv_parameters(redacted_argv),
                "extra_args": redact_command(_split_extra_args(self.args.get("extra_args"))),
            },
            "dataset": {
                "abbr": identity.get("dataset_abbr"),
                "variant": dataset.get("variant"),
                "source": dataset.get("source"),
                "registry_alias": dataset.get("registry_alias"),
                "repo": dataset.get("repo"),
                "revision": dataset.get("revision"),
                "expected_sha256": dataset.get("expected_sha256"),
                "actual_sha256": dataset.get("actual_sha256"),
                "sha256_mismatch": dataset.get("sha256_mismatch"),
                "trace_file": dataset.get("trace_file"),
                "file_size": dataset.get("file_size"),
                "num_dataset_entries": dataset.get("num_dataset_entries"),
            },
            "model": {
                "name": identity.get("model"),
                "abbr": identity.get("model_abbr"),
                "path": self.model_cfg.get("path"),
            },
            "engine": engine,
            "test_params": self._test_params(plan, raw),
            "result": {
                "status": run.get("status"),
                "parse_status": run.get("parse_status"),
                "data_status": run.get("data_status"),
                "benchmark_valid": run.get("benchmark_valid"),
                "comparison_scope": run.get("comparison_scope"),
                "return_code": run.get("return_code"),
                "was_cancelled": run.get("was_cancelled"),
                "harness_submission_valid": validity.get("harness_submission_valid"),
                "invalid_reasons": invalid_reasons,
                "all_invalid_reasons": all_invalid_reasons,
                "aisbench_invalid_reasons": validity.get("aisbench_invalid_reasons"),
                "errors": result.get("errors") or [],
                "summary_json": None,
                "summary_csv": None,
                "raw_artifacts": None,
            },
        }

    def _test_params(self, plan: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
        metadata = raw.get("metadata") if isinstance(raw, dict) else None
        config = metadata.get("config") if isinstance(metadata, dict) else None
        extra_args = self.args.get("extra_args")

        def pick(*keys):
            for source in (config, metadata):
                if isinstance(source, dict):
                    for key in keys:
                        if source.get(key) is not None:
                            return source[key]
            return None

        duration = plan.get("duration")
        if duration is None:
            duration = _extra_arg_value(extra_args, "--benchmark-duration")
        if duration is None:
            duration = pick("benchmark_duration", "benchmark_duration_s", "duration")
        if duration is None:
            duration = PRESET_BENCHMARK_DURATION
        if isinstance(duration, str):
            try:
                duration = float(duration)
            except ValueError:
                pass

        seed = _extra_arg_value(extra_args, "--random-seed") or _extra_arg_value(extra_args, "--seed")
        if seed is None:
            seed = pick("random_seed", "seed")
        if seed is None:
            seed = PRESET_RANDOM_SEED
        if isinstance(seed, str):
            try:
                seed = int(seed)
            except ValueError:
                pass

        warmup = pick("warmup_requests_per_lane", "num_warmup_requests", "warmup_requests")
        if warmup is None:
            warmup = PRESET_WARMUP_REQUESTS_PER_LANE
        return {
            "batch_size": plan["concurrency"],
            "benchmark_duration_s": duration,
            "warmup_requests_per_lane": warmup,
            "random_seed": seed,
            "scenario": AGENTX_SCENARIO,
            "trace_limit": plan.get("trace_limit"),
        }

    def _read_summary(self, artifact_dir: str, expected_path: str) -> Tuple[Dict[str, Any], str, Optional[str]]:
        matches = sorted(str(path) for path in Path(artifact_dir).rglob(AGENTX_SUMMARY_NAME) if path.is_file())
        if len(matches) > 1:
            return {}, "invalid_json", f"Multiple {AGENTX_SUMMARY_NAME} files found in the fresh run directory."
        if not osp.isfile(expected_path):
            detail = f"Expected summary is missing: {expected_path}"
            if matches:
                detail += f"; unexpected alternate locations: {matches}"
            return {}, "missing", detail
        try:
            with open(expected_path, "r", encoding="utf-8") as file:
                raw = json.load(file)
        except json.JSONDecodeError as exc:
            return {}, "invalid_json", f"Invalid summary JSON: {exc}"
        except OSError as exc:
            return {}, "io_error", f"Cannot read summary: {exc}"
        if not isinstance(raw, dict):
            return {}, "invalid_json", "AgentX summary top level must be an object."
        if raw.get("schema_version") != "1.4":
            return raw, "unsupported_schema", f"Unsupported AgentX schema_version: {raw.get('schema_version')!r}."
        return raw, "success", None

    def _execute(self, command: List[str], log_path: str, env: Dict[str, str], run_id: str,
                 secrets: Optional[List[str]] = None, adapter_log: Optional[str] = None
                 ) -> Tuple[Optional[int], bool, Optional[str]]:
        """流式消费日志；取消和异常路径均回收整个进程树。"""
        secrets = secrets or []
        popen_kwargs: Dict[str, Any] = {"start_new_session": True}
        process: Optional[subprocess.Popen] = None
        cancelled = False
        try:
            with open(log_path, "w", encoding="utf-8") as log_file:
                log_file.write(f"AISBench AgentX run_id={run_id}\n")
                log_file.flush()
                try:
                    process = subprocess.Popen(
                        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, env=env,
                        cwd=osp.dirname(osp.dirname(osp.dirname(log_path))),
                        **popen_kwargs,
                    )
                except OSError as exc:
                    return None, False, str(exc)
                if adapter_log:
                    self._append_log(adapter_log, f"started run_id={run_id} pid={process.pid}")
                assert process.stdout is not None
                try:
                    for line in process.stdout:
                        safe_line = _screen_line(_redact_text(line, secrets))
                        log_file.write(safe_line)
                        log_file.flush()
                        self.logger.info("%s", safe_line.rstrip())
                        self._update_state({"stage": "agentx", "run_id": run_id})
                    return process.wait(), cancelled, None
                except KeyboardInterrupt:
                    cancelled = True
                    if adapter_log:
                        self._append_log(adapter_log, f"cancellation requested run_id={run_id}")
                    code = self._terminate(process)
                    return code, True, None
                except BaseException:
                    self._terminate(process)
                    raise
        finally:
            if process is not None and process.poll() is None:
                self._terminate(process)
            if process is not None and process.stdout is not None:
                process.stdout.close()

    @staticmethod
    def _terminate(process: subprocess.Popen) -> int:
        """POSIX 进程组分级回收。"""
        if process.poll() is not None:
            return process.returncode or 0
        try:
            os.killpg(process.pid, signal.SIGINT)
            return process.wait(timeout=30)
        except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
            pass
        try:
            os.killpg(process.pid, signal.SIGTERM)
            return process.wait(timeout=10)
        except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()

    def _adapter_log(self, plan: Dict[str, Any], message: str) -> None:
        self._append_log(plan["adapter_log"], _redact_text(message, plan.get("secrets") or []))

    @staticmethod
    def _append_log(path: str, message: str) -> None:
        mkdir_or_exist(osp.dirname(path))
        with open(path, "a", encoding="utf-8") as file:
            file.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")

    def _update_state(self, payload: Dict[str, Any]) -> None:
        if self.task_state_manager is None:
            return
        now = time.time()
        if now - self._last_state_ts < self._PROGRESS_INTERVAL and payload.get("status") is None:
            return
        self._last_state_ts = now
        self.task_state_manager.update_task_state({"other_kwargs": payload})


def parse_args():
    parser = argparse.ArgumentParser(description="AISBench AgentX performance task")
    parser.add_argument("config", help="Config file path")
    return parser.parse_args()


if __name__ == "__main__":
    logger = AISLogger(__name__)
    args = parse_args()
    cfg = Config.fromfile(args.config)
    task_state_manager = TaskStateManager(
        tmp_path=os.path.join(cfg["work_dir"], "status_tmp"),
        task_name=task_abbr_from_cfg(cfg), is_debug=cfg["cli_args"]["debug"],
    )
    manager_t = threading.Thread(target=task_state_manager.launch, args=())
    manager_t.start()
    task_state_manager.update_task_state({"status": "start", "task_log_path": AgentXTask.log_subdir})
    started = time.perf_counter()
    try:
        AgentXTask(cfg).run(task_state_manager)
    except BaseException:
        task_state_manager.update_task_state({"status": "error"})
        raise
    task_state_manager.update_task_state({"status": "finish"})
    logger.info("AgentX task time elapsed: %.2fs", time.perf_counter() - started)
    manager_t.join()
