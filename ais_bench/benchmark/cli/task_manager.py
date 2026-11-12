from ais_bench.benchmark.cli.argument_parser import ArgumentParser
from ais_bench.benchmark.utils.logging.logger import AISLogger


class TaskManager:
    def __init__(self) -> None:
        self.logger = AISLogger()
        self.args_parser = ArgumentParser()
        self.args = self.args_parser.parse_args()

        # To accelerate cmd "ais_bench -h", we only import config_manager and work_flow after argparser
        from ais_bench.benchmark.cli.config_manager import ConfigManager


        self.config_manager = ConfigManager(self.args)


    def run(self) -> None:
        # search
        if self.args.search:
            self.config_manager.search_configs_location()
            return

        from ais_bench.benchmark.cli.workers import WORK_FLOW, WorkFlowExecutor
        run_mode = self.args.mode
        reuse_without_run = False
        if run_mode in ("perf", "agentx") and self.args.reuse:
            self.logger.warning(
                "Detected --reuse in %s mode. Execution will be skipped and existing "
                "results will be loaded from the reuse work dir.", run_mode,
            )
            if run_mode == "perf":
                run_mode = "perf_viz"
            else:
                # AgentX 无独立 viz 模式，reuse 时只保留汇总 worker。
                reuse_without_run = True
        self.workflow = [worker_class(self.args) for worker_class in WORK_FLOW.get(run_mode)]
        if reuse_without_run:
            self.workflow = [worker for worker in self.workflow
                             if worker.__class__.__name__ != "AgentXRun"]
        if self.args.mode == "agentx" and self.args.dry_run:
            # dry-run 只输出执行计划，不尝试汇总不存在的正式结果。
            self.workflow = [worker for worker in self.workflow
                             if worker.__class__.__name__ != "AgentXViz"]

        # load config
        cfg = self.config_manager.load_config(self.workflow)

        # run
        workflow_executor = WorkFlowExecutor(cfg, self.workflow)
        workflow_executor.execute()