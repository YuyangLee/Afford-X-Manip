import os
from loguru import logger
from datetime import datetime
import yaml


class Helper:
    def __init__(self, cfg, logdir):
        self.update_logdir(logdir)
        self.cfg = cfg
        # with open(os.path.join(self.logdir, "cfg.yml.log"), "w", encoding='utf8') as f:
        #     yaml.safe_dump(dict(cfg), f, default_flow_style=False)

    def update_logdir(self, logdir: str):
        self.logdir = logdir
        os.makedirs(self.logdir, exist_ok=True)

    def get_export_path(self, suffix="result", ext="log", time_format=False):
        filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") if time_format else str(datetime.now().timestamp())
        return os.path.join(self.logdir, f"{filename}_{suffix}.{ext}")

    def get_export_paths(self, suffixes, exts, time_format=False):
        filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") if time_format else str(datetime.now().timestamp())
        return [os.path.join(self.logdir, f"{filename}_{suffix}.{ext}") for suffix, ext in zip(suffixes, exts)]
