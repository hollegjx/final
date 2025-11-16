#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征缓存阶段的轻量封装（供训练脚本触发 Stage1 → Stage2 过渡时调用）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, Any


def run_cache_features(superclass_name: str, model_path: str, cache_dir: str, batch_size: int,
                       num_workers: int, gpu: int, prop_train_labels: float, seed: int) -> None:
    cmd = [
        sys.executable,
        "scripts/cache_features.py",
        "--superclass_name", superclass_name,
        "--model_path", model_path,
        "--auto_find_best", "False",
        "--cache_dir", cache_dir,
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
        "--gpu", str(gpu),
        "--prop_train_labels", str(prop_train_labels),
        "--seed", str(seed),
        "--overwrite",
    ]
    print("🚀 [阶段1 → 特征缓存] 运行命令:")
    print("    ", " ".join(cmd))
    subprocess.run(cmd, check=True)
