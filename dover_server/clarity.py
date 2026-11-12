"""清晰度推理封装：只加载技术支路，默认使用完整模型。"""
from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

log = logging.getLogger("dover_server.clarity")

_PACKAGE_DIR = Path(__file__).resolve().parent
_DOVER_ROOT = _PACKAGE_DIR / "DOVER"
_DEFAULT_OPT = _DOVER_ROOT / "clarity.yml"

# 保证可 import 上游 dover 包（未 editable 安装时也能用）。
if str(_DOVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_DOVER_ROOT))

_MEAN = torch.FloatTensor([123.675, 116.28, 103.53])
_STD = torch.FloatTensor([58.395, 57.12, 57.375])
# 完整模型技术支路的标定：原始分中心约 -0.08285，尺度约 0.03774。
_TECH_CENTER = -0.08285
_TECH_SCALE = 0.03774


def to_positive_score(raw_score: float) -> float:
    """把技术支路原始分换成 0～1。0.5 大约是普通画质，越高越清晰。"""
    z = (raw_score - _TECH_CENTER) / _TECH_SCALE
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    eased = math.exp(z)
    return eased / (1.0 + eased)


class ClarityEvaluator:
    """常驻加载一次模型，对视频路径反复打分。"""

    def __init__(
        self,
        *,
        opt_path: Path | str | None = None,
        device: str = "cpu",
        weight_path: Path | str | None = None,
    ) -> None:
        from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
        from dover.models import DOVER

        self._sample = spatial_temporal_view_decomposition
        self._sampler_cls = UnifiedFrameSampler

        opt_file = Path(opt_path) if opt_path else _DEFAULT_OPT
        with opt_file.open("r", encoding="utf-8") as handle:
            opt = yaml.safe_load(handle)

        self.device = device
        self.model_name = str(opt.get("name") or "DOVER")
        dopt = opt["data"]["clarity"]["args"]
        self._sample_types = dopt["sample_types"]
        if "technical" not in self._sample_types:
            raise ValueError("配置必须包含 sample_types.technical")
        # 强制只保留技术支路。
        self._sample_types = {"technical": self._sample_types["technical"]}

        self._temporal_samplers: dict[str, Any] = {}
        for stype, sopt in self._sample_types.items():
            if "t_frag" not in sopt:
                self._temporal_samplers[stype] = self._sampler_cls(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                self._temporal_samplers[stype] = self._sampler_cls(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )

        model_args = dict(opt["model"]["args"])
        model_args["backbone_preserve_keys"] = "technical"
        backbone = dict(model_args.get("backbone") or {})
        model_args["backbone"] = {"technical": backbone["technical"]}

        self._evaluator = DOVER(**model_args).to(self.device)
        if weight_path:
            load_path = Path(weight_path)
        else:
            configured = Path(opt["test_load_path"])
            load_path = configured if configured.is_file() else (_DOVER_ROOT / configured)
        if not load_path.is_file():
            raise FileNotFoundError(f"找不到权重文件: {load_path}")
        state = torch.load(str(load_path), map_location=self.device)
        # 权重里还带美学支路，这里只灌技术支路，避免把未加载的层记成异常。
        technical_state = {
            key: value
            for key, value in state.items()
            if key.startswith("technical_")
        }
        missing, unexpected = self._evaluator.load_state_dict(technical_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "清晰度权重与网络不匹配 "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"missing_head={missing[:8]}"
            )
        log.info(
            "清晰度模型已加载 name=%s path=%s device=%s layers=%d",
            self.model_name,
            load_path,
            self.device,
            len(technical_state),
        )
        self._evaluator.eval()

    def score_video(self, video_path: str | Path) -> dict[str, Any]:
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"视频不存在: {path}")

        started = time.time()
        views, _ = self._sample(str(path), self._sample_types, self._temporal_samplers)
        prepared: dict[str, torch.Tensor] = {}
        for key, tensor in views.items():
            num_clips = self._sample_types[key].get("num_clips", 1)
            prepared[key] = (
                ((tensor.permute(1, 2, 3, 0) - _MEAN) / _STD)
                .permute(3, 0, 1, 2)
                .reshape(tensor.shape[0], num_clips, -1, *tensor.shape[2:])
                .transpose(0, 1)
                .to(self.device)
            )

        with torch.no_grad():
            outputs = self._evaluator(prepared)
        if not outputs:
            raise RuntimeError("模型未返回技术分")
        raw_score = float(outputs[0].mean().item())
        cost = round(time.time() - started, 3)
        return {
            "technical_score": to_positive_score(raw_score),
            "technical_raw_score": raw_score,
            "model": self.model_name,
            "device": self.device,
            "cost_time_s": cost,
        }


_singleton: Optional[ClarityEvaluator] = None


def get_evaluator(*, device: str = "cpu", opt_path: Path | str | None = None) -> ClarityEvaluator:
    global _singleton
    if _singleton is None:
        _singleton = ClarityEvaluator(device=device, opt_path=opt_path)
    return _singleton


def score_video(video_path: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    return get_evaluator(device=device).score_video(video_path)
