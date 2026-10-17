# -*- coding: utf-8 -*-
"""int8 weight-only 版切片模型的 VRAM vs S 曲線（補 Table 6 的 Max frames 實測）。"""
import gc
import json

import torch

from slicegpt_slice import load_sliced
from quantize import try_torchao_int8

S_LIST = [2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 80, 96]


@torch.no_grad()
def vram_at(model, s):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    images = torch.rand(1, s, 3, 518, 518, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(images)
    torch.cuda.synchronize()
    del images
    return torch.cuda.max_memory_allocated() / 1e9


def main():
    m = load_sliced(r"runs\sliced\recover_agg256_ch512\best.pt", "cuda")
    m, name = try_torchao_int8(m)
    assert m is not None, "torchao int8 失敗"
    print(f"[{name}]", flush=True)
    curve, max_ok = {}, 0
    for s in S_LIST:
        try:
            gb = vram_at(m, s)
            curve[s] = round(gb, 2)
            max_ok = s
            print(f"  S={s:3d}  {gb:5.2f} GB", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  S={s:3d}  OOM", flush=True)
            break
    print(f"int8 sliced：16GB 預算內最大 S = {max([s for s, g in curve.items() if g <= 16.0], default=0)}")
    with open(r"reports\vram_curve_int8.json", "w", encoding="utf-8") as f:
        json.dump({"curve": curve, "max_frames_alloc": max_ok}, f, indent=1)
    print("已輸出 reports/vram_curve_int8.json")


if __name__ == "__main__":
    main()
