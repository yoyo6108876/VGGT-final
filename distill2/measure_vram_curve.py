# -*- coding: utf-8 -*-
"""推論 VRAM vs 幀數 S 曲線 + 同卡最大可處理幀數（OOM 邊界）。

對照教授要求「與現有模型的優勢」：
  teacher VGGT-1B vs Path B(842M) vs 三流切片(466M)
在同一張卡（RTX 5080 16GB）、518×518、bf16、no_grad 下：
  - 各 S 的峰值 VRAM
  - 撐到幾幀才 OOM

用法（在 distill2/ 下）：python measure_vram_curve.py
"""

import gc
import json

import torch

S_LIST = [2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 80, 96]
RES = 518


def load_model(name):
    if name == "teacher":
        from vggt.models.vggt import VGGT
        return VGGT.from_pretrained("facebook/VGGT-1B").eval().cuda()
    if name == "pathb":
        from eval import load_student
        return load_student(r"runs\distill_b1\best.pt", "cuda")
    if name == "sliced":
        from slicegpt_slice import load_sliced
        return load_sliced(r"runs\sliced\recover_agg256_ch512\best.pt", "cuda")
    raise ValueError(name)


@torch.no_grad()
def vram_at(model, s):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    images = torch.rand(1, s, 3, RES, RES, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(images)
    torch.cuda.synchronize()
    del images
    return torch.cuda.max_memory_allocated() / 1e9


def main():
    results = {}
    for name in ["sliced", "pathb", "teacher"]:   # 小→大，teacher 最後（最會 OOM）
        print(f"== {name} ==", flush=True)
        model = load_model(name)
        curve, max_ok = {}, 0
        for s in S_LIST:
            try:
                gb = vram_at(model, s)
                curve[s] = round(gb, 2)
                max_ok = s
                print(f"  S={s:3d}  {gb:5.2f} GB", flush=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  S={s:3d}  OOM ← 上限", flush=True)
                break
        results[name] = {"curve": curve, "max_frames": max_ok}
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\n===== 總表（RTX 5080 16GB, 518x518, bf16, no_grad）=====")
    print(f"{'S':>4} {'teacher(GB)':>12} {'PathB(GB)':>10} {'sliced(GB)':>11}")
    for s in S_LIST:
        row = [results.get(m, {}).get("curve", {}).get(s) for m in ("teacher", "pathb", "sliced")]
        print(f"{s:>4} " + " ".join(f"{v:>11.2f}" if v is not None else f"{'—':>11}" for v in row))
    print("\n同卡最大幀數：teacher={} / PathB={} / sliced={}".format(
        *(results.get(m, {}).get("max_frames", "?") for m in ("teacher", "pathb", "sliced"))))

    with open(r"reports\vram_curve.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("已輸出 reports/vram_curve.json")


if __name__ == "__main__":
    main()
