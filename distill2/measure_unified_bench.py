# -*- coding: utf-8 -*-
"""統一批次 latency/VRAM 量測（同一 session、同一條件），供論文 Table 3/6 使用。

模型：teacher / PathB(d12) / d8 / d6 / d4 / sliced(agg256+ch512) / sliced+int8
條件：S=4、518×518、bf16 autocast、no_grad、warmup 5、20 次平均。
"""
import gc
import json
import time

import torch

RES, S, RUNS, WARM = 518, 4, 20, 5


def loaders():
    def teacher():
        from vggt.models.vggt import VGGT
        return VGGT.from_pretrained("facebook/VGGT-1B").eval().cuda()

    def student(path):
        from eval import load_student
        return load_student(path, "cuda")

    def sliced():
        from slicegpt_slice import load_sliced
        return load_sliced(r"runs\sliced\recover_agg256_ch512\best.pt", "cuda")

    def sliced_int8():
        from quantize import try_torchao_int8
        m, _ = try_torchao_int8(sliced())
        return m

    return [
        ("teacher", teacher),
        ("pathb_d12", lambda: student(r"runs\distill_b1\best.pt")),
        ("d8", lambda: student(r"runs\prune\d8\best.pt")),
        ("d6", lambda: student(r"runs\prune\d6\best.pt")),
        ("d4", lambda: student(r"runs\prune\d4\best.pt")),
        ("sliced", sliced),
        ("sliced_int8", sliced_int8),
    ]


@torch.no_grad()
def bench(model):
    images = torch.rand(1, S, 3, RES, RES, device="cuda", dtype=torch.bfloat16)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(WARM):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(images)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(RUNS):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(images)
    torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / RUNS * 1000
    vram = torch.cuda.max_memory_allocated() / 1e9
    return lat, vram


def main():
    out = {}
    for name, make in loaders():
        model = make()
        lat, vram = bench(model)
        out[name] = {"latency_ms": round(lat, 1), "vram_gb": round(vram, 2)}
        print(f"{name:12s} latency={lat:6.1f} ms  VRAM={vram:5.2f} GB", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    with open(r"reports\unified_bench.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("已輸出 reports/unified_bench.json")


if __name__ == "__main__":
    main()
