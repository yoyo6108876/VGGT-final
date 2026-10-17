# -*- coding: utf-8 -*-
"""Memory-budget emulation that INCLUDES the CUDA context (paper Appendix D / Table 7).

For each model and each device budget B (GB), cap the PyTorch caching allocator so that
    CUDA-context overhead + allocator reserve  <=  B
then increase the number of frames S until the forward pass runs out of memory.
Reported: the largest S that runs, i.e. what a B-GB card could process in one pass.

Also reports CPU-only latency (fp32, S=2) as a second hardware class.
Run from distill2/:  python measure_budget_incl_context.py
"""
import gc
import json
import platform
import subprocess
import time

import torch

S_LIST = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 96]
BUDGETS = [4, 6, 8, 12, 16]   # 16 GB = the whole card, reported for reference
RES = 518
GB = 1024 ** 3


def smi_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().split(",")
    return float(out[0]), float(out[1])


def load(name, device):
    if name == "teacher":
        from vggt.models.vggt import VGGT
        m = VGGT.from_pretrained("facebook/VGGT-1B").eval().to(device)
    elif name == "pathb":
        from eval import load_student
        m = load_student(r"runs\distill_b1\best.pt", device)
    elif name in ("sliced", "sliced_int8"):
        from slicegpt_slice import load_sliced
        m = load_sliced(r"runs\sliced\recover_agg256_ch512\best.pt", device)
        if name == "sliced_int8":
            from quantize import try_torchao_int8
            m, _ = try_torchao_int8(m)
    else:
        raise ValueError(name)
    # MODE "frozen": parameters frozen as in deployment (autocast then keeps no bf16 weight copies).
    # MODE "paper": parameters left as loaded (requires_grad=True), matching the paper's Fig. 2 / Table 6.
    m.requires_grad_(MODE == "paper")
    rg = any(p.requires_grad for p in m.parameters())
    print(f"[{MODE}] {name}: parameters require grad = {rg}", flush=True)
    return m


@torch.no_grad()
def fwd(model, s, device, dtype):
    x = torch.rand(1, s, 3, RES, RES, device=device, dtype=dtype)
    with torch.autocast("cuda", dtype=dtype, enabled=(device == "cuda")):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()


import sys as _sys
MODE = _sys.argv[2] if len(_sys.argv) > 2 else "frozen"
OUT = rf"reports\budget_incl_context_{MODE}.json"


def save(res):
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


def main():
    import os
    import sys
    models = sys.argv[1].split(",") if len(sys.argv) > 1 else ["sliced_int8", "sliced", "pathb", "teacher"]
    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    res.update({"budgets_gb": BUDGETS, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__})

    # --- CUDA context overhead: device memory used by this process beyond the allocator's reserve
    base_used, total_mib = smi_used_mib()            # before this process touches CUDA
    if base_used > 4500:
        raise SystemExit(f"ABORT: other processes already use {base_used:.0f} MiB of GPU memory; "
                         "budgets above the free memory would be wrong. Close other GPU programs and rerun.")
    torch.cuda.init()
    a = torch.randn(1024, 1024, device="cuda")
    (a @ a).sum().item()                              # cuBLAS handle/workspace
    conv = torch.nn.Conv2d(3, 8, 3).cuda()
    conv(torch.randn(1, 3, 64, 64, device="cuda")).sum().item()   # cuDNN
    torch.cuda.synchronize()
    used_after, _ = smi_used_mib()
    reserved = torch.cuda.memory_reserved() / 2 ** 20
    ctx_mib = (used_after - base_used) - reserved
    del a, conv
    torch.cuda.empty_cache()
    res.update({"other_processes_mib": base_used, "context_mib": round(ctx_mib, 1), "total_mib": total_mib})
    print(f"context overhead ~ {ctx_mib:.0f} MiB (other processes {base_used:.0f} MiB, card {total_mib:.0f} MiB)", flush=True)

    total_bytes = torch.cuda.get_device_properties(0).total_memory
    for name in models:
        model = load(name, "cuda")
        res[name] = {}
        if name != "sliced_int8" and MODE == "frozen":   # device-independent compute measure, S=4
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(display=False) as fc:
                fwd(model, 4, "cuda", torch.bfloat16)
            res[name]["gflops_S4"] = round(fc.get_total_flops() / 1e9, 1)
            print(f"{name:12s} GFLOPs @S=4: {res[name]['gflops_S4']}", flush=True)
            torch.cuda.empty_cache()
        for B in BUDGETS:
            cap = B * GB - ctx_mib * 2 ** 20
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(max(cap, 0) / total_bytes, 0)
            best = 0
            for s in S_LIST:
                try:
                    fwd(model, s, "cuda", torch.bfloat16)
                    best = s
                except torch.cuda.OutOfMemoryError:
                    break
                finally:
                    torch.cuda.empty_cache()
            res[name][str(B)] = best
            print(f"{name:12s} budget {B:2d} GB -> max frames {best}", flush=True)
        torch.cuda.set_per_process_memory_fraction(1.0, 0)
        save(res)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if MODE != "frozen" or "nocpu" in _sys.argv:
        save(res)
        print("saved " + OUT)
        return

    # --- CPU-only latency (fp32), a GPU-less host
    torch.set_grad_enabled(False)
    cpu = {"cpu": platform.processor(), "threads": torch.get_num_threads()}
    for name in ["sliced", "teacher"]:
        model = load(name, "cpu").float()
        fwd(model, 2, "cpu", torch.float32)            # warm-up
        t0 = time.perf_counter()
        for _ in range(3):
            fwd(model, 2, "cpu", torch.float32)
        cpu[name + "_ms_S2"] = round((time.perf_counter() - t0) / 3 * 1000, 1)
        print(f"CPU {name}: {cpu[name + '_ms_S2']} ms (S=2, fp32)", flush=True)
        del model
        gc.collect()
    res["cpu_latency"] = cpu
    save(res)
    print("saved " + OUT)


if __name__ == "__main__":
    main()
