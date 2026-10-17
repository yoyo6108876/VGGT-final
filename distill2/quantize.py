"""
distill2/quantize.py

把已經 work 的 Path B（best.pt）量化，取得「部署層級」壓縮（模型大小 / VRAM / latency），
並用 eval 確認 rotation 精度不掉。這條路不改架構、不碰塌縮風險，是壓縮的保底方案。

量化方法（自動選）：
  1) torchao int8 weight-only（GPU，推薦）  ── pip install torchao
  2) PyTorch 動態量化 int8（CPU 後備，較慢；只量 nn.Linear）

輸出：量化前後對照表 + quantize_report.md，並存 quantized_int8.pt。

用法：
  python quantize.py --ckpt runs/distill_b1/best.pt \
      --data_root D:/vggt-1/vggt-distillation/teacher_dataset_diverse/test --num_scenes 8
  python quantize.py --ckpt runs/distill_b1/best.pt --method dynamic   # 強制 CPU 動態量化
"""

import argparse
import io
import time

import torch

from eval import load_student, quick_eval


def load_any(ckpt, device):
    """一般 ckpt 走 load_student；含 Q 矩陣的切片 ckpt 走 load_sliced。"""
    ck = torch.load(ckpt, map_location="cpu")
    sliced = any(k in ck for k in ("Q", "Q_agg"))
    del ck
    if sliced:
        from slicegpt_slice import load_sliced
        return load_sliced(ckpt, device)
    return load_student(ckpt, device)


def size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


@torch.no_grad()
def latency_vram(model, images, device, dtype, runs=15, warmup=4):
    model.eval()
    amp = (device == "cuda")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    for _ in range(warmup):
        with torch.autocast("cuda", dtype=dtype, enabled=amp):
            model(images)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        with torch.autocast("cuda", dtype=dtype, enabled=amp):
            model(images)
    if device == "cuda":
        torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / runs * 1000
    vram = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else float("nan")
    return lat, vram


def try_torchao_int8(model):
    """GPU int8 weight-only；相容新舊 torchao API；回傳量化後模型或 None。"""
    try:
        from torchao.quantization import quantize_
        try:
            from torchao.quantization import int8_weight_only          # 舊版
            cfg = int8_weight_only()
        except ImportError:
            from torchao.quantization import Int8WeightOnlyConfig      # 新版
            cfg = Int8WeightOnlyConfig()
        quantize_(model, cfg)
        return model, "torchao int8 weight-only (GPU)"
    except Exception as e:
        print(f"  [torchao 不可用] {type(e).__name__}: {e}")
        return None, None


def dynamic_int8(model):
    """CPU 動態量化（只量 nn.Linear）。"""
    qm = torch.ao.quantization.quantize_dynamic(
        model.cpu().eval(), {torch.nn.Linear}, dtype=torch.qint8)
    return qm, "PyTorch dynamic int8 (CPU, Linear only)"


def fmt(v, suf="", nd=1):
    return "N/A" if v is None or (isinstance(v, float) and v != v) else f"{v:.{nd}f}{suf}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True, help="test split 目錄（量 rot_err 用）")
    ap.add_argument("--num_scenes", type=int, default=8)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=518)
    ap.add_argument("--method", choices=["auto", "torchao", "dynamic"], default="auto")
    ap.add_argument("--out", default="quantized_int8.pt")
    ap.add_argument("--report", default="quantize_report.md")
    args = ap.parse_args()

    has_cuda = torch.cuda.is_available()
    dev = "cuda" if has_cuda else "cpu"
    dtype = torch.bfloat16 if has_cuda else torch.float32
    S, R = args.frames, args.img_size
    imgs = torch.rand(1, S, 3, R, R)

    # ── 基準（未量化）──
    print("載入基準模型 ...")
    base = load_any(args.ckpt, dev)
    base_params = sum(p.numel() for p in base.parameters()) / 1e6
    base_size = size_mb(base)
    print("量測基準 rot_err ...")
    base_eval = quick_eval(base, args.data_root, args.num_scenes, dev, dtype)
    base_lat, base_vram = latency_vram(base, imgs.to(dev, dtype), dev, dtype)
    print(f"  基準：{base_params:.1f}M | {base_size:.0f}MB | rot_err={base_eval['mean_rot_err']:.2f} | "
          f"VRAM={fmt(base_vram,'GB',2)} | {fmt(base_lat,'ms')}")

    # ── 量化 ──
    qmodel, qname, qdev, qdtype = None, None, dev, dtype
    if args.method in ("auto", "torchao") and has_cuda:
        m = load_any(args.ckpt, "cuda")
        qmodel, qname = try_torchao_int8(m)
    if qmodel is None and args.method in ("auto", "dynamic"):
        m = load_any(args.ckpt, "cpu")
        qmodel, qname = dynamic_int8(m)
        qdev, qdtype = "cpu", torch.float32
    if qmodel is None:
        print("量化失敗（torchao 不可用且未選 dynamic）。可 pip install torchao 或加 --method dynamic。")
        return

    print(f"\n量化方法：{qname}")
    q_size = size_mb(qmodel)
    print("量測量化後 rot_err ...")
    q_eval = quick_eval(qmodel, args.data_root, args.num_scenes, qdev, qdtype)
    try:
        q_lat, q_vram = latency_vram(qmodel, imgs.to(qdev, qdtype), qdev, qdtype)
    except Exception as e:
        print(f"  [latency 略過] {type(e).__name__}: {e}"); q_lat, q_vram = None, None
    torch.save({"model": qmodel.state_dict(), "quant": qname}, args.out)

    # ── 報告 ──
    rows = [
        ["基準 (未量化)", fmt(base_size, "MB", 0), fmt(base_eval["mean_rot_err"], "°", 2),
         fmt(base_eval["abs_rel"], "", 3), fmt(base_vram, "GB", 2), fmt(base_lat, "ms")],
        [qname, fmt(q_size, "MB", 0), fmt(q_eval["mean_rot_err"], "°", 2),
         fmt(q_eval["abs_rel"], "", 3), fmt(q_vram, "GB", 2), fmt(q_lat, "ms")],
    ]
    hdr = ["模型", "大小", "rot_err", "abs_rel", "VRAM", "latency"]
    print("\n" + "=" * 96)
    print("{:<34} {:>8} {:>9} {:>9} {:>9} {:>10}".format(*hdr))
    for r in rows:
        print("{:<34} {:>8} {:>9} {:>9} {:>9} {:>10}".format(*r))
    print("=" * 96)
    ratio = base_size / q_size if q_size else float("nan")
    print(f"模型大小壓縮：{ratio:.2f}×（{base_size:.0f}MB → {q_size:.0f}MB）")

    with open(args.report, "w", encoding="utf-8") as f:
        f.write("# 量化壓縮報告\n\n")
        f.write(f"- 基準 checkpoint：{args.ckpt}\n- 量化方法：{qname}\n")
        f.write(f"- 設定：S={S}、{R}×{R}、rot_err 取 {args.num_scenes} 場景平均\n\n")
        f.write("| 模型 | 大小 | rot_err | abs_rel | VRAM | latency |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")
        f.write(f"\n模型大小壓縮：**{ratio:.2f}×**（{base_size:.0f}MB → {q_size:.0f}MB），"
                f"rot_err {base_eval['mean_rot_err']:.2f}° → {q_eval['mean_rot_err']:.2f}°。\n")
    print(f"已輸出：{args.report}、" + args.out)


if __name__ == "__main__":
    main()
