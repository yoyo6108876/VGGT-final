"""
distill2/measure_efficiency.py

量測各模型的效率 / 壓縮指標（對應指導教授 P0-2：輕量化證據表）。
指標：parameters（總/可訓練）、FLOPs、峰值 VRAM、latency、throughput。
涵蓋模型：
  - Teacher          : VGGT-1B
  - Student-conv     : 初版（conv patch embed, embed_dim 512, 12 層）
  - Student-DINOv2-S : DINOv2 ViT-S/14, embed_dim 384, 12 層
  - Student-PathB    : DINOv2 ViT-L/14, embed_dim 1024, 12 層（teacher-init 用的架構）

輸出：終端表格 + efficiency_report.md + efficiency_report.csv

用法：
  python measure_efficiency.py --frames 4 --img_size 518 --runs 20
  python measure_efficiency.py --no_teacher              # 跳過 1B teacher（省 VRAM / 時間）
  python measure_efficiency.py --pathb_ckpt runs/distill_b1/best.pt   # 量測實際訓練權重

注意：
  - FLOPs 由 fvcore 估算（以 MAC 計，注意力等自訂算子可能低估），僅供相對比較。
  - latency / VRAM 在 AMP bf16、無梯度下量測，較貼近實際推論。
  - 每個模型獨立 try/except，單一模型失敗（如 teacher OOM）不影響其他列。
"""

import argparse
import csv
import time
import torch

# vggt 來自 editable install（-dinov2s）；不要動 sys.path
from vggt.models.vggt_student import VGGTStudent


def count_params(m):
    tot = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return tot, tr


def measure_flops(model, images_fp32):
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        fca = FlopCountAnalysis(model, (images_fp32,))
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return float(fca.total())  # fvcore 以 MAC 計
    except Exception as e:
        print(f"    [FLOPs 略過] {type(e).__name__}: {e}")
        return None


@torch.no_grad()
def measure_latency_vram(model, images, device, dtype, runs=20, warmup=5):
    model.eval()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    amp = (device == "cuda")
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
    t1 = time.perf_counter()
    lat_ms = (t1 - t0) / runs * 1000.0
    vram_gb = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else float("nan")
    return lat_ms, vram_gb


def build_models(args):
    """回傳 [(name, build_fn)]，延遲建立以節省記憶體。"""
    specs = []
    if not args.no_teacher:
        def _teacher():
            from vggt.models.vggt import VGGT
            return VGGT.from_pretrained(args.teacher_name)
        specs.append(("Teacher (VGGT-1B)", _teacher))

    specs.append(("Student-conv (512, 12層)",
                  lambda: VGGTStudent(embed_dim=512, agg_depth=12, num_heads=8,
                                      patch_embed="conv", enable_track=False)))
    specs.append(("Student-DINOv2-S (384, 12層)",
                  lambda: VGGTStudent(embed_dim=384, agg_depth=12, num_heads=6,
                                      patch_embed="dinov2_vits14_reg",
                                      pretrained_patch_embed=False, enable_track=False)))

    def _pathb():
        m = VGGTStudent(embed_dim=1024, agg_depth=12, num_heads=16,
                        patch_embed="dinov2_vitl14_reg",
                        pretrained_patch_embed=False, enable_track=False)
        if args.pathb_ckpt:
            ck = torch.load(args.pathb_ckpt, map_location="cpu")
            m.load_state_dict(ck.get("model", ck), strict=False)
        return m
    specs.append(("Student-PathB (1024, 12層, teacher-init)", _pathb))

    # 任意 ckpt（如層剪枝 d8/d6/d4）：從 meta 自動建架構
    # label 可含逗號，故支援「;」作項目分隔
    if args.extra_ckpts:
        sep = ";" if ";" in args.extra_ckpts else ","
        for item in args.extra_ckpts.split(sep):
            label, path = item.split("=", 1)
            specs.append((label.strip(), _make_ckpt_builder(path.strip())))
    return specs


def _make_ckpt_builder(path):
    def _build():
        from eval import build_model
        ck = torch.load(path, map_location="cpu")
        m = build_model(ck.get("meta", {}), "cpu")
        m.load_state_dict(ck.get("model", ck), strict=False)
        return m
    return _build


def fmt(v, suf="", nd=1):
    return "N/A" if v is None or (isinstance(v, float) and v != v) else f"{v:.{nd}f}{suf}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=4, help="序列幀數 S（效率隨幀數變化）")
    ap.add_argument("--img_size", type=int, default=518)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no_teacher", action="store_true", help="跳過 1B teacher")
    ap.add_argument("--teacher_name", type=str, default="facebook/VGGT-1B")
    ap.add_argument("--pathb_ckpt", type=str, default=None, help="載入實際訓練的 Path B 權重來量測")
    ap.add_argument("--extra_ckpts", type=str, default="",
                    help="逗號分隔 label=ckpt 路徑，從 meta 自動建架構（如層剪枝模型）")
    ap.add_argument("--out_prefix", type=str, default="efficiency_report")
    args = ap.parse_args()

    device = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    S, R = args.frames, args.img_size
    try:
        gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    except Exception:
        gpu, device, dtype = "CPU", "cpu", torch.float32

    print(f"\n環境：{gpu} | device={device} | dtype={dtype} | S={S} 幀 | 解析度={R}x{R} | runs={args.runs}\n")

    teacher_params = None
    rows = []
    for name, build in build_models(args):
        print(f"== {name} ==")
        try:
            model = build().to(device).eval()
        except Exception as e:
            print(f"    [建立失敗] {type(e).__name__}: {e}\n")
            rows.append({"model": name, "params_M": None, "trainable_M": None, "gmacs": None,
                         "vram_gb": None, "latency_ms": None, "throughput": None})
            continue

        tot, tr = count_params(model)
        if "Teacher" in name:
            teacher_params = tot

        images = torch.rand(1, S, 3, R, R, device=device)
        # 先量 latency/VRAM（乾淨模型），再算 FLOPs
        # ── fvcore 的 FlopCountAnalysis 會在模型掛 instrumentation，會嚴重拖慢之後的計時，
        #    所以務必先計時、後算 FLOPs（FLOPs 是這個 model 的最後一次使用）。
        try:
            lat, vram = measure_latency_vram(model, images.to(dtype), device, dtype, args.runs, args.warmup)
        except Exception as e:
            print(f"    [latency 量測失敗] {type(e).__name__}: {e}")
            lat, vram = None, None
        gmacs = measure_flops(model, images.float())
        thr = (S / (lat / 1000.0)) if lat else None  # 幀/秒

        rows.append({"model": name, "params_M": tot / 1e6, "trainable_M": tr / 1e6,
                     "gmacs": (gmacs / 1e9) if gmacs else None, "vram_gb": vram,
                     "latency_ms": lat, "throughput": thr})
        print(f"    params={tot/1e6:.1f}M | GMACs={fmt((gmacs/1e9) if gmacs else None)} | "
              f"VRAM={fmt(vram,'GB',2)} | latency={fmt(lat,'ms')} | throughput={fmt(thr,' fps')}\n")

        del model, images
        if device == "cuda":
            torch.cuda.empty_cache()

    # 壓縮比（相對 teacher）；--no_teacher 時用已量測過的 1256.4M（見 reports/efficiency_report.md）
    if teacher_params is None:
        teacher_params = 1256.4e6
    for r in rows:
        r["compress_x"] = (teacher_params / 1e6 / r["params_M"]) if (teacher_params and r["params_M"]) else None

    # ---- 終端表格 ----
    hdr = ["Model", "Params(M)", "Trainable(M)", "GMACs", "VRAM(GB)", "Latency(ms)", "Throughput(fps)", "壓縮x"]
    print("=" * 110)
    print(f"效率 / 壓縮表（S={S}, {R}x{R}, {gpu}, {dtype}）")
    print("=" * 110)
    print("{:<40} {:>10} {:>12} {:>9} {:>9} {:>12} {:>16} {:>7}".format(*hdr))
    for r in rows:
        print("{:<40} {:>10} {:>12} {:>9} {:>9} {:>12} {:>16} {:>7}".format(
            r["model"], fmt(r["params_M"], nd=1), fmt(r["trainable_M"], nd=1),
            fmt(r["gmacs"]), fmt(r["vram_gb"], nd=2), fmt(r["latency_ms"]),
            fmt(r["throughput"]), fmt(r["compress_x"], "x", 1)))
    print("=" * 110)

    # ---- markdown ----
    md = [f"# 效率 / 壓縮量測\n",
          f"- 環境：{gpu}，dtype={dtype}\n- 設定：S={S} 幀、解析度 {R}×{R}、latency 取 {args.runs} 次平均\n",
          f"- FLOPs 以 fvcore 估算（MAC，注意力等自訂算子可能低估），僅供相對比較。\n",
          "| Model | Params(M) | Trainable(M) | GMACs | VRAM(GB) | Latency(ms) | Throughput(fps) | 壓縮× |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["model"], fmt(r["params_M"], nd=1), fmt(r["trainable_M"], nd=1),
            fmt(r["gmacs"]), fmt(r["vram_gb"], nd=2), fmt(r["latency_ms"]),
            fmt(r["throughput"]), fmt(r["compress_x"], "x", 1)))
    with open(args.out_prefix + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    # ---- csv ----
    with open(args.out_prefix + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "params_M", "trainable_M", "gmacs", "vram_gb", "latency_ms", "throughput_fps", "compress_x",
                    "frames", "img_size", "gpu", "dtype"])
        for r in rows:
            w.writerow([r["model"], r["params_M"], r["trainable_M"], r["gmacs"], r["vram_gb"],
                        r["latency_ms"], r["throughput"], r["compress_x"], S, R, gpu, str(dtype)])

    print(f"\n已輸出：{args.out_prefix}.md、{args.out_prefix}.csv")


if __name__ == "__main__":
    main()
