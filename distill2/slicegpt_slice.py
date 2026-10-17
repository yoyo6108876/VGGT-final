"""
distill2/slicegpt_slice.py

SliceGPT 式寬度切片（真正的權重手術版；探針見 slicegpt_probe.py）。

原理：對殘差流做共享正交旋轉 Q（activation 未中心化二階矩的特徵基底），
把權重摺疊到旋轉座標後，只保留前 k 維：
  - LayerNorm → SlicedNorm（只算 σ）＋ affine/中心化摺進下一個 Linear：
      W·LN(x) + b = [W diag(γ)(Q_k − 1vᵀ/d)] (x̃/σ) + (Wβ + b)，v = Q_kᵀ1
      σ² = (‖x̃‖² − (vᵀx̃)²/d)/d + eps        （x̃ = Q_kᵀx 為 k 維殘差流）
  - LayerScale 摺進 attn.proj / mlp.fc2（同時把輸出旋轉進 Q_k 座標）

可切三條互相獨立的殘差流：
  1. aggregator（1024）：camera/register token 旋轉、patch embed 出口 adapter、
     head 介面把 24 個 tap de-slice 回 1024 再串接 2048（head 本體不動）
  2. DINOv2 patch embed（ViT-L，1024）：conv 輸出通道 / cls / register / mask /
     pos_embed 旋轉、24 個 block 摺疊、最終 LN 摺成 de-slice adapter（輸出回 1024）
  3. camera head trunk（2048）：入口 adapter 進切片座標、4 個 block 摺疊、
     trunk_norm 摺進 pose_branch.fc1；adaLN 調變是 data-dependent 的對角運算
     （不可旋轉），保留在原 2048 基底
切片順序：PE → camera trunk → aggregator（PE 須先於 aggregator 包 adapter）。

正確性 oracle：k=滿維 + 任意正交 Q 時，切片模型必須重現原模型輸出
（--check 用隨機正交 Q 同時驗證三條流，誤差應在 fp32 雜訊等級）。

用法（在 distill2/ 下）：
  python slicegpt_slice.py --check                                    # CPU 等價性測試
  python slicegpt_slice.py --run --ks 512,384,256                     # aggregator 掃 k
  python slicegpt_slice.py --run --ks 256 --k_pe 512 --k_ch 1024 --save_k 256   # 三流合切
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from eval import load_student, quick_eval, _load_scene_images


D = 1024  # 原殘差流寬度


# ──────────────────────────────────────────────
# 校準：各殘差流所有 block 輸出（含首個 block 輸入）合併的未中心化二階矩 → 共享 Q
# ──────────────────────────────────────────────

class _StreamAcc:
    def __init__(self, dim, device):
        self.n = 0
        self.c = torch.zeros(dim, dim, dtype=torch.float64, device=device)

    @torch.no_grad()
    def bump(self, x):
        x = x.reshape(-1, x.shape[-1]).double()
        self.n += x.shape[0]
        self.c += x.T @ x

    def pca(self):
        evals, evecs = torch.linalg.eigh(self.c / self.n)
        order = torch.argsort(evals, descending=True)
        return evecs[:, order].cpu(), evals[order].cpu()   # float64


@torch.no_grad()
def calibrate_streams(model, calib_root, num_scenes, num_frames, device, dtype):
    """一次前向收三條流：aggregator(1024) / DINOv2 PE(1024) / camera trunk(2048)。"""
    agg = model.aggregator
    vit = agg.patch_embed
    ch = model.camera_head
    accs = {"agg": _StreamAcc(D, device),
            "pe": _StreamAcc(vit.embed_dim, device),
            "ch": _StreamAcc(2048, device)}

    handles = [
        agg.frame_blocks[0].register_forward_pre_hook(lambda m, inp: accs["agg"].bump(inp[0])),
        vit.blocks[0].register_forward_pre_hook(lambda m, inp: accs["pe"].bump(inp[0])),
        ch.trunk.register_forward_pre_hook(lambda m, inp: accs["ch"].bump(inp[0])),
    ]
    for blk in list(agg.frame_blocks) + list(agg.global_blocks):
        handles.append(blk.register_forward_hook(lambda m, i, o: accs["agg"].bump(o)))
    for blk in vit.blocks:
        handles.append(blk.register_forward_hook(lambda m, i, o: accs["pe"].bump(o)))
    for blk in ch.trunk:
        handles.append(blk.register_forward_hook(lambda m, i, o: accs["ch"].bump(o)))

    scenes = sorted([d for d in Path(calib_root).iterdir()
                     if d.is_dir() and (d / "meta.json").exists()])[:num_scenes]
    for sc in scenes:
        meta = json.load(open(sc / "meta.json", encoding="utf-8"))
        n = meta["num_images"]
        ids = sorted(set(np.linspace(0, n - 1, min(num_frames, n)).astype(int).tolist()))
        images, _ = _load_scene_images(sc, device, dtype, ids)
        model(images)
    for h in handles:
        h.remove()

    return {name: acc.pca() for name, acc in accs.items()}


# ──────────────────────────────────────────────
# 手術元件
# ──────────────────────────────────────────────

class SlicedNorm(nn.Module):
    """LN 的 σ 部分（中心化/affine 已摺進下一個 Linear）。"""

    def __init__(self, v, d, eps):
        super().__init__()
        self.register_buffer("v", v)
        self.d, self.eps = d, eps

    def forward(self, x):
        s2 = (x.pow(2).sum(-1) - (x @ self.v).pow(2) / self.d) / self.d
        sigma = torch.sqrt(s2.clamp_min(0) + self.eps).unsqueeze(-1)
        return x / sigma


class SlicedAttention(nn.Module):
    """同原 Attention，但輸出 reshape 用 num_heads*head_dim（≠ 輸入維度 k）。"""

    def __init__(self, attn, qkv, proj):
        super().__init__()
        self.num_heads, self.head_dim = attn.num_heads, attn.head_dim
        self.fused_attn = attn.fused_attn
        self.q_norm, self.k_norm, self.rope = attn.q_norm, attn.k_norm, attn.rope
        self.qkv, self.proj = qkv, proj

    def forward(self, x, pos=None):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        return self.proj(x)


class AdaptedPatchEmbed(nn.Module):
    """patch embed 輸出 → k 維旋轉座標。"""

    def __init__(self, pe, lin):
        super().__init__()
        self.pe, self.lin = pe, lin

    def forward(self, images):
        out = self.pe(images)
        if isinstance(out, dict):
            out = out["x_norm_patchtokens"]
        return self.lin(out)


def _lin(weight, bias=None):
    m = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    with torch.no_grad():
        m.weight.copy_(weight.float())
        if bias is not None:
            m.bias.copy_(bias.float())
    return m


def _fold_in(linear, gamma, beta, M):
    """輸入側摺疊：W·LN → (W diag(γ) M, b + Wβ)。M = Q_k − 1vᵀ/d，float64。"""
    W = linear.weight.double()
    b = linear.bias.double() if linear.bias is not None else torch.zeros(W.shape[0], dtype=torch.float64)
    return _lin((W * gamma.unsqueeze(0)) @ M, b + W @ beta)


def _fold_out(linear, ls_gamma, Qk):
    """輸出側摺疊：LayerScale + 旋轉 → (Q_kᵀ diag(ls) W, Q_kᵀ(ls⊙b))。"""
    W = linear.weight.double()
    b = linear.bias.double() if linear.bias is not None else torch.zeros(W.shape[0], dtype=torch.float64)
    return _lin(Qk.T @ (W * ls_gamma.unsqueeze(1)), Qk.T @ (ls_gamma * b))


def _ls_gamma(ls, dim):
    return ls.gamma.double() if hasattr(ls, "gamma") else torch.ones(dim, dtype=torch.float64)


def _qvm(Q, k, d):
    """共用的旋轉幾何量：Q_k、v = Q_kᵀ1、M = Q_k − 1vᵀ/d。"""
    Qk = Q[:, :k].contiguous().double()
    v = Qk.sum(0)
    M = Qk - torch.ones(d, dtype=torch.float64).outer(v) / d
    return Qk, v, M


def _surgery_block(blk, Qk, v, M, d):
    """單一 pre-LN block 的摺疊手術（aggregator / DINOv2 / camera trunk 通用）。"""
    eps1, eps2 = blk.norm1.eps, blk.norm2.eps
    g1, b1 = blk.norm1.weight.double(), blk.norm1.bias.double()
    g2, b2 = blk.norm2.weight.double(), blk.norm2.bias.double()
    ls1, ls2 = _ls_gamma(blk.ls1, d), _ls_gamma(blk.ls2, d)

    qkv = _fold_in(blk.attn.qkv, g1, b1, M)
    proj = _fold_out(blk.attn.proj, ls1, Qk)
    blk.attn = SlicedAttention(blk.attn, qkv, proj)
    blk.mlp.fc1 = _fold_in(blk.mlp.fc1, g2, b2, M)
    blk.mlp.fc2 = _fold_out(blk.mlp.fc2, ls2, Qk)
    blk.norm1 = SlicedNorm(v.float(), d, eps1)
    blk.norm2 = SlicedNorm(v.float(), d, eps2)
    blk.ls1 = nn.Identity()
    blk.ls2 = nn.Identity()


def slice_patch_embed_(model, Q, k):
    """就地切 DINOv2 patch embed（ViT-L）。須在 slice_model_ 之前呼叫。"""
    agg = model.aggregator
    vit = agg.patch_embed
    assert not isinstance(vit, AdaptedPatchEmbed), "PE 切片須在 aggregator 切片之前"
    d = vit.embed_dim
    Qk, v, M = _qvm(Q, k, d)

    with torch.no_grad():
        vit.cls_token = nn.Parameter((vit.cls_token.double() @ Qk).float())
        if vit.register_tokens is not None:
            vit.register_tokens = nn.Parameter((vit.register_tokens.double() @ Qk).float())
        vit.pos_embed = nn.Parameter((vit.pos_embed.double() @ Qk).float())
        mt = nn.Parameter((vit.mask_token.double() @ Qk).float())
        mt.requires_grad_(False)
        vit.mask_token = mt

    conv = vit.patch_embed.proj
    Wf = conv.weight.double().reshape(d, -1)
    nc = nn.Conv2d(conv.in_channels, k, kernel_size=conv.kernel_size,
                   stride=conv.stride, bias=conv.bias is not None)
    with torch.no_grad():
        nc.weight.copy_((Qk.T @ Wf).reshape(k, *conv.weight.shape[1:]).float())
        if conv.bias is not None:
            nc.bias.copy_((Qk.T @ conv.bias.double()).float())
    vit.patch_embed.proj = nc
    vit.patch_embed.embed_dim = k

    for blk in vit.blocks:
        _surgery_block(blk, Qk, v, M, d)

    # 最終 LN → SlicedNorm + affine/de-slice 摺成出口 adapter（輸出回 d=1024）
    g, b = vit.norm.weight.double(), vit.norm.bias.double()
    out_lin = _lin(g.unsqueeze(1) * M, b)            # [d, k]
    vit.norm = SlicedNorm(v.float(), d, vit.norm.eps)
    agg.patch_embed = AdaptedPatchEmbed(vit, out_lin)


def slice_camera_trunk_(model, Q, k):
    """就地切 camera head trunk（2048 流）。adaLN 調變保留在原基底。"""
    ch = model.camera_head
    d = ch.token_norm.normalized_shape[0]            # 2048
    Qk, v, M = _qvm(Q, k, d)

    blocks = list(ch.trunk)
    for blk in blocks:
        _surgery_block(blk, Qk, v, M, d)
    entry = _lin(Qk.T)                               # x @ Q_k（進切片座標）
    ch.trunk = nn.Sequential(entry, *blocks)

    tn = ch.trunk_norm
    ch.pose_branch.fc1 = _fold_in(ch.pose_branch.fc1, tn.weight.double(), tn.bias.double(), M)
    ch.trunk_norm = SlicedNorm(v.float(), d, tn.eps)


def slice_model_(model, Q, k):
    """就地切 aggregator 殘差流。Q: [D, D] float64 正交；保留前 k 行（column）。"""
    Qk, v, M = _qvm(Q, k, D)
    agg = model.aggregator

    with torch.no_grad():
        agg.camera_token = nn.Parameter((agg.camera_token.double() @ Qk).float())
        agg.register_token = nn.Parameter((agg.register_token.double() @ Qk).float())

    pe_lin = _lin(Qk.T)                              # x @ Q_k
    agg.patch_embed = AdaptedPatchEmbed(agg.patch_embed, pe_lin)

    for blk in list(agg.frame_blocks) + list(agg.global_blocks):
        _surgery_block(blk, Qk, v, M, D)

    # head 介面：把 k 維快照 de-slice 回 1024 再串接（head 不動）
    agg.register_buffer("deslice_Q", Qk.T.float())   # [k, D]
    if getattr(agg, "_deslice_handle", None) is not None:
        agg._deslice_handle.remove()

    def deslice_hook(mod, inputs, output):
        lst, psi = output
        QT = mod.deslice_Q
        # .to(t.dtype)：autocast 下 matmul 會降成 bf16，但 head 的 LN 需要與
        # 殘差流一致的 dtype（stock 模型此處是 fp32）
        return ([torch.cat([t[..., :k] @ QT, t[..., k:] @ QT], dim=-1).to(t.dtype)
                 for t in lst], psi)

    agg._deslice_handle = agg.register_forward_hook(deslice_hook)
    return model


def storage_m(module):
    return sum(p.numel() for p in module.state_dict().values()) / 1e6


@torch.no_grad()
def bench_latency(model, device, dtype, frames=4, runs=10, warmup=3):
    import time
    images = torch.rand(1, frames, 3, 518, 518, device=device, dtype=dtype)
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
    return (time.perf_counter() - t0) / runs * 1000.0


# ──────────────────────────────────────────────
# 等價性測試（--check）：隨機正交 Q + k=滿維，三流合切必須重現原模型
# ──────────────────────────────────────────────

@torch.no_grad()
def check(ckpt, data_root):
    device, dtype = "cpu", torch.float32
    model = load_student(ckpt, device)
    scene = sorted([d for d in (Path(data_root) / "test").iterdir()
                    if d.is_dir() and (d / "meta.json").exists()])[0]
    images, _ = _load_scene_images(scene, device, dtype, [0, 5])

    ref = model(images)
    torch.manual_seed(0)
    Q_pe, _ = torch.linalg.qr(torch.randn(D, D, dtype=torch.float64))
    Q_ch, _ = torch.linalg.qr(torch.randn(2048, 2048, dtype=torch.float64))
    Q_agg, _ = torch.linalg.qr(torch.randn(D, D, dtype=torch.float64))
    slice_patch_embed_(model, Q_pe, D)
    slice_camera_trunk_(model, Q_ch, 2048)
    slice_model_(model, Q_agg, D)
    out = model(images)

    for key in ["pose_enc", "depth"]:
        diff = (out[key].float() - ref[key].float()).abs().max().item()
        scale = ref[key].float().abs().max().item()
        print(f"  {key:10s} max|Δ|={diff:.3e}（值域尺度 {scale:.3e}）")
    print("→ 若 max|Δ| 在 1e-3 以下（fp32 重排序雜訊等級）即通過（PE+trunk+agg 三流合切）")


# ──────────────────────────────────────────────
# 校準 + 掃 k + eval（--run）
# ──────────────────────────────────────────────

def apply_slices_(model, ck, k_agg=None, k_pe=None, k_ch=None):
    """依存檔（或覆寫的 k 值）套用三流切片。舊格式（Q/k）僅含 aggregator。"""
    k_pe = ck.get("k_pe", 0) if k_pe is None else k_pe
    k_ch = ck.get("k_ch", 0) if k_ch is None else k_ch
    k_agg = ck.get("k_agg", ck.get("k", 0)) if k_agg is None else k_agg
    if k_pe:
        slice_patch_embed_(model, ck["Q_pe"], k_pe)
    if k_ch:
        slice_camera_trunk_(model, ck["Q_ch"], k_ch)
    if k_agg:
        slice_model_(model, ck.get("Q_agg", ck.get("Q")), k_agg)
    return model


def load_sliced(path, device="cpu"):
    """載入 train_sliced.py / --save_k 存的切片模型（重建結構後灌權重）。"""
    from eval import build_model
    ck = torch.load(path, map_location="cpu")
    m = build_model(ck.get("meta", {}), "cpu")
    m.eval()
    apply_slices_(m, ck)
    missing, unexpected = m.load_state_dict(ck["model"], strict=False)
    missing = [x for x in missing if not x.startswith("track_head.")]
    if missing or unexpected:
        print(f"  [load_sliced 警告] missing={missing[:4]} unexpected={unexpected[:4]}")
    return m.to(device).eval()


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    ks = [int(x) for x in args.ks.split(",")]
    test_root = str(Path(args.data_root) / "test")

    base_meta = torch.load(args.ckpt, map_location="cpu").get("meta", {})
    model = load_student(args.ckpt, device)
    base_total = storage_m(model)
    base_agg_blocks = base_total - storage_m(model.aggregator.patch_embed)

    print(f"校準（train split，{args.calib_scenes} scenes × {args.calib_frames} frames）...", flush=True)
    bases = calibrate_streams(model, str(Path(args.data_root) / "train"),
                              args.calib_scenes, args.calib_frames, device, dtype)
    (Q, evals), (Q_pe, _), (Q_ch, _) = bases["agg"], bases["pe"], bases["ch"]

    rows = []
    base = quick_eval(model, test_root, args.eval_scenes, device, dtype)
    base_lat = bench_latency(model, device, dtype) if args.bench else None
    print(f"baseline: rot_err={base['mean_rot_err']:.2f}° rra15={base['rra_15']:.3f} "
          f"abs_rel={base['abs_rel']:.4f}  storage={base_total:.0f}M"
          + (f"  latency={base_lat:.0f}ms" if base_lat else ""), flush=True)
    rows.append({"k": "原模型", **base, "storage": base_total, "energy": 1.0, "lat": base_lat})
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    tot_e = evals.sum()
    tag_extra = (f"+pe{args.k_pe}" if args.k_pe else "") + (f"+ch{args.k_ch}" if args.k_ch else "")
    for k in ks:
        m = load_student(args.ckpt, "cpu")   # 手術在 CPU（float64 摺疊），完成後才上 GPU
        if args.k_pe:
            slice_patch_embed_(m, Q_pe, args.k_pe)
        if args.k_ch:
            slice_camera_trunk_(m, Q_ch, args.k_ch)
        if k:
            slice_model_(m, Q, k)
        m = m.to(device)
        label = f"agg{k}{tag_extra}"
        met = quick_eval(m, test_root, args.eval_scenes, device, dtype)
        st = storage_m(m)
        lat = bench_latency(m, device, dtype) if args.bench else None
        rows.append({"k": label, **met, "storage": st,
                     "energy": float(evals[:k].sum() / tot_e) if k else float("nan"),
                     "lat": lat})
        print(f"{label}: rot_err={met['mean_rot_err']:.2f}° rra15={met['rra_15']:.3f} "
              f"abs_rel={met['abs_rel']:.4f}  storage={st:.0f}M"
              + (f"  latency={lat:.0f}ms" if lat else ""), flush=True)
        if args.save_k == k:
            out = Path(args.save_dir); out.mkdir(parents=True, exist_ok=True)
            torch.save({"model": m.state_dict(), "meta": base_meta,
                        "Q_agg": Q, "k_agg": k, "Q_pe": Q_pe, "k_pe": args.k_pe,
                        "Q_ch": Q_ch, "k_ch": args.k_ch,
                        "note": "sliced; 用 slicegpt_slice.load_sliced() 載入"},
                       out / f"sliced_{label}.pt")
            print(f"  已存 {out / f'sliced_{label}.pt'}")
        del m
        if device == "cuda":
            torch.cuda.empty_cache()

    md = ["# SliceGPT 切片結果（共享基底、head 介面 de-slice、head 未切）\n",
          f"模型：`{args.ckpt}`。校準 {args.calib_scenes} scenes；eval {args.eval_scenes} test scenes。",
          "storage = state_dict 總元素數（含 304M 凍結 patch embed 與 de-slice 矩陣）。\n",
          "| k | 共享基底能量 | rot_err | RRA@15 | abs_rel | delta_1 | storage (M) | latency (ms) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        e = (f"{r['energy']*100:.2f}%"
             if isinstance(r["energy"], float) and r["energy"] == r["energy"] else "—")
        lat = f"{r['lat']:.0f}" if r.get("lat") else "—"
        md.append(f"| {r['k']} | {e} | {r['mean_rot_err']:.2f}° | {r['rra_15']:.3f} "
                  f"| {r['abs_rel']:.4f} | {r['delta_1']:.3f} | {r['storage']:.0f} | {lat} |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"已輸出 {out}（blocks 部分 base ≈ {base_agg_blocks:.0f}M）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"runs\distill_b1\best.pt")
    ap.add_argument("--data_root", default=r"D:\vggt-1\vggt-distillation\teacher_dataset_diverse")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--ks", default="512,384,256", help="aggregator 的 k 掃描（0=不切）")
    ap.add_argument("--k_pe", type=int, default=0, help="DINOv2 patch embed 的 k（0=不切）")
    ap.add_argument("--k_ch", type=int, default=0, help="camera head trunk 的 k（0=不切；流寬 2048）")
    ap.add_argument("--calib_scenes", type=int, default=8)
    ap.add_argument("--calib_frames", type=int, default=6)
    ap.add_argument("--eval_scenes", type=int, default=12)
    ap.add_argument("--bench", action="store_true", help="附帶量測 S=4 推論延遲")
    ap.add_argument("--save_k", type=int, default=-1)
    ap.add_argument("--save_dir", default=r"runs\sliced")
    ap.add_argument("--out", default=r"reports\slicegpt_slice.md")
    args = ap.parse_args()
    if args.check:
        check(args.ckpt, args.data_root)
    if args.run:
        run(args)


if __name__ == "__main__":
    main()
