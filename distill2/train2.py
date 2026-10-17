r"""
distill2/train2.py

二次蒸餾：用已經 work 的 Path B（best.pt）當「線上 teacher」，把「不塌縮」的
camera-token 行為，蒸餾進一個「更小」的 student（預設 DINOv2-S / 384）。

  - response KD：小 student 的 pose/depth vs 原始 1B soft labels（disk）
  - feature KD ：小 student camera token → Path B camera token（逐幀 cosine）
  - teacher（Path B, ~842M）比 1B 小，可與小 student 一起放進 16GB 做 live 蒸餾。

目標：拿回壓縮率（小 student ~110–150M）同時避免 rotation 塌縮。

用法：
  python train2.py --teacher_ckpt runs\distill_b1\best.pt \
      --embed_dim 384 --patch_embed dinov2_vits14_reg \
      --data_root D:\vggt-1\vggt-distillation\teacher_dataset_diverse \
      --out_dir runs\distill2_s
"""

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vggt.models.vggt_student import VGGTStudent
from data import TeacherDataset
from loss2 import DistillLoss2
from eval import build_model, quick_eval


def cosine_warmup(step, total, warmup, base_lr, min_lr=1e-7):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def student_forward(student, images):
    """跑 aggregator + camera/depth head，並額外回傳 camera token（供 feature KD）。"""
    agg, ps = student.aggregator(images)
    cam = agg[-1][:, :, 0]  # [B,S,2*embed]
    with torch.cuda.amp.autocast(enabled=False):
        pose_list = student.camera_head(agg)
        depth, dconf = student.depth_head(agg, images=images, patch_start_idx=ps)
    return {"pose_enc": pose_list[-1], "pose_enc_list": pose_list,
            "depth": depth, "depth_conf": dconf, "camera_token": cam}


@torch.no_grad()
def teacher_cam_token(teacher, images):
    """只跑 teacher 的 aggregator 取 camera token（不需 heads，省算力）。"""
    agg, _ = teacher.aggregator(images)
    return agg[-1][:, :, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", required=True, help="Path B best.pt（當 teacher）")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="runs/distill2_s")
    ap.add_argument("--embed_dim", type=int, default=384, help="小 student 寬度（384=ViT-S）")
    ap.add_argument("--patch_embed", type=str, default="dinov2_vits14_reg")
    ap.add_argument("--agg_depth", type=int, default=12)
    ap.add_argument("--num_heads", type=int, default=0, help="0=自動 embed_dim//64")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps_per_epoch", type=int, default=500)
    ap.add_argument("--img_per_seq", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--feature_weight", type=float, default=3.0)
    ap.add_argument("--struct_weight", type=float, default=4.0, help="跨幀結構對齊（抗塌縮）權重")
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--eval_scenes", type=int, default=8)
    ap.add_argument("--throttle", type=float, default=0.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ── teacher = Path B（凍結）──
    tck = torch.load(args.teacher_ckpt, map_location="cpu")
    tmeta = tck.get("meta", {})
    teacher = build_model(tmeta, device)
    teacher.load_state_dict(tck.get("model", tck), strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher_embed = tmeta.get("embed_dim", 1024)
    print(f"Teacher（Path B）：embed_dim={teacher_embed}，來自 {args.teacher_ckpt}")

    # ── 小 student ──
    nh = args.num_heads or max(1, args.embed_dim // 64)
    student = VGGTStudent(
        img_size=518, patch_size=14, embed_dim=args.embed_dim, agg_depth=args.agg_depth,
        num_heads=nh, patch_embed=args.patch_embed, pretrained_patch_embed=True,
        enable_camera=True, enable_depth=True, enable_point=False, enable_track=False,
    ).to(device)
    smeta = {"embed_dim": args.embed_dim, "agg_depth": args.agg_depth, "num_heads": nh,
             "patch_embed": args.patch_embed}
    n_params = sum(p.numel() for p in student.parameters())
    print(f"Student：embed_dim={args.embed_dim}, patch_embed={args.patch_embed}, "
          f"參數 {n_params/1e6:.1f}M（vs teacher ~{teacher_embed and 842}M）")

    student_cam_dim = 2 * args.embed_dim
    teacher_cam_dim = 2 * teacher_embed
    loss_fn = DistillLoss2(student_cam_dim, teacher_cam_dim,
                           feat_weight=args.feature_weight, struct_weight=args.struct_weight).to(device)

    params = [p for p in student.parameters() if p.requires_grad] + list(loss_fn.proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    ds = TeacherDataset(args.data_root, "train", args.img_per_seq, args.steps_per_epoch)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)

    total_steps = args.epochs * args.steps_per_epoch
    warmup = int(total_steps * args.warmup_frac)
    step = 0
    best_rot = float("inf")

    for epoch in range(args.epochs):
        student.train()
        t0 = time.time(); run = {}
        for batch in dl:
            lr = cosine_warmup(step, total_steps, warmup, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            batch = {k: v.to(device) for k, v in batch.items()}
            imgs = batch["images"]

            with torch.autocast("cuda", dtype=dtype, enabled=(device == "cuda")):
                tcam = teacher_cam_token(teacher, imgs)      # 線上 teacher camera token
                pred = student_forward(student, imgs)
                total, logs = loss_fn(pred, tcam, batch)

            if not torch.isfinite(total):
                print(f"[step {step}] non-finite，跳過"); step += 1; continue
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            if args.throttle > 0:
                if device == "cuda":
                    torch.cuda.synchronize()
                time.sleep(args.throttle)

            for k, v in logs.items():
                run[k] = run.get(k, 0.0) + float(v)
            step += 1
            if step % 50 == 0:
                msg = " ".join(f"{k}={run[k]/50:.4f}" for k in ["loss_rot", "loss_feat", "loss_struct", "loss_depth", "objective"] if k in run)
                print(f"[e{epoch} s{step}] lr={lr:.2e} {msg}"); run = {}

        print(f"epoch {epoch} 完成（{time.time()-t0:.0f}s）")

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            student.eval()
            m = quick_eval(student, str(Path(args.data_root) / "test"), args.eval_scenes, device, dtype)
            print(f"  [EVAL e{epoch}] rot_err={m['mean_rot_err']:.2f}  rra15={m['rra_15']:.3f}  "
                  f"abs_rel={m['abs_rel']:.4f}  d1={m['delta_1']:.3f}")
            torch.save({"model": student.state_dict(), "meta": smeta, "epoch": epoch}, out / "last.pt")
            if m["mean_rot_err"] < best_rot:
                best_rot = m["mean_rot_err"]
                torch.save({"model": student.state_dict(), "meta": smeta, "epoch": epoch, "eval": m}, out / "best.pt")
                print(f"  ✓ 新 best（rot_err={best_rot:.2f}），已存 best.pt（{n_params/1e6:.1f}M）")

    print(f"\n二次蒸餾完成。最佳 rot_err={best_rot:.2f}，student {n_params/1e6:.1f}M")


if __name__ == "__main__":
    main()
