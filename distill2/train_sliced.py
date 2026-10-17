"""
distill2/train_sliced.py

SliceGPT 切片模型的恢復蒸餾（recovery fine-tune）。

零重訓切片後 k=512 有 5.62°（原 2.87°）、k=384 崩到 34.6°——本腳本把
切片模型放回既有蒸餾 loop 短程微調，看恢復訓練能收回多少。
凍結 DINOv2 patch embed 本體（Path B 教訓：前端一動就塌）；
旋轉 adapter（patch_embed.lin）、blocks、tokens、heads 皆可訓練。

用法（在 distill2/ 下）：
  python train_sliced.py --k 384 --q_ckpt runs\sliced\sliced_k512.pt \
      --data_root D:\vggt-1\vggt-distillation\teacher_dataset_diverse \
      --out_dir runs\sliced\recover_k384 --epochs 6 --img_per_seq 2 --throttle 0.05
（--q_ckpt 只取其中的完整 Q 與 meta，k 由 --k 決定）
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import TeacherDataset
from loss import DistillLoss
from eval import load_student, quick_eval
from train import cosine_warmup
from slicegpt_slice import apply_slices_, AdaptedPatchEmbed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", default=r"runs\distill_b1\best.pt")
    ap.add_argument("--q_ckpt", default=r"runs\sliced\sliced_k512.pt", help="含完整 Q 的切片存檔")
    ap.add_argument("--k", type=int, required=True, help="aggregator 的 k（0=不切）")
    ap.add_argument("--k_pe", type=int, default=0, help="DINOv2 patch embed 的 k（0=不切）")
    ap.add_argument("--k_ch", type=int, default=0, help="camera head trunk 的 k（0=不切）")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps_per_epoch", type=int, default=400)
    ap.add_argument("--img_per_seq", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--eval_scenes", type=int, default=8)
    ap.add_argument("--throttle", type=float, default=0.0)
    ap.add_argument("--train_pe", action="store_true",
                    help="連 DINOv2 本體一起訓練（預設凍結；PE 有切片損傷時的救援選項，"
                         "注意 Path B 教訓：全寬功能完好時微調 PE 會塌）")
    ap.add_argument("--gt_pose", type=int, default=0,
                    help="1：pose 目標改用 CO3D GT 外參（需先跑 make_gt_labels.py）")
    ap.add_argument("--resume_sliced", default=None,
                    help="給已恢復的切片 ckpt 路徑：用 load_sliced 續訓（忽略 base_ckpt/--k*）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tag = f"agg{args.k}" + (f"_pe{args.k_pe}" if args.k_pe else "") + \
          (f"_ch{args.k_ch}" if args.k_ch else "")
    out = Path(args.out_dir or rf"runs\sliced\recover_{tag}")
    out.mkdir(parents=True, exist_ok=True)

    if args.resume_sliced:
        from slicegpt_slice import load_sliced
        ref = torch.load(args.resume_sliced, map_location="cpu")
        meta = ref.get("meta", {})
        qs = {kk: ref[kk] for kk in ("Q", "Q_agg", "Q_pe", "Q_ch") if kk in ref and ref[kk] is not None}
        del ref
        model = load_sliced(args.resume_sliced, "cpu").to(device)
        print(f"已載入切片 ckpt 續訓：{args.resume_sliced}")
    else:
        ref = torch.load(args.q_ckpt, map_location="cpu")
        meta = ref.get("meta", {})
        qs = {kk: ref[kk] for kk in ("Q", "Q_agg", "Q_pe", "Q_ch") if kk in ref}
        del ref

        model = load_student(args.base_ckpt, "cpu")
        apply_slices_(model, qs, k_agg=args.k, k_pe=args.k_pe, k_ch=args.k_ch)
        model = model.to(device)
        print(f"已切片：{tag}")

    # DINOv2 本體預設凍結（adapter 保持可訓練）；PE 已切時 vit 藏在 adapter 鏈裡
    if not args.train_pe:
        pe = model.aggregator.patch_embed
        while isinstance(pe, AdaptedPatchEmbed):
            pe = pe.pe
        n_frozen = 0
        for p in pe.parameters():
            p.requires_grad = False
            n_frozen += p.numel()
        print(f"已凍結 patch_embed（DINOv2 本體）：{n_frozen/1e6:.1f}M 參數")
    else:
        print("⚠ --train_pe：DINOv2 本體一起訓練")

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"可訓練參數：{sum(p.numel() for p in trainable)/1e6:.1f}M")

    loss_fn = DistillLoss().to(device)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    ds = TeacherDataset(args.data_root, "train", args.img_per_seq, args.steps_per_epoch,
                        gt_pose=bool(args.gt_pose))
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=args.num_workers,
                    pin_memory=False, drop_last=True)

    total_steps = args.epochs * args.steps_per_epoch
    warmup = int(total_steps * args.warmup_frac)
    step, best_rot = 0, float("inf")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        run = {}
        for batch in dl:
            lr = cosine_warmup(step, total_steps, warmup, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            batch = {kk: v.to(device) for kk, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype, enabled=(device == "cuda")):
                pred = model(batch["images"])
                total, logs = loss_fn(pred, batch)
            if not torch.isfinite(total):
                print(f"[step {step}] non-finite loss，跳過"); step += 1; continue
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            if args.throttle > 0:
                if device == "cuda":
                    torch.cuda.synchronize()
                time.sleep(args.throttle)
            for kk, v in logs.items():
                run[kk] = run.get(kk, 0.0) + float(v)
            step += 1
            if step % 50 == 0:
                msg = " ".join(f"{kk}={run[kk]/50:.4f}" for kk in
                               ["loss_rot", "loss_transfov", "loss_depth", "objective"] if kk in run)
                print(f"[e{epoch} s{step}] lr={lr:.2e} {msg}", flush=True)
                run = {}
        print(f"epoch {epoch} 完成（{time.time()-t0:.0f}s）", flush=True)

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            m = quick_eval(model, str(Path(args.data_root) / "test"), args.eval_scenes, device, dtype)
            print(f"  [EVAL e{epoch}] rot_err={m['mean_rot_err']:.2f}  rra15={m['rra_15']:.3f}  "
                  f"abs_rel={m['abs_rel']:.4f}  d1={m['delta_1']:.3f}", flush=True)
            save = {"model": model.state_dict(), "meta": meta, "epoch": epoch,
                    "Q_agg": qs.get("Q_agg", qs.get("Q")), "k_agg": args.k,
                    "Q_pe": qs.get("Q_pe"), "k_pe": args.k_pe,
                    "Q_ch": qs.get("Q_ch"), "k_ch": args.k_ch,
                    "note": "sliced+recovered; 用 slicegpt_slice.load_sliced() 載入"}
            torch.save(save, out / "last.pt")
            if m["mean_rot_err"] < best_rot:
                best_rot = m["mean_rot_err"]
                torch.save({**save, "eval": m}, out / "best.pt")
                print(f"  ✓ 新 best（rot_err={best_rot:.2f}），已存 best.pt", flush=True)

    print(f"\n恢復蒸餾完成。最佳 rot_err={best_rot:.2f}")


if __name__ == "__main__":
    main()
