"""
distill2/verify_teacher_copy.py

驗證 build_student.py 的權重複製（Path B teacher-init）：把 student checkpoint 的每個張量，
依 build_student.py --copy all 的對應規則，和 teacher (VGGT-1B) 的原始權重逐位元比對。

對應規則（teacher 24 層 → student 12 層，stride 2）：
  aggregator.patch_embed.*                 <- 同名
  aggregator.camera_token / register_token <- 同名
  aggregator.frame_blocks.{i}.*            <- aggregator.frame_blocks.{2i}.*
  aggregator.global_blocks.{i}.*           <- aggregator.global_blocks.{2i}.*
  camera_head.*                            <- 同名
  depth_head / point_head                  不複製

低記憶體：student 用 torch.load(mmap=True)、teacher 用 safetensors 逐張量讀，不需 GPU。

用法（在 distill2/ 下）：
  python verify_teacher_copy.py --ckpt student_init.pt runs\\distill_b1\\best.pt
預期：student_init.pt 的 copied 張量全部 EQUAL；distill_b1 只有 patch_embed 仍 EQUAL（凍結）。
"""

import argparse
import re
from collections import Counter

import torch
from safetensors import safe_open

GROUPS = ("aggregator.patch_embed", "aggregator.frame_blocks", "aggregator.global_blocks",
          "aggregator.camera_token", "aggregator.register_token", "camera_head",
          "depth_head", "point_head")


def map_key(k, stride):
    """student key -> 對應的 teacher key；None = build_student.py 不複製。"""
    if k.startswith("aggregator.patch_embed.") or k in ("aggregator.camera_token",
                                                        "aggregator.register_token"):
        return k
    m = re.match(r"aggregator\.(frame_blocks|global_blocks)\.(\d+)\.(.*)", k)
    if m:
        return f"aggregator.{m[1]}.{int(m[2]) * stride}.{m[3]}"
    if k.startswith("camera_head."):
        return k
    return None


def group(k):
    for g in GROUPS:
        if k.startswith(g):
            return g
    return k.split(".")[0]


def check(path, teacher_path, stride):
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    sd = ck["model"]
    print(f"\n=== {path}")
    print("meta:", ck.get("meta"))
    stats, bad = Counter(), []
    with safe_open(teacher_path, "pt") as t:
        tkeys = set(t.keys())
        for k, v in sd.items():
            tk = map_key(k, stride)
            if tk is None:
                same = (k in tkeys and list(t.get_slice(k).get_shape()) == list(v.shape)
                        and torch.equal(t.get_tensor(k), v))
                stats[(group(k), "not-copied", "EQUAL-teacher" if same else "differs")] += 1
                continue
            if tk not in tkeys:
                res = "teacher-key-missing"
            else:
                tv = t.get_tensor(tk)
                if tv.shape != v.shape:
                    res = "shape-mismatch"
                else:
                    res = "EQUAL" if torch.equal(tv, v.to(tv.dtype)) else "differs"
            stats[(group(k), "copied", res)] += 1
            if res != "EQUAL":
                bad.append(k)
    for (g, kind, res), n in sorted(stats.items()):
        print(f"  {g:28s} {kind:11s} {res:20s} {n:5d} keys")
    print(f"  copied keys not equal to teacher: {len(bad)}" + (f"  e.g. {bad[:4]}" if bad else ""))
    return len(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="一或多個 student checkpoint")
    ap.add_argument("--teacher", default=None,
                    help="teacher model.safetensors 路徑；預設從 HF 快取取 facebook/VGGT-1B")
    ap.add_argument("--stride", type=int, default=2, help="teacher 層數 / student 層數（24/12=2）")
    args = ap.parse_args()

    teacher = args.teacher
    if teacher is None:
        from huggingface_hub import hf_hub_download
        teacher = hf_hub_download("facebook/VGGT-1B", "model.safetensors")
    print(f"teacher: {teacher}")
    for p in args.ckpt:
        check(p, teacher, args.stride)


if __name__ == "__main__":
    main()
