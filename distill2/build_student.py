"""
distill2/build_student.py

從 teacher (VGGT-1B) 切出 student，並「選擇性」複製權重（供消融實驗 P0-4 用）。
可複製元件：
  - patch_embed   : aggregator.patch_embed（DINOv2 ViT-L/14）
  - aggregator    : aggregator 每隔一層的 block + camera/register token
  - camera_head   : 整個相機頭（pose 能力的關鍵）
未複製者：patch_embed → 用 hub 預訓練 DINOv2-L；aggregator / camera_head → 隨機初始化。
depth/point head 一律不複製（student 較小），由蒸餾學。

用法：
  python build_student.py --out student_init.pt --copy all           # = Path B（複製全部）
  python build_student.py --out abl_headonly.pt --copy camera_head   # 只複製相機頭
  python build_student.py --out abl_random.pt   --copy none          # 不複製（patch embed 用 hub 預訓練）
"""

import argparse
import torch

from vggt.models.vggt_student import VGGTStudent

COMPONENTS = ["patch_embed", "aggregator", "camera_head"]


def _copy(dst, src, name):
    missing, unexpected = dst.load_state_dict(src.state_dict(), strict=False)
    n = len(list(src.state_dict().keys()))
    tag = "OK" if not missing and not unexpected else f"missing={len(missing)} unexpected={len(unexpected)}"
    print(f"  [複製 {name}] {n} keys → {tag}")


def parse_copy(s):
    s = s.strip().lower()
    if s in ("all", "*"):
        return set(COMPONENTS)
    if s in ("none", "", "-"):
        return set()
    out = set()
    for c in s.split(","):
        c = c.strip()
        if c not in COMPONENTS:
            raise ValueError(f"未知元件 '{c}'，可用：{COMPONENTS} / all / none")
        out.add(c)
    return out


def build_student(out_path, copy, teacher_name="facebook/VGGT-1B", student_depth=12):
    print(f"複製設定 copy={sorted(copy) or 'none'}")
    copy_pe = "patch_embed" in copy

    student = VGGTStudent(
        img_size=518, patch_size=14, embed_dim=1024, agg_depth=student_depth,
        num_heads=16, patch_embed="dinov2_vitl14_reg",
        pretrained_patch_embed=(not copy_pe),   # 不複製時用 hub 預訓練；複製時待會蓋掉
        enable_camera=True, enable_depth=True, enable_point=True, enable_track=False,
    )

    if copy:  # 只有需要複製時才載入 teacher
        print(f"載入 teacher：{teacher_name} ...")
        teacher = VGGT_load(teacher_name)
        t_agg, s_agg = teacher.aggregator, student.aggregator
        t_depth = t_agg.depth
        stride = t_depth // student_depth

        if copy_pe:
            _copy(s_agg.patch_embed, t_agg.patch_embed, "patch_embed")
        if "aggregator" in copy:
            with torch.no_grad():
                s_agg.camera_token.copy_(t_agg.camera_token)
                s_agg.register_token.copy_(t_agg.register_token)
            for i in range(student_depth):
                j = i * stride
                _copy(s_agg.frame_blocks[i], t_agg.frame_blocks[j], f"frame[{i}]<-t[{j}]")
                _copy(s_agg.global_blocks[i], t_agg.global_blocks[j], f"global[{i}]<-t[{j}]")
        if "camera_head" in copy and student.camera_head is not None and getattr(teacher, "camera_head", None):
            _copy(student.camera_head, teacher.camera_head, "camera_head")

    meta = {"embed_dim": 1024, "agg_depth": student_depth, "patch_embed": "dinov2_vitl14_reg",
            "num_heads": 16, "copy": sorted(copy), "init_from": teacher_name}
    torch.save({"model": student.state_dict(), "meta": meta}, out_path)
    n = sum(p.numel() for p in student.parameters())
    print(f"✓ 已存 {out_path}｜{n/1e6:.1f}M｜copy={sorted(copy) or 'none'}")


def VGGT_load(name):
    from vggt.models.vggt import VGGT
    m = VGGT.from_pretrained(name)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="student_init.pt")
    ap.add_argument("--copy", default="all", help="all / none / 逗號分隔：patch_embed,aggregator,camera_head")
    ap.add_argument("--teacher", default="facebook/VGGT-1B")
    ap.add_argument("--student_depth", type=int, default=12)
    args = ap.parse_args()
    build_student(args.out, parse_copy(args.copy), args.teacher, args.student_depth)


if __name__ == "__main__":
    main()
