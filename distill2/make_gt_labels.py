"""
distill2/make_gt_labels.py

為 teacher_dataset_diverse 的每個 scene 產生 gt_extrinsics.npy（CO3D 官方標註外參，
flip_xy 慣例、與 teacher extrinsics 同座標約定——已由 eval_gt teacher RRA@15=1.000 驗證）。
之後 data.py 以 gt_pose=True 讀取 → pose 目標改為 GT（first-frame 正規化照舊）。
類別標註檔只解析一次（快取），492 場景約數分鐘。

用法：
  python make_gt_labels.py --data_root D:\vggt-1\vggt-distillation\teacher_dataset_diverse \
      --co3d_root D:\vggt-1\vggt-distillation\vggt_input_full
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from eval_gt import pt3d_to_opencv

_CACHE = {}


def _category_ann(co3d_root, category):
    if category not in _CACHE:
        ann = Path(co3d_root) / category / "frame_annotations.jgz"
        if not ann.exists():
            _CACHE[category] = None
        else:
            with gzip.open(ann, "rt", encoding="utf-8") as f:
                data = json.load(f)
            by_seq = {}
            for a in data:
                by_seq.setdefault(a.get("sequence_name"), {})[Path(a["image"]["path"]).name] = a["viewpoint"]
            _CACHE[category] = by_seq
    return _CACHE[category]


def gt_extri(co3d_root, scene_name, source_names, convention):
    if "__" not in scene_name:
        return None
    category, seqid = scene_name.split("__", 1)
    by_seq = _category_ann(co3d_root, category)
    if by_seq is None or seqid not in by_seq:
        return None
    vp_by_name = by_seq[seqid]
    extri = []
    for name in source_names:
        vp = vp_by_name.get(name)
        if vp is None:
            return None
        extri.append(pt3d_to_opencv(vp["R"], vp["T"], convention))
    return np.stack(extri)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--co3d_root", required=True)
    ap.add_argument("--convention", default="flip_xy")
    args = ap.parse_args()

    for split in ("train", "test"):
        ok, skip = 0, 0
        scenes = sorted((Path(args.data_root) / split).iterdir(),
                        key=lambda d: d.name)          # 同類別相鄰 → 快取有效
        for scene in scenes:
            if not (scene / "meta.json").exists():
                continue
            meta = json.load(open(scene / "meta.json", encoding="utf-8"))
            extri = gt_extri(args.co3d_root, meta["scene_name"],
                             meta["source_images"], args.convention)
            if extri is None:
                skip += 1
                print(f"  [skip] {scene.name}")
                continue
            np.save(scene / "gt_extrinsics.npy", extri.astype(np.float32))
            ok += 1
        print(f"[{split}] 已產生 {ok}，skip {skip}", flush=True)


if __name__ == "__main__":
    main()
