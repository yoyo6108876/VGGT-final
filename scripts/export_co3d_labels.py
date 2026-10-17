"""
scripts/export_co3d_labels.py

【用途】
  批次處理 CO3D 資料集，對每個場景執行 VGGT-1B Teacher 推理，
  將結果儲存成 .npy 格式並自動分成 train / test 兩份。

【CO3D 目錄結構】
  co3d_data/
  └── apple/                      ← 物體類別
      ├── 106_12648_23157/         ← 場景 ID
      │   └── images/              ← 這裡面才是影像
      │       ├── frame0001.jpg
      │       └── ...
      └── ...

【使用方式】
  python scripts/export_co3d_labels.py \
      --co3d_root   D:/co3d_data \
      --categories  apple teddybear \
      --out_root    teacher_dataset \
      --save_points \
      --max_scenes  200            # 每個類別最多處理幾個場景（測試用）
      --test_ratio  0.1            # 10% 做為測試集
      --min_images  5              # 場景至少要有幾張影像才處理

【輸出結構】
  teacher_dataset/
  ├── train/
  │   ├── apple__106_12648_23157/
  │   │   ├── images/
  │   │   ├── depth.npy
  │   │   └── ...
  │   └── ...
  └── test/
      └── ...
"""

import argparse
import json
import random
import shutil
import sys
import traceback
from pathlib import Path

# 把專案根目錄加入路徑，讓 Python 找得到 vggt 套件
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ──────────────────────────────────────────────
# 工具函數（與 export_teacher_labels.py 相同）
# ──────────────────────────────────────────────

def collect_image_paths(images_dir: Path) -> list[str]:
    """收集資料夾內所有影像路徑（排序後）。"""
    paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return [str(p) for p in paths]


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def remove_batch_dim(x: torch.Tensor) -> torch.Tensor:
    return x[0]


def save_npy(array, path: Path) -> None:
    if torch.is_tensor(array):
        array = tensor_to_numpy(array)
    np.save(path, array.astype(np.float32))


def save_images_as_png(images: torch.Tensor, out_dir: Path) -> None:
    imgs_np = tensor_to_numpy(images)
    imgs_np = np.clip(imgs_np * 255.0, 0, 255).astype(np.uint8)
    for i in range(imgs_np.shape[0]):
        hwc = np.transpose(imgs_np[i], (1, 2, 0))
        Image.fromarray(hwc).save(out_dir / f"{i:04d}.png")


# ──────────────────────────────────────────────
# 單場景處理
# ──────────────────────────────────────────────

def subsample_images(image_paths: list[str], max_images: int) -> list[str]:
    """
    從場景中均勻抽取 max_images 張影像。
    例如 202 張抽 10 張：每隔約 20 張取一張，確保時間上均勻分布。
    """
    if len(image_paths) <= max_images:
        return image_paths
    indices = [round(i * (len(image_paths) - 1) / (max_images - 1)) for i in range(max_images)]
    return [image_paths[i] for i in indices]


def process_scene(
    model: VGGT,
    image_paths: list[str],
    out_scene: Path,
    device: str,
    dtype: torch.dtype,
    target_size: int,
    save_points: bool,
    scene_name: str,
    max_images: int = 10,
) -> bool:
    """
    對單一場景執行 Teacher 推理並儲存結果。
    回傳 True 代表成功，False 代表跳過（已存在）或失敗。
    """
    # 已處理過則跳過
    if (out_scene / "meta.json").exists():
        return False   # False = skip（不是錯誤）

    # 均勻抽取影像，避免 OOM
    image_paths = subsample_images(image_paths, max_images)

    out_images = out_scene / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    # 影像預處理
    images, original_coords = load_and_preprocess_images_square(
        image_paths, target_size=target_size
    )
    save_images_as_png(images, out_images)
    images = images.to(device=device, dtype=dtype)

    # Teacher 推理
    with torch.no_grad():
        outputs = model(images)

    depth_pred      = outputs.get("depth")
    depth_conf_pred = outputs.get("depth_conf")
    pose_enc_pred   = outputs.get("pose_enc")
    point_pred      = outputs.get("world_points")
    point_conf_pred = outputs.get("world_points_conf")

    if depth_pred is None or depth_conf_pred is None or pose_enc_pred is None:
        raise KeyError("Teacher 缺少必要輸出 (depth/depth_conf/pose_enc)")

    image_hw = tuple(images.shape[-2:])
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc_pred, image_hw)

    depth_pred      = remove_batch_dim(depth_pred).squeeze(-1)   # [S, H, W]
    depth_conf_pred = remove_batch_dim(depth_conf_pred)           # [S, H, W]
    pose_enc_pred   = remove_batch_dim(pose_enc_pred)             # [S, 9]
    extrinsics      = remove_batch_dim(extrinsics)                # [S, 3, 4]
    intrinsics      = remove_batch_dim(intrinsics)                # [S, 3, 3]
    point_mask      = depth_pred > 1e-8                           # [S, H, W]

    save_npy(depth_pred,       out_scene / "depth.npy")
    save_npy(depth_conf_pred,  out_scene / "depth_conf.npy")
    save_npy(point_mask.float(), out_scene / "point_mask.npy")
    save_npy(extrinsics,       out_scene / "extrinsics.npy")
    save_npy(intrinsics,       out_scene / "intrinsics.npy")
    save_npy(pose_enc_pred,    out_scene / "pose_enc.npy")
    save_npy(original_coords,  out_scene / "original_coords.npy")

    saved_files = {
        "depth": "depth.npy",
        "depth_conf": "depth_conf.npy",
        "point_mask": "point_mask.npy",
        "extrinsics": "extrinsics.npy",
        "intrinsics": "intrinsics.npy",
        "pose_enc": "pose_enc.npy",
        "original_coords": "original_coords.npy",
        "world_points": None,
        "world_points_conf": None,
    }

    if save_points and point_pred is not None:
        pt = remove_batch_dim(point_pred)
        save_npy(pt, out_scene / "world_points.npy")
        saved_files["world_points"] = "world_points.npy"
        if point_conf_pred is not None:
            pc = remove_batch_dim(point_conf_pred)
            save_npy(pc, out_scene / "world_points_conf.npy")
            saved_files["world_points_conf"] = "world_points_conf.npy"

    meta = {
        "scene_name":    scene_name,
        "num_images":    len(image_paths),
        "image_paths":   [f"{i:04d}.png" for i in range(len(image_paths))],
        "source_images": [Path(p).name for p in image_paths],
        "teacher_model": "facebook/VGGT-1B",
        "target_size":   target_size,
        "image_hw":      list(image_hw),
        "save_points":   save_points,
        "files":         saved_files,
    }
    with open(out_scene / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return True   # 成功


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="批次處理 CO3D，對每個場景匯出 VGGT-1B Teacher labels。"
    )
    parser.add_argument(
        "--co3d_root", type=str, required=True,
        help="CO3D 資料集根目錄，例如 D:/co3d_data"
    )
    parser.add_argument(
        "--categories", type=str, nargs="+", required=True,
        help="要處理的物體類別，例如 apple teddybear"
    )
    parser.add_argument(
        "--out_root", type=str, default="teacher_dataset",
        help="輸出根目錄（預設：teacher_dataset）"
    )
    parser.add_argument(
        "--save_points", action="store_true",
        help="是否儲存 3D 點雲 (world_points)"
    )
    parser.add_argument(
        "--target_size", type=int, default=518,
        help="預處理影像邊長（必須是 14 的倍數，預設：518）"
    )
    parser.add_argument(
        "--max_scenes", type=int, default=None,
        help="每個類別最多處理幾個場景（測試用；預設：全部）"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1,
        help="測試集比例（預設：0.1 = 10%%）"
    )
    parser.add_argument(
        "--min_images", type=int, default=5,
        help="場景至少要有幾張影像才處理（預設：5）"
    )
    parser.add_argument(
        "--max_images_per_scene", type=int, default=10,
        help="每個場景最多取幾張影像（均勻抽樣；3050 建議 8，5080 建議 20-30，預設：10）"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="train/test 分割的隨機種子（預設：42）"
    )
    args = parser.parse_args()

    # ── 驗證 ──
    if args.target_size % 14 != 0:
        raise ValueError(f"target_size={args.target_size} 不是 14 的倍數。")

    co3d_root = Path(args.co3d_root)
    out_root  = Path(args.out_root)
    train_dir = out_root / "train"
    test_dir  = out_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # ── 設定裝置 ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    print(f"使用裝置：{device}  精度：{dtype}\n")

    # ── 載入 Teacher 模型 ──
    print("載入 Teacher 模型 (facebook/VGGT-1B)...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    print("模型載入完成。\n")

    rng = random.Random(args.seed)

    total_ok = 0
    total_skip = 0
    total_fail = 0

    for category in args.categories:
        cat_dir = co3d_root / category
        if not cat_dir.is_dir():
            print(f"[警告] 找不到類別目錄：{cat_dir}，跳過。")
            continue

        # 收集此類別下所有場景（子目錄 = 場景 ID）
        scene_dirs = sorted(
            d for d in cat_dir.iterdir()
            if d.is_dir() and (d / "images").is_dir()
        )

        # 過濾影像數量不足的場景
        valid_scenes = []
        for sd in scene_dirs:
            imgs = collect_image_paths(sd / "images")
            if len(imgs) >= args.min_images:
                valid_scenes.append((sd, imgs))

        print(f"類別 [{category}]：找到 {len(valid_scenes)} 個有效場景（≥{args.min_images} 張影像）")

        # 限制場景數量
        if args.max_scenes is not None and len(valid_scenes) > args.max_scenes:
            rng.shuffle(valid_scenes)
            valid_scenes = valid_scenes[:args.max_scenes]
            print(f"  → 限制為 {args.max_scenes} 個場景")

        # 打散後分 train / test
        rng.shuffle(valid_scenes)
        n_test  = max(1, round(len(valid_scenes) * args.test_ratio))
        n_train = len(valid_scenes) - n_test
        train_scenes = valid_scenes[:n_train]
        test_scenes  = valid_scenes[n_train:]
        print(f"  train: {len(train_scenes)}  test: {len(test_scenes)}\n")

        for split_name, split_scenes in [("train", train_scenes), ("test", test_scenes)]:
            split_dir = out_root / split_name
            for idx, (scene_dir, image_paths) in enumerate(split_scenes, 1):
                # 場景名稱：類別__場景ID，避免不同類別同名衝突
                scene_name = f"{category}__{scene_dir.name}"
                out_scene  = split_dir / scene_name

                print(f"  [{split_name}] {idx}/{len(split_scenes)}  {scene_name}  ({len(image_paths)} 張) ... ", end="", flush=True)

                try:
                    ok = process_scene(
                        model=model,
                        image_paths=image_paths,
                        out_scene=out_scene,
                        device=device,
                        dtype=dtype,
                        target_size=args.target_size,
                        save_points=args.save_points,
                        scene_name=scene_name,
                        max_images=args.max_images_per_scene,
                    )
                    torch.cuda.empty_cache()
                    if ok:
                        print("完成")
                        total_ok += 1
                    else:
                        print("已存在，跳過")
                        total_skip += 1
                except Exception as e:
                    print(f"失敗：{e}")
                    traceback.print_exc()
                    # 清除未完成的輸出，避免損壞的資料被 Dataset 讀到
                    if out_scene.exists():
                        shutil.rmtree(out_scene)
                    total_fail += 1

    print(f"\n{'='*50}")
    print(f"全部完成")
    print(f"  成功：{total_ok}  跳過（已存在）：{total_skip}  失敗：{total_fail}")
    print(f"  train 目錄：{train_dir}")
    print(f"  test  目錄：{test_dir}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
