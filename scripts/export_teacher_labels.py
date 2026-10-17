"""
scripts/export_teacher_labels.py

【用途】
  蒸餾流程的第一步：用 Teacher 模型 (VGGT-1B) 對一個場景的影像進行推理，
  把輸出結果（depth、相機參數等）儲存成 .npy 檔，供後續 Student 訓練使用。

【名詞說明】
  - scene (場景)：一組拍攝同一個空間的影像，例如一個房間的 20 張照片
  - depth map：每個像素的深度值（距相機的距離），形狀 [H, W]
  - extrinsics (外參)：相機在世界座標系的位置與方向，形狀 [3, 4]
  - intrinsics (內參)：相機本身光學參數（焦距等），形狀 [3, 3]
  - pose_enc：把 extrinsics + intrinsics 壓縮成 9D 向量的緊湊格式（VGGT 專用）
  - confidence (信心分數)：模型對自身預測結果的把握程度，值越高越可信

【使用方式】
  python scripts/export_teacher_labels.py \
      --scene_dir path/to/images \
      --out_root  teacher_dataset \
      --save_points          # 可選，是否也儲存 3D 點雲
      --target_size 518      # 預設 518，必須是 14 的倍數

【輸出目錄結構】
  teacher_dataset/
  └── <scene_name>/
      ├── images/            ← 預處理後的影像 PNG
      ├── depth.npy          ← 深度圖 [S, H, W]
      ├── depth_conf.npy     ← 深度信心分數 [S, H, W]
      ├── extrinsics.npy     ← 相機外參 [S, 3, 4]
      ├── intrinsics.npy     ← 相機內參 [S, 3, 3]
      ├── pose_enc.npy       ← 壓縮相機姿態 [S, 9]
      ├── world_points.npy   ← 3D 點雲（可選）[S, H, W, 3]
      ├── world_points_conf.npy ← 點雲信心分數（可選）[S, H, W]
      ├── original_coords.npy   ← 原始影像在預處理後的座標資訊 [S, 6]
      └── meta.json          ← 場景元資料
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# VGGT 自帶的工具函數
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# ──────────────────────────────────────────────
# 工具函數
# ──────────────────────────────────────────────

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def collect_image_paths(scene_dir: Path) -> list[str]:
    """收集資料夾內所有影像的路徑（排序後）。"""
    paths = sorted(
        p for p in scene_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"在 {scene_dir} 找不到任何影像檔案。")
    return [str(p) for p in paths]


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    """把 GPU tensor 轉成 float32 numpy array。"""
    return x.detach().cpu().float().numpy()


def remove_batch_dim(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    移除批次維度（batch dimension）。
    Teacher 推理時 batch size 固定為 1，所以第 0 維可以直接 squeeze。

    例如：shape [1, 10, 518, 518] → [10, 518, 518]
    """
    if x.shape[0] != 1:
        raise ValueError(f"{name} 的 batch size 應為 1，實際為 {x.shape[0]}。")
    return x[0]


def save_npy(array: torch.Tensor | np.ndarray, path: Path) -> None:
    """儲存 numpy 陣列到指定路徑。"""
    if torch.is_tensor(array):
        array = tensor_to_numpy(array)
    np.save(path, array)
    print(f"  已儲存：{path.name}  {array.shape}")


def save_images_as_png(images: torch.Tensor, out_dir: Path) -> None:
    """
    把預處理後的影像 tensor 儲存成 PNG 檔。
    images: [S, 3, H, W]，值域 [0, 1]
    """
    imgs_np = tensor_to_numpy(images)                         # [S, 3, H, W]
    imgs_np = np.clip(imgs_np * 255.0, 0, 255).astype(np.uint8)

    for i in range(imgs_np.shape[0]):
        hwc = np.transpose(imgs_np[i], (1, 2, 0))            # [H, W, 3]
        Image.fromarray(hwc).save(out_dir / f"{i:04d}.png")


def resolve_scene_name(scene_dir: Path, explicit_name: str | None) -> str:
    """
    決定場景名稱：
    - 如果使用者有明確指定 → 直接用
    - 如果資料夾名稱是 'images' → 往上一層取名（避免所有場景都叫 images）
    - 否則 → 用資料夾名稱
    """
    if explicit_name:
        return explicit_name
    if scene_dir.name.lower() == "images" and scene_dir.parent != scene_dir:
        return scene_dir.parent.name
    return scene_dir.name


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="用 VGGT-1B Teacher 模型匯出場景的 soft labels，供 Student 蒸餾訓練使用。"
    )
    parser.add_argument(
        "--scene_dir", type=str, required=True,
        help="包含影像的資料夾路徑（一個場景）。"
    )
    parser.add_argument(
        "--scene_name", type=str, default=None,
        help="輸出場景資料夾的名稱（選填，預設從 scene_dir 推斷）。"
    )
    parser.add_argument(
        "--out_root", type=str, default="teacher_dataset",
        help="Teacher labels 的根輸出目錄（預設：teacher_dataset）。"
    )
    parser.add_argument(
        "--save_points", action="store_true",
        help="是否同時儲存 3D 世界座標點雲（world_points）和信心分數。"
    )
    parser.add_argument(
        "--target_size", type=int, default=518,
        help="預處理後影像的邊長（必須是 14 的倍數，預設：518）。"
    )
    args = parser.parse_args()

    # ── 驗證 target_size ──
    if args.target_size % 14 != 0:
        raise ValueError(
            f"target_size={args.target_size} 不是 14 的倍數。"
            f"請改用 {(args.target_size // 14) * 14} 或 {((args.target_size // 14) + 1) * 14}。"
        )

    # ── 路徑設定 ──
    scene_dir  = Path(args.scene_dir)
    scene_name = resolve_scene_name(scene_dir, args.scene_name)
    out_scene  = Path(args.out_root) / scene_name
    out_images = out_scene / "images"

    print(f"\n{'='*50}")
    print(f"場景名稱：{scene_name}")
    print(f"輸入路徑：{scene_dir}")
    print(f"輸出路徑：{out_scene}")
    print(f"{'='*50}\n")

    # 若輸出目錄已存在則清除（避免舊資料混入）
    if out_scene.exists():
        print(f"偵測到舊資料，清除中：{out_scene}")
        shutil.rmtree(out_scene)
    out_images.mkdir(parents=True, exist_ok=True)

    # ── 收集影像 ──
    image_paths = collect_image_paths(scene_dir)
    print(f"共找到 {len(image_paths)} 張影像。\n")

    # ── 設定 device（GPU 或 CPU）──
    # bfloat16: Ampere 以上 GPU（RTX 3xxx 起）支援的精度格式，速度快且記憶體省
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    print(f"使用裝置：{device}  精度格式：{dtype}\n")

    # ── 載入 Teacher 模型（VGGT-1B）──
    print("載入 Teacher 模型（facebook/VGGT-1B）...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()   # 切換成推理模式，關閉 dropout 等訓練專用層
    print("模型載入完成。\n")

    # ── 影像預處理 ──
    # load_and_preprocess_images_square：將影像 padding 成正方形後 resize 到 target_size
    # 回傳 images [S, 3, H, W] 和 original_coords [S, 6]（記錄原始影像在新空間中的座標）
    print("預處理影像中...")
    images, original_coords = load_and_preprocess_images_square(
        image_paths, target_size=args.target_size
    )
    save_images_as_png(images, out_images)  # 把預處理後的影像存成 PNG

    # 將影像送到 GPU，並轉換成半精度以加速推理
    images = images.to(device=device, dtype=dtype)
    print(f"影像形狀：{tuple(images.shape)}  （S × 3 × H × W）\n")

    # ── Teacher 推理（不計算梯度，節省記憶體）──
    print("執行 Teacher 推理中...")
    with torch.no_grad():
        outputs = model(images)
    print("推理完成。\n")

    # ── 取出各項輸出 ──
    depth_pred      = outputs.get("depth")        # [1, S, H, W, 1]
    depth_conf_pred = outputs.get("depth_conf")   # [1, S, H, W]
    pose_enc_pred   = outputs.get("pose_enc")     # [1, S, 9]
    point_pred      = outputs.get("world_points") # [1, S, H, W, 3]，可能為 None
    point_conf_pred = outputs.get("world_points_conf")  # [1, S, H, W]，可能為 None

    # 必要輸出的存在性檢查
    for name, tensor in [
        ("depth", depth_pred),
        ("depth_conf", depth_conf_pred),
        ("pose_enc", pose_enc_pred),
    ]:
        if tensor is None:
            raise KeyError(f"Teacher 輸出中找不到 '{name}'，請確認模型版本。")

    # ── 把 pose_enc 轉成 extrinsics / intrinsics ──
    # pose_encoding_to_extri_intri：把 9D 壓縮表示解碼回 [3,4] 外參和 [3,3] 內參
    image_hw = tuple(images.shape[-2:])  # (H, W)
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc_pred, image_hw)
    # extrinsics: [1, S, 3, 4]，intrinsics: [1, S, 3, 3]

    # ── 移除 batch 維度（batch size = 1）──
    depth_pred      = remove_batch_dim(depth_pred,      "depth")        # [S, H, W, 1]
    depth_conf_pred = remove_batch_dim(depth_conf_pred, "depth_conf")   # [S, H, W]
    pose_enc_pred   = remove_batch_dim(pose_enc_pred,   "pose_enc")     # [S, 9]
    extrinsics      = remove_batch_dim(extrinsics,      "extrinsics")   # [S, 3, 4]
    intrinsics      = remove_batch_dim(intrinsics,      "intrinsics")   # [S, 3, 3]

    # depth 的最後一維是 channel（固定為 1），squeeze 掉讓形狀變 [S, H, W]
    depth_map = depth_pred.squeeze(-1)  # [S, H, W]

    # point_mask：標記哪些像素有有效深度（>0 的位置）
    # 訓練時只在有效位置計算 loss，避免 padding 或無效區域干擾
    point_mask = depth_map > 1e-8       # [S, H, W]，bool

    # ── 處理 3D 點雲（可選）──
    if args.save_points:
        if point_pred is not None:
            point_pred = remove_batch_dim(point_pred, "world_points")        # [S, H, W, 3]
        else:
            # 若模型沒有直接輸出點雲，則從深度圖和相機參數重建
            print("模型未輸出 world_points，改由 depth + camera params 重建...")
            point_pred = unproject_depth_map_to_point_map(depth_map, extrinsics, intrinsics)

        if point_conf_pred is not None:
            point_conf_pred = remove_batch_dim(point_conf_pred, "world_points_conf")  # [S, H, W]

    # ── 儲存所有 teacher labels ──
    print("儲存 Teacher labels 中...")
    save_npy(depth_map,      out_scene / "depth.npy")
    save_npy(depth_conf_pred, out_scene / "depth_conf.npy")
    save_npy(point_mask,     out_scene / "point_mask.npy")
    save_npy(extrinsics,     out_scene / "extrinsics.npy")
    save_npy(intrinsics,     out_scene / "intrinsics.npy")
    save_npy(pose_enc_pred,  out_scene / "pose_enc.npy")
    save_npy(original_coords, out_scene / "original_coords.npy")

    saved_files = {
        "depth":          "depth.npy",
        "depth_conf":     "depth_conf.npy",
        "point_mask":     "point_mask.npy",
        "extrinsics":     "extrinsics.npy",
        "intrinsics":     "intrinsics.npy",
        "pose_enc":       "pose_enc.npy",
        "original_coords":"original_coords.npy",
        "world_points":   None,
        "world_points_conf": None,
    }

    if args.save_points:
        save_npy(point_pred, out_scene / "world_points.npy")
        saved_files["world_points"] = "world_points.npy"
        if point_conf_pred is not None:
            save_npy(point_conf_pred, out_scene / "world_points_conf.npy")
            saved_files["world_points_conf"] = "world_points_conf.npy"

    # ── 儲存 meta.json（元資料）──
    # 記錄這個場景的基本資訊，讓 Dataset 載入時可以核對
    meta = {
        "scene_name":     scene_name,
        "num_images":     len(image_paths),
        "image_paths":    [f"{i:04d}.png" for i in range(len(image_paths))],
        "source_images":  [Path(p).name for p in image_paths],
        "teacher_model":  "facebook/VGGT-1B",
        "target_size":    args.target_size,
        "image_hw":       list(image_hw),
        "save_points":    args.save_points,
        "files":          saved_files,
    }

    with open(out_scene / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  已儲存：meta.json\n")

    print(f"{'='*50}")
    print(f"完成！全部結果已儲存至：{out_scene}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
