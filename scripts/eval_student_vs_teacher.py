"""
scripts/eval_student_vs_teacher.py

【用途】
  蒸餾流程的最後一步：評估 Student 模型的表現，
  並和 Teacher 的 soft labels 做比較，量化蒸餾的效果。

【評估的三個維度】
  1. 相機姿態誤差：Student 預測的相機位置/方向有多準確？
  2. 深度圖誤差：Student 預測的深度和 Teacher 相差多少？
  3. 整體壓縮效益：參數量降低了多少？推理速度快了多少？

【名詞說明】
  - RRA@15°（Relative Rotation Accuracy at 15°）：
    預測旋轉誤差 < 15° 的幀數比例，值越高越好
  - Abs Rel（Absolute Relative Error）：
    |預測深度 - 目標深度| / 目標深度 的平均值，值越低越好
  - RMSE（Root Mean Squared Error，均方根誤差）：
    sqrt(mean((預測 - 目標)²))，對大誤差更敏感，值越低越好
  - δ < 1.25（Delta threshold）：
    max(預測/目標, 目標/預測) < 1.25 的像素比例，值越高越好
  - 推理時間（Inference Time）：處理一個 batch 所需的毫秒數
  - 壓縮比（Compression Ratio）：Teacher 參數量 / Student 參數量

【使用方式】
  python scripts/eval_student_vs_teacher.py \
      --ckpt_path  training/logs/distill_v1/ckpts/checkpoint.pth \
      --data_dir   teacher_dataset \
      --num_scenes 10                 # 評估幾個場景（-1 表示全部）
      --device     cuda

【輸出】
  - 印出每個場景的指標
  - 最後印出所有場景的平均指標
  - 儲存結果到 eval_results.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── import vggt ──
# ⚠ 重要：不要把 -yo 插進 sys.path 最前面！否則 `import vggt` 會載入 -yo/vggt
#   （trunk_depth=2、只有 conv），但訓練實際用的是 editable-install 的
#   -dinov2s/vggt（trunk_depth=4、支援 DINOv2 patch embed）。
#   保持乾淨：直接用 editable install 解析 vggt，確保 eval 與訓練同源。
import sys

from vggt.models.vggt import VGGT
from vggt.models.vggt_student import VGGTStudent
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# ──────────────────────────────────────────────
# 模型載入
# ──────────────────────────────────────────────

def load_student(ckpt_path: str, device: str) -> VGGTStudent:
    """
    從 checkpoint 載入訓練好的 Student 模型。

    checkpoint 是訓練過程中存下來的模型狀態（所有參數的數值），
    副檔名通常是 .pth。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # checkpoint 裡可能有 model / state_dict 等不同格式
    state_dict = ckpt.get("model", ckpt)

    import re

    # 從 state_dict 推斷 embed_dim（檢查 aggregator 的第一個線性層維度）
    embed_dim = 512  # 預設值
    for key, val in state_dict.items():
        if "aggregator.frame_blocks.0.attn.qkv.weight" in key:
            embed_dim = val.shape[1]  # qkv weight: [3*D, D] → D 是 embed_dim
            break

    # ── 偵測 patch_embed 種類（conv vs DINOv2）──
    # conv patch embed 只有少數參數；DINOv2 是完整 ViT，patch_embed 下有大量參數。
    n_pe = sum(1 for k in state_dict if k.startswith("aggregator.patch_embed."))
    DINOV2_BY_DIM = {384: "dinov2_vits14_reg", 768: "dinov2_vitb14_reg", 1024: "dinov2_vitl14_reg"}
    is_dinov2 = n_pe > 10 and embed_dim in DINOV2_BY_DIM

    if is_dinov2:
        patch_embed = DINOV2_BY_DIM[embed_dim]
        num_heads = embed_dim // 64
        print(f"  [架構] DINOv2 patch embed = {patch_embed}, embed_dim={embed_dim}, num_heads={num_heads}")
        student = VGGTStudent(
            embed_dim=embed_dim,
            num_heads=num_heads,
            patch_embed=patch_embed,
            pretrained_patch_embed=False,   # 權重從 ckpt 載入，不需重新下載
        ).to(device)
    else:
        print(f"  [架構] conv patch embed, embed_dim={embed_dim}")
        student = VGGTStudent(embed_dim=embed_dim).to(device)

    # ── 依 checkpoint 推斷 camera_head trunk 層數，並重建以匹配 ──
    # 否則 VGGTStudent 預設 trunk_depth=2，但 checkpoint 可能是用 4 訓練的，
    # 載入時後面幾層會被當成「多餘 key」丟掉 → camera head 被截斷 → 旋轉評估失真。
    ckpt_trunk = -1
    for key in state_dict:
        m = re.match(r"camera_head\.trunk\.(\d+)\.", key)
        if m:
            ckpt_trunk = max(ckpt_trunk, int(m.group(1)))
    ckpt_trunk_depth = ckpt_trunk + 1
    if student.camera_head is not None and ckpt_trunk_depth > 0:
        cur_depth = len(student.camera_head.trunk)
        if ckpt_trunk_depth != cur_depth:
            from vggt.heads.camera_head import CameraHead
            print(f"  [架構對齊] camera_head trunk_depth: 模型預設 {cur_depth} → 依 checkpoint 重建為 {ckpt_trunk_depth}")
            num_heads = student.camera_head.trunk[0].attn.num_heads
            student.camera_head = CameraHead(
                dim_in=2 * embed_dim,
                trunk_depth=ckpt_trunk_depth,
                num_heads=num_heads,
            ).to(device)

    missing, unexpected = student.load_state_dict(state_dict, strict=False)

    # 過濾掉與本次評估無關的 key（track_head 等未啟用的 head），只警告真正的缺漏
    missing = [k for k in missing if not k.startswith(("track_head.",))]
    unexpected = [k for k in unexpected if not k.startswith(("track_head.",))]
    if missing:
        print(f"  [警告] 以下 key 在 checkpoint 中找不到：{missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  [警告] checkpoint 有多餘的 key：{unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    if not missing and not unexpected:
        print("  [載入 OK] 模型與 checkpoint 完全匹配")

    student.eval()
    return student


# ──────────────────────────────────────────────
# 資料載入
# ──────────────────────────────────────────────

def load_scene_data(scene_dir: Path, device: str, dtype: torch.dtype):
    """
    從 teacher_dataset 的一個場景目錄載入影像和 teacher labels。

    Returns:
        images:       [1, S, 3, H, W] tensor（已送到 device）
        teacher_depth:[S, H, W] numpy
        teacher_conf: [S, H, W] numpy
        teacher_pose: [S, 9] numpy
        point_mask:   [S, H, W] numpy（bool）
        image_hw:     (H, W)
    """
    meta_path = scene_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"找不到 meta.json：{meta_path}")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    image_paths = [str(scene_dir / "images" / p) for p in meta["image_paths"]]

    # 讀取影像（使用和 export 時相同的預處理方式）
    images, _ = load_and_preprocess_images_square(
        image_paths, target_size=meta.get("target_size", 518)
    )
    # images: [S, 3, H, W] → 加上 batch 維度 → [1, S, 3, H, W]
    images = images.unsqueeze(0).to(device=device, dtype=dtype)
    image_hw = tuple(images.shape[-2:])

    # 載入 teacher labels
    teacher_depth = np.load(scene_dir / "depth.npy")      # [S, H, W]
    teacher_conf  = np.load(scene_dir / "depth_conf.npy") # [S, H, W]
    teacher_pose  = np.load(scene_dir / "pose_enc.npy")   # [S, 9]
    point_mask    = np.load(scene_dir / "point_mask.npy") # [S, H, W]

    return images, teacher_depth, teacher_conf, teacher_pose, point_mask, image_hw


# ──────────────────────────────────────────────
# 評估指標計算
# ──────────────────────────────────────────────

def rotation_error_deg(R_pred: np.ndarray, R_target: np.ndarray) -> float:
    """
    計算兩個旋轉矩陣之間的角度誤差（單位：度）。

    【數學原理】
    兩個旋轉的差 = R_pred @ R_target.T（一個旋轉的逆）
    如果兩者完全相同，結果是單位矩陣（無旋轉）
    用 trace（對角線元素的和）可以求出旋轉角度：
      cos(angle) = (trace - 1) / 2

    Args:
        R_pred:   [3, 3] 預測旋轉矩陣
        R_target: [3, 3] 目標旋轉矩陣
    Returns:
        旋轉誤差（度），範圍 [0, 180]
    """
    R_rel = R_pred @ R_target.T                              # 相對旋轉
    cos_angle = (np.trace(R_rel) - 1.0) / 2.0               # cos(θ)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)               # 防止數值誤差超出 [-1, 1]
    angle_rad = np.arccos(cos_angle)
    return float(np.degrees(angle_rad))


def depth_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    """
    計算深度估計的多項評估指標。

    Args:
        pred:   [S, H, W] Student 預測的深度圖
        target: [S, H, W] Teacher 的深度圖（作為目標）
        mask:   [S, H, W] bool，有效像素遮罩
    Returns:
        dict，包含以下指標：
        - abs_rel：絕對相對誤差（越小越好）
        - rmse：均方根誤差（越小越好）
        - delta_1：δ < 1.25 的像素比例（越大越好）
        - delta_2：δ < 1.25² 的像素比例（越大越好）
    """
    # 只計算有效像素（teacher depth > 0 且在 mask 內）
    valid = mask.astype(bool) & (target > 1e-6) & (pred > 1e-6)

    if valid.sum() < 100:
        return {"abs_rel": float("nan"), "rmse": float("nan"),
                "delta_1": float("nan"), "delta_2": float("nan")}

    p = pred[valid].astype(np.float64)
    t = target[valid].astype(np.float64)

    # Abs Rel：|pred - target| / target 的平均值
    abs_rel = float(np.mean(np.abs(p - t) / t))

    # RMSE：sqrt(mean((pred - target)²))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))

    # δ threshold：max(pred/target, target/pred) < 1.25^n 的比例
    ratio = np.maximum(p / t, t / p)
    delta_1 = float(np.mean(ratio < 1.25))
    delta_2 = float(np.mean(ratio < 1.25 ** 2))

    return {
        "abs_rel": abs_rel,
        "rmse":    rmse,
        "delta_1": delta_1,
        "delta_2": delta_2,
    }


def camera_metrics(pred_pose: np.ndarray, target_pose: np.ndarray, image_hw: tuple) -> dict:
    """
    計算相機姿態估計的誤差指標。

    Args:
        pred_pose:   [S, 9] Student 預測的 pose_enc
        target_pose: [S, 9] Teacher 的 pose_enc
        image_hw:    (H, W) 影像尺寸（用於還原 intrinsics）
    Returns:
        dict，包含：
        - rra_15：旋轉誤差 < 15° 的幀數比例（越大越好）
        - mean_rot_err：平均旋轉誤差（度，越小越好）
        - mean_trans_err：平均平移誤差（L1，越小越好）
        - focal_rel_err：焦距相對誤差（越小越好）
    """
    # 把 pose_enc 轉換成 tensor 以使用 VGGT 的工具函數
    pred_t  = torch.from_numpy(pred_pose).float().unsqueeze(0)    # [1, S, 9]
    tgt_t   = torch.from_numpy(target_pose).float().unsqueeze(0)  # [1, S, 9]

    pred_extri,  pred_intri  = pose_encoding_to_extri_intri(pred_t,  image_hw)
    tgt_extri,   tgt_intri   = pose_encoding_to_extri_intri(tgt_t,   image_hw)

    # [S, 3, 4] → numpy
    pred_extri = pred_extri[0].numpy()   # [S, 3, 4]
    tgt_extri  = tgt_extri[0].numpy()
    pred_intri = pred_intri[0].numpy()   # [S, 3, 3]
    tgt_intri  = tgt_intri[0].numpy()

    S = pred_extri.shape[0]
    rot_errors = []
    trans_errors = []

    for i in range(S):
        R_pred = pred_extri[i, :3, :3]
        R_tgt  = tgt_extri[i, :3, :3]
        rot_errors.append(rotation_error_deg(R_pred, R_tgt))

        t_pred = pred_extri[i, :3, 3]
        t_tgt  = tgt_extri[i, :3, 3]
        trans_errors.append(float(np.abs(t_pred - t_tgt).mean()))

    rot_errors   = np.array(rot_errors)
    trans_errors = np.array(trans_errors)

    # RRA@15°：旋轉誤差 < 15° 的比例
    rra_15 = float(np.mean(rot_errors < 15.0))

    # 焦距相對誤差：|fx_pred - fx_target| / fx_target
    fx_pred = pred_intri[:, 0, 0]
    fx_tgt  = tgt_intri[:, 0, 0]
    focal_rel = float(np.mean(np.abs(fx_pred - fx_tgt) / (fx_tgt + 1e-8)))

    return {
        "rra_15":        rra_15,
        "mean_rot_err":  float(rot_errors.mean()),
        "mean_trans_err":float(trans_errors.mean()),
        "focal_rel_err": focal_rel,
    }


# ──────────────────────────────────────────────
# 單一場景評估
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate_scene(
    student: VGGTStudent,
    scene_dir: Path,
    device: str,
    dtype: torch.dtype,
) -> dict:
    """
    對一個場景執行 Student 推理，計算並回傳所有評估指標。

    @torch.no_grad()：推理時不需要計算梯度，省記憶體、加速
    """
    # ── 載入資料 ──
    images, teacher_depth, teacher_conf, teacher_pose, point_mask, image_hw = \
        load_scene_data(scene_dir, device, dtype)

    # ── Student 推理（記錄推理時間）──
    torch.cuda.synchronize() if device == "cuda" else None
    t_start = time.perf_counter()

    outputs = student(images)

    torch.cuda.synchronize() if device == "cuda" else None
    inference_ms = (time.perf_counter() - t_start) * 1000.0

    # ── 取出 Student 的預測結果 ──
    pred_depth = outputs["depth"]     # [1, S, H, W, 1]
    pred_pose  = outputs["pose_enc"]  # [1, S, 9]

    # 移除 batch 維度和 channel 維度
    pred_depth = pred_depth[0].squeeze(-1).cpu().float().numpy()  # [S, H, W]
    pred_pose  = pred_pose[0].cpu().float().numpy()               # [S, 9]

    # ── 計算各項指標 ──
    depth_m  = depth_metrics(pred_depth, teacher_depth, point_mask)
    camera_m = camera_metrics(pred_pose, teacher_pose, image_hw)

    # ── 計算深度信心分數的一致性（Student conf vs Teacher conf）──
    conf_mse = float("nan")
    if "depth_conf" in outputs:
        pred_conf = outputs["depth_conf"][0].cpu().float().numpy()  # [S, H, W]
        valid = point_mask.astype(bool) & (teacher_conf > 0)
        if valid.sum() > 100:
            conf_mse = float(np.mean((pred_conf[valid] - teacher_conf[valid]) ** 2))

    return {
        "scene_name":    scene_dir.name,
        "inference_ms":  inference_ms,
        **depth_m,
        **camera_m,
        "conf_mse":      conf_mse,
    }


# ──────────────────────────────────────────────
# 模型效率分析
# ──────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """計算模型的可訓練參數總數。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_efficiency_report(student: VGGTStudent) -> dict:
    """
    比較 Student 和 Teacher 的規格差異。

    Teacher 的參數量是預先知道的（約 1B）。
    這裡精確計算 Student 的參數量並計算壓縮比。
    """
    student_params = count_parameters(student)

    # Teacher 的已知參數量（VGGT-1B）
    # 可從 HuggingFace 下載後用同樣方式計算，這裡用近似值
    teacher_params_approx = 1_000_000_000  # 1B

    compression_ratio = teacher_params_approx / student_params

    return {
        "student_params":     student_params,
        "student_params_M":   student_params / 1e6,        # 以百萬為單位
        "teacher_params_approx_B": teacher_params_approx / 1e9,
        "compression_ratio":  compression_ratio,
    }


# ──────────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────────

def mean_metrics(results: list[dict]) -> dict:
    """計算多個場景結果的平均值，跳過 NaN。"""
    if not results:
        return {}

    keys = [k for k in results[0] if k != "scene_name" and isinstance(results[0][k], float)]
    averaged = {}
    for k in keys:
        vals = [r[k] for r in results if not np.isnan(r[k])]
        averaged[k] = float(np.mean(vals)) if vals else float("nan")
    return averaged


def main():
    parser = argparse.ArgumentParser(
        description="評估 Student 模型 vs Teacher soft labels 的差距"
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True,
        help="Student 模型的 checkpoint 路徑（.pth 檔）"
    )
    parser.add_argument(
        "--data_dir", type=str, default="teacher_dataset",
        help="teacher_dataset 的根目錄路徑（預設：teacher_dataset）"
    )
    parser.add_argument(
        "--num_scenes", type=int, default=-1,
        help="評估幾個場景（-1 = 全部）"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="使用的裝置（cuda / cpu）"
    )
    parser.add_argument(
        "--out_json", type=str, default="eval_results.json",
        help="評估結果輸出路徑（預設：eval_results.json）"
    )
    args = parser.parse_args()

    device = args.device
    dtype  = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16 if device == "cuda"
        else torch.float32
    )

    # ── 載入 Student ──
    print(f"\n載入 Student 模型：{args.ckpt_path}")
    student = load_student(args.ckpt_path, device)

    # ── 效率報告 ──
    efficiency = model_efficiency_report(student)
    print(f"\n{'='*55}")
    print(f"  模型效率分析")
    print(f"{'='*55}")
    print(f"  Student 參數量：{efficiency['student_params_M']:.1f}M")
    print(f"  Teacher 參數量：~{efficiency['teacher_params_approx_B']:.0f}B")
    print(f"  壓縮比：{efficiency['compression_ratio']:.1f}x")
    print(f"{'='*55}\n")

    # ── 掃描場景 ──
    data_dir = Path(args.data_dir)
    scene_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and (d / "meta.json").exists()
    )

    if args.num_scenes > 0:
        scene_dirs = scene_dirs[:args.num_scenes]

    if not scene_dirs:
        print(f"在 {data_dir} 找不到任何場景。請先執行 export_teacher_labels.py。")
        return

    print(f"共找到 {len(scene_dirs)} 個場景，開始評估...\n")

    # ── 逐場景評估 ──
    all_results = []

    for i, scene_dir in enumerate(scene_dirs):
        print(f"[{i+1}/{len(scene_dirs)}] {scene_dir.name}", end="  ")
        try:
            result = evaluate_scene(student, scene_dir, device, dtype)
            all_results.append(result)

            print(
                f"abs_rel={result['abs_rel']:.4f}  "
                f"rra15={result['rra_15']:.3f}  "
                f"δ<1.25={result['delta_1']:.3f}  "
                f"time={result['inference_ms']:.1f}ms"
            )
        except Exception as e:
            print(f"[失敗] {e}")

    # ── 平均指標 ──
    avg = mean_metrics(all_results)

    print(f"\n{'='*55}")
    print(f"  平均評估結果（共 {len(all_results)} 個場景）")
    print(f"{'='*55}")
    print(f"  【深度估計（vs Teacher soft label）】")
    print(f"    Abs Rel  ↓  {avg.get('abs_rel',float('nan')):.4f}  （越小越好）")
    print(f"    RMSE     ↓  {avg.get('rmse',float('nan')):.4f}  （越小越好）")
    print(f"    δ<1.25   ↑  {avg.get('delta_1',float('nan')):.4f}  （越大越好）")
    print(f"    δ<1.25²  ↑  {avg.get('delta_2',float('nan')):.4f}  （越大越好）")
    print(f"  【相機姿態（vs Teacher soft label）】")
    print(f"    RRA@15°  ↑  {avg.get('rra_15',float('nan')):.4f}  （越大越好）")
    print(f"    rot err  ↓  {avg.get('mean_rot_err',float('nan')):.4f}°（越小越好）")
    print(f"    focal err↓  {avg.get('focal_rel_err',float('nan')):.4f}  （越小越好）")
    print(f"  【效率】")
    print(f"    推理時間 ↓  {avg.get('inference_ms',float('nan')):.1f}ms")
    print(f"    壓縮比      {efficiency['compression_ratio']:.1f}x")
    print(f"{'='*55}\n")

    # ── 儲存結果 ──
    output = {
        "efficiency": efficiency,
        "average":    avg,
        "per_scene":  all_results,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"評估結果已儲存至：{args.out_json}\n")


if __name__ == "__main__":
    main()
