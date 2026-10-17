# Derived from VGGT (Copyright (c) Meta Platforms, Inc. and affiliates), distributed under the VGGT License
# (see LICENSE.txt). Student models for "Compressing a 3D Foundation Model: Teacher-Initialized Distillation
# and Global Projection Slicing for Deployable VGGT" (ICS 2026). The specification table in the docstring
# below is from an early stage; the paper's students are built by distill2/build_student.py.

"""
vggt/models/vggt_student.py

【用途】
  蒸餾流程的第三步：定義比 Teacher 更小的 Student 模型。
  Student 的輸入輸出格式與 Teacher (VGGT) 完全相同，
  因此可以直接替換到相同的 Trainer 和 Loss 流程中。

【名詞說明】
  - embed_dim (嵌入維度)：每個 token 的向量長度。
      Teacher = 1024；Student-conv = 512；Student-dinov2s = 384（被 ViT-S 鎖定）
  - depth (層數)：Transformer 的 attention block 數量。Teacher = 24，Student = 12
  - num_heads (注意力頭數)：Teacher = 16；Student-conv = 8；Student-dinov2s = 6
  - patch_embed：把影像切成 patch 的方式，支援兩種：
      "conv"：單純卷積，可自由設定 embed_dim
      "dinov2_vits14_reg"（或 vitb14/vitl14）：載入 DINOv2 預訓練權重
  - pretrained_patch_embed：是否載入 DINOv2 官方預訓練權重（僅 dinov2_* 有效）
      訓練從頭開始時 True；eval 載入 ckpt 時應傳 False（避免重複下載）
  - intermediate_layer_idx：DPT head 從 Aggregator 的哪幾層抽取特徵。
      Teacher depth=24 用 [4, 11, 17, 23]；Student depth=12 比例縮放後用 [2, 5, 8, 11]。

【Teacher vs Student 規格對比】
  ┌─────────────┬───────────────┬───────────────┬───────────────────┐
  │             │   Teacher     │ Student-conv  │ Student-dinov2s   │
  ├─────────────┼───────────────┼───────────────┼───────────────────┤
  │ embed_dim   │   1024        │     512       │       384         │
  │ depth       │    24         │      12       │        12         │
  │ num_heads   │    16         │       8       │         6         │
  │ patch_embed │ dinov2_vitl14 │     conv      │ dinov2_vits14_reg │
  │ 參數量      │   ~1B         │    ~121M      │      ~96M         │
  └─────────────┴───────────────┴───────────────┴───────────────────┘
"""

import torch
import torch.nn as nn

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead

# Teacher aggregator 的單一 embed_dim（1024）
# 實際 token 維度是 2×1024=2048（frame+global 拼接），投影目標也用 2048
TEACHER_EMBED_DIM = 1024


class VGGTStudent(nn.Module):
    """
    VGGT 的精簡版 Student 模型，用於知識蒸餾訓練。

    設計原則：
    1. forward() 的輸入輸出格式與 VGGT 完全相同，方便替換。
    2. 新增 feature_projector，讓 Loss 計算時可以做 Feature-level KD。
    3. 比 Teacher 小約 6~7 倍，在單張消費級 GPU 也可訓練。

    Args:
        img_size (int): 輸入影像邊長，預設 518（需是 patch_size 的倍數）
        patch_size (int): 每個 patch 的邊長，預設 14
        embed_dim (int): token 嵌入維度。conv 版可自由設定（預設 512）；
                          dinov2 版需匹配對應 ViT 維度（vits14=384, vitb14=768, vitl14=1024）
        agg_depth (int): Aggregator 的 attention block 層數，預設 12
        num_heads (int): 每個 attention block 的頭數。預設 8（搭配 embed_dim=512）；
                          dinov2-s 版應設 6（embed_dim=384 / 64）
        patch_embed (str): "conv" 或 "dinov2_vits14_reg" / "dinov2_vitb14_reg" / "dinov2_vitl14_reg"
        pretrained_patch_embed (bool): dinov2_* 時是否下載官方預訓練權重。
                                        eval 載入 ckpt 時應設 False，避免下載又被 ckpt 覆蓋
        enable_camera (bool): 是否啟用相機姿態預測 head
        enable_depth (bool): 是否啟用深度估計 head
        enable_point (bool): 是否啟用 3D 點圖 head
        enable_track (bool): 是否啟用點追蹤 head
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 512,
        agg_depth: int = 12,
        num_heads: int = 8,
        patch_embed: str = "conv",    # "conv" 或 "dinov2_vits14_reg" / "dinov2_vitb14_reg" 等
        pretrained_patch_embed: bool = True,  # eval 載入 ckpt 時可設 False 跳過 DINOv2 下載
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_point: bool = True,
        enable_track: bool = False,   # 預設關閉，追蹤對蒸餾較次要
    ):
        super().__init__()

        # 若使用 DINOv2 系列 patch_embed，embed_dim 必須匹配對應 ViT 的固定維度
        # vits14: 384, vitb14: 768, vitl14: 1024
        DINOV2_DIMS = {
            "dinov2_vits14_reg": 384,
            "dinov2_vitb14_reg": 768,
            "dinov2_vitl14_reg": 1024,
        }
        if patch_embed in DINOV2_DIMS:
            expected_dim = DINOV2_DIMS[patch_embed]
            if embed_dim != expected_dim:
                raise ValueError(
                    f"patch_embed={patch_embed} 要求 embed_dim={expected_dim}, "
                    f"但收到 embed_dim={embed_dim}。請在 yaml 裡同步修改。"
                )

        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # ── Aggregator（主幹 Transformer）──
        # patch_embed="conv"：輕量卷積（可自由設定 embed_dim）
        # patch_embed="dinov2_xxx"：使用預訓練 DINOv2，引入 142M 張影像的視覺先驗
        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=agg_depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            patch_embed=patch_embed,
            aa_order=["frame", "global"],
            aa_block_size=1,
            qk_norm=True,
            rope_freq=100,
            init_values=0.01,
            pretrained_patch_embed=pretrained_patch_embed,
        )

        # Aggregator 輸出後會把 frame tokens 和 global tokens 拼接（cat）：
        # 所以 head 的輸入維度是 2 × embed_dim
        head_dim_in = 2 * embed_dim

        # ── DPT Head 的中間層索引 ──
        # DPTHead 從 Aggregator 的特定層抽取多尺度特徵（類似 FPN 的概念）
        # Teacher depth=24：用 [4, 11, 17, 23]（第 5, 12, 18, 24 層）
        # Student depth=12：按比例縮放 → [2, 5, 8, 11]（第 3, 6, 9, 12 層）
        intermediate_layer_idx = self._compute_intermediate_idx(agg_depth)

        # ── Camera Head（預測相機外參 + 內參）──
        # trunk_depth=4 對齊 teacher，提供足夠 rotation 容量。
        self.camera_head = (
            CameraHead(
                dim_in=head_dim_in,
                trunk_depth=4,
                num_heads=num_heads,
            )
            if enable_camera else None
        )

        # ── Depth Head（預測深度圖 + 信心分數）──
        # output_dim=2：深度值 + 信心分數各 1 個 channel
        self.depth_head = (
            DPTHead(
                dim_in=head_dim_in,
                patch_size=patch_size,
                output_dim=2,
                activation="exp",
                conf_activation="expp1",
                features=128,          # Teacher 預設 256，Student 縮小
                out_channels=[128, 256, 512, 512],  # Teacher: [256,512,1024,1024]
                intermediate_layer_idx=intermediate_layer_idx,
            )
            if enable_depth else None
        )

        # ── Point Head（預測 3D 點圖 + 信心分數）──
        # output_dim=4：xyz 座標 3 個 + 信心分數 1 個 channel
        self.point_head = (
            DPTHead(
                dim_in=head_dim_in,
                patch_size=patch_size,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
                features=128,
                out_channels=[128, 256, 512, 512],
                intermediate_layer_idx=intermediate_layer_idx,
            )
            if enable_point else None
        )

        # ── Track Head（點追蹤，選用）──
        self.track_head = (
            TrackHead(dim_in=head_dim_in, patch_size=patch_size)
            if enable_track else None
        )

        # Feature Projector 暫時移除，等 Feature-level KD 實作時再加回來
        # self.feature_projector = nn.Linear(head_dim_in, TEACHER_EMBED_DIM * 2, bias=False)

    # ──────────────────────────────────────────────
    # 輔助方法
    # ──────────────────────────────────────────────

    @staticmethod
    def _compute_intermediate_idx(depth: int) -> list[int]:
        """
        根據 Aggregator 的層數，計算 DPT Head 要抽取哪幾層特徵。

        Teacher (depth=24) 的設計是取第 1/6、1/2、3/4、末層：
            [4, 11, 17, 23]  →  比例約 0.20, 0.46, 0.71, 0.96

        Student 按相同比例縮放，並確保最後一個索引是 depth-1：
        例如 depth=12：[2, 5, 8, 11]
        """
        ratios = [4/23, 11/23, 17/23, 1.0]
        idx = []
        for r in ratios:
            i = min(round(r * (depth - 1)), depth - 1)
            idx.append(i)
        # 去重並排序（邊界情況保護）
        idx = sorted(set(idx))
        # 若去重後不足 4 個（depth 極小時），補回末層
        while len(idx) < 4:
            idx.append(depth - 1)
        return idx[-4:]  # 只取最後 4 個，確保數量正確

    # ──────────────────────────────────────────────
    # Forward（與 VGGT 完全相同的介面）
    # ──────────────────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
    ) -> dict:
        """
        Student 模型的前向傳播。輸入輸出格式與 VGGT.forward() 完全相同。

        Args:
            images: 輸入影像，形狀 [S, 3, H, W] 或 [B, S, 3, H, W]，值域 [0, 1]
                    B=batch size，S=序列長度（幀數），3=RGB
            query_points: 追蹤用的查詢點（選用），形狀 [N, 2] 或 [B, N, 2]

        Returns:
            dict，包含以下 key（視啟用的 head 而定）：
                pose_enc        [B, S, 9]       壓縮相機姿態（最後一次迭代）
                pose_enc_list   list of [B, S, 9]，camera head 多 stage 結果（給 loss 用）
                depth           [B, S, H, W, 1] 深度圖
                depth_conf      [B, S, H, W]    深度信心分數
                world_points    [B, S, H, W, 3] 3D 世界座標
                world_points_conf [B, S, H, W]  點圖信心分數
                images          [B, S, 3, H, W] 原始輸入影像（僅推理時）

            註：Feature-level KD（student_features / student_features_proj）
            目前已停用，等 feature_projector 重新啟用時再加回。
        """
        # 若無 batch 維度，自動補上
        if images.dim() == 4:
            images = images.unsqueeze(0)

        if query_points is not None and query_points.dim() == 2:
            query_points = query_points.unsqueeze(0)

        # ── Aggregator 前向傳播 ──
        # aggregated_tokens_list：每個 attention block 的輸出 token 列表
        # patch_start_idx：token 序列中，patch token 從第幾個開始
        #   （前幾個是 camera token + register token）
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        # ── Head 前向傳播（關閉混合精度，確保數值穩定）──
        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images,
                patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        # ── 推理模式：儲存原始影像供視覺化 ──
        if not self.training:
            predictions["images"] = images

        # Feature-level KD 暫時停用（feature_projector 移除中）
        # if self.training:
        #     last_tokens = aggregated_tokens_list[-1]
        #     predictions["student_features"] = last_tokens

        return predictions
