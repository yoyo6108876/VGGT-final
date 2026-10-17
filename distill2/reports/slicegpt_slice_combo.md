# SliceGPT 三流合切結果（aggregator + DINOv2 PE + camera trunk）

模型：`runs\distill_b1\best.pt`。校準 8 scenes；zero-shot eval 12 test scenes；
恢復蒸餾 eval 8 test scenes。storage = state_dict 總元素數。
三流合切等價性測試（隨機正交 Q、k=滿維）：pose max|Δ|=3.0e-7、depth 2.6e-6（機器精度）。

## 零重訓

| 組合 | rot_err | RRA@15 | abs_rel | delta_1 | storage (M) | latency (ms) |
|---|---|---|---|---|---|---|
| 原模型 | 2.87° | 1.000 | 0.1622 | 0.756 | 842 | 224 |
| agg0+pe512+ch1024 | 72.82° | 0.236 | 0.2447 | 0.599 | 591 | 211 |
| agg256+pe512+ch1024 | 74.75° | 0.167 | 0.2458 | 0.544 | 365 | 187 |

## 恢復蒸餾（train_sliced.py，6–8 epochs）

| 組合 | 恢復條件 | rot_err | RRA@15 | storage (M) | 結論 |
|---|---|---|---|---|---|
| agg256+pe512+ch1024 | 凍結 DINOv2，6ep | 69.4° | 0.177 | 365 | ❌ |
| agg256+pe512+ch1024 | **--train_pe**，8ep | 77.2° | 0.188 | 365 | ❌ |
| agg256+ch1024 | 凍結 DINOv2，6ep | 5.36° | 0.948 | 516 | ✅ |
| **agg256+ch512（trunk 1/4）** | 凍結 DINOv2，6ep | **5.20°** | **0.969** | **466** | ✅ `runs/sliced/recover_agg256_ch512/best.pt` |
| agg256+ch512+pe768 | 凍結 DINOv2，6ep | 48.1° | 0.208 | ~390 | ❌ |

## 結論

- **camera trunk 流（2048）非常可切**：k_ch=512（1/4）比砍半更好（5.20° vs 5.36°，雜訊內）。
  **最終壓縮前沿：842M → 466M（vs teacher 2.70×）@ 5.20°**。
- **DINOv2 PE 流不可切，邊界確認到 k_pe=768**：512 三種恢復條件全塌（73°/69°/77°）、
  768 同預算也只到 48°（trunk/agg 同預算 <6°）——前端特徵一損傷就落回已知塌縮域，
  與「Path B 必須精確複製前端」的教訓一致（重要負面結果）。未試：PE 專用 per-site 基底。
- int8 疊加見 `quantize_sliced_report.md`。
