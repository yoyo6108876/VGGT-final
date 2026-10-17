# SliceGPT 切片結果（共享基底、head 介面 de-slice、head 未切）

模型：`runs\distill_b1\best.pt`。校準 8 scenes；eval 12 test scenes。
storage = state_dict 總元素數（含 304M 凍結 patch embed 與 de-slice 矩陣）。

| k | 共享基底能量 | rot_err | RRA@15 | abs_rel | delta_1 | storage (M) | latency (ms) |
|---|---|---|---|---|---|---|---|
| 原模型 | 100.00% | 2.87° | 1.000 | 0.1622 | 0.756 | 842 | 223 |
| 512 | 99.52% | 5.62° | 0.958 | 0.1681 | 0.772 | 692 | 208 |
| 384 | 99.12% | 34.60° | 0.382 | 0.1769 | 0.752 | 654 | 203 |
| 256 | 98.30% | 49.74° | 0.264 | 0.1949 | 0.709 | 616 | 199 |

## 恢復蒸餾（train_sliced.py，6 epochs × 400 steps，lr 2e-5，凍結 DINOv2 本體）

零重訓在 k<512 崩掉，但短程恢復蒸餾幾乎完全救回（eval 8 test scenes）：

| k | 零重訓 rot_err | 恢復後 rot_err | RRA@15 | abs_rel | 權重 |
|---|---|---|---|---|---|
| 384 | 34.60° | **4.18°** | 1.000 | 0.181 | `runs/sliced/recover_k384/best.pt` |
| 256 | 49.74° | **4.95°** | 0.990 | 0.188 | `runs/sliced/recover_k256/best.pt` |

→ aggregator 殘差流切到 1/4 寬度（1024→256）仍可達 Path B 品質，
與光譜探針的預測（k=256 資訊保留、4.91°）一致。
**「事後切片 + 恢復蒸餾」正式打通寬度壓縮**（對照：所有「窄模型從頭訓練」策略皆塌）。
載入方式：`slicegpt_slice.load_sliced(path)`（存檔含 Q / k / meta）。
