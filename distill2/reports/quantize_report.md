# 量化壓縮報告

- 基準 checkpoint：runs\distill_b1\best.pt
- 量化方法：torchao int8 weight-only (GPU)
- 設定：S=4、518×518、rot_err 取 8 場景平均

| 模型 | 大小 | rot_err | abs_rel | VRAM | latency |
|---|---|---|---|---|---|
| 基準 (未量化) | 3369MB | 3.42° | 0.177 | 5.85GB | 214.8ms |
| torchao int8 weight-only (GPU) | 915MB | 3.42° | 0.177 | 5.57GB | 237.8ms |

模型大小壓縮：**3.68×**（3369MB → 915MB），rot_err 3.42° → 3.42°。
