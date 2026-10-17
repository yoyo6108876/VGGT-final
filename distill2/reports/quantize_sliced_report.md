# 量化壓縮報告

- 基準 checkpoint：runs\sliced\recover_agg256_ch512\best.pt
- 量化方法：torchao int8 weight-only (GPU)
- 設定：S=4、518×518、rot_err 取 8 場景平均

| 模型 | 大小 | rot_err | abs_rel | VRAM | latency |
|---|---|---|---|---|---|
| 基準 (未量化) | 1858MB | 5.20° | 0.193 | 3.89GB | 223.9ms |
| torchao int8 weight-only (GPU) | 537MB | 5.21° | 0.193 | 3.68GB | 248.8ms |

模型大小壓縮：**3.46×**（1858MB → 537MB），rot_err 5.20° → 5.21°。
