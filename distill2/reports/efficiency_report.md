# 效率 / 壓縮量測

- 環境：NVIDIA GeForce RTX 5080，dtype=torch.bfloat16
- 設定：S=4 幀、解析度 518×518、latency 取 20 次平均

- FLOPs 以 fvcore 估算（MAC，注意力等自訂算子可能低估），僅供相對比較。

| Model | Params(M) | Trainable(M) | GMACs | VRAM(GB) | Latency(ms) | Throughput(fps) | 壓縮× |
|---|---|---|---|---|---|---|---|
| Teacher (VGGT-1B) | 1256.5 | 1256.5 | N/A | 9.22 | 384.6 | 10.4 | 1.0x |
| Student-conv (512, 12層) | 146.4 | 146.4 | 738.5 | 1.76 | 123.8 | 32.3 | 8.6x |
| Student-DINOv2-S (384, 12層) | 110.7 | 110.7 | 668.8 | 1.52 | 109.9 | 36.4 | 11.3x |
| Student-PathB (1024, 12層, teacher-init) | 842.2 | 842.2 | 3666.1 | 5.86 | 233.9 | 17.1 | 1.5x |
