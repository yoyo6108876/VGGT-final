# 模型權重（Git LFS）

這個資料夾放論文用到的三個模型。檔案都走 Git LFS，`git clone` 時（已安裝 git-lfs）會自動下載，總共約 5.8 GB。

| 檔案 | 論文中的名稱 | 參數 | 大小（bytes） | 來源（訓練時的路徑） |
|---|---|---|---|---|
| `vggt_teacher_init_842M.pt.part0` + `.part1` | Teacher-initialized（842M） | 842.2M | 3,368,981,430（合併後） | `runs/distill_b1/best.pt` |
| `vggt_sliced_464M_agg256_ch512.pt` | Sliced, agg256+trunk512（464M），論文主模型 | 464.4M（state dict 元素數；可訓練參數 464.16M） | 1,908,372,429 | `runs/sliced/recover_agg256_ch512/best.pt` |
| `vggt_sliced_464M_int8.pt` | Sliced + INT8 weight-only（537 MB） | 464M | 537,215,008 | `runs/sliced/quantized_sliced_int8.pt` |

SHA256 見 [`SHA256SUMS.txt`](SHA256SUMS.txt)。這三個檔案都是訓練時存下的原檔，沒有重新存過。

## 只要程式、不要權重

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/yoyo6108876/VGGT-final.git
cd VGGT-final
git lfs pull --include "weights/vggt_sliced_464M_agg256_ch512.pt"   # 之後只抓需要的那一個
```

## 842M 模型：先合併兩段

GitHub LFS 單檔上限是 2 GB，所以 842M 的檔案（3.37 GB）用 `split` 切成兩段，合併後與原檔逐位元相同：

```bash
# Linux / macOS / Git Bash
cat weights/vggt_teacher_init_842M.pt.part0 weights/vggt_teacher_init_842M.pt.part1 > weights/vggt_teacher_init_842M.pt
sha256sum weights/vggt_teacher_init_842M.pt     # 應為 df35c85a…a05（見 SHA256SUMS.txt）
```

```bat
:: Windows（cmd）
copy /b weights\vggt_teacher_init_842M.pt.part0 + weights\vggt_teacher_init_842M.pt.part1 weights\vggt_teacher_init_842M.pt
certutil -hashfile weights\vggt_teacher_init_842M.pt SHA256
```

## 載入方式

先在 repo 根目錄 `pip install -e .`（安裝本 repo 的 `vggt` 套件），再到 `distill2/` 底下執行：

```python
import torch
from eval import load_student                 # distill2/eval.py
from slicegpt_slice import load_sliced        # distill2/slicegpt_slice.py

# 842M teacher-initialized 學生（一般結構，load_student 依存檔的 meta 自動建出架構）
m842 = load_student("../weights/vggt_teacher_init_842M.pt", "cuda")

# 464M 切片模型（必須用 load_sliced：先依存檔裡的旋轉基底 Q_agg / Q_pe / Q_ch 與 k 重建切片結構，再灌權重）
m464 = load_sliced("../weights/vggt_sliced_464M_agg256_ch512.pt", "cuda")

images = torch.rand(1, 4, 3, 518, 518, device="cuda")   # [B, S, 3, H, W]，值域 0~1
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = m464(images)        # out["pose_enc"]（9 維：平移 3、四元數 4 [scalar-last]、視角 2）、out["depth"]、out["depth_conf"] …
```

INT8 檔是 `torch.save({"model": state_dict, "quant": ...})` 存的 torchao weight-only 量化結果，載入時需要同樣的切片結構與 torchao（論文用 torchao 0.17、torch 2.10.0+cu128）：

```python
from torchao.quantization import quantize_, Int8WeightOnlyConfig

mq = load_sliced("../weights/vggt_sliced_464M_agg256_ch512.pt", "cuda")   # 只用來建立結構
quantize_(mq, Int8WeightOnlyConfig())
ck = torch.load("../weights/vggt_sliced_464M_int8.pt", map_location="cuda", weights_only=False)
mq.load_state_dict(ck["model"])
```

weight-only INT8 的量化是對權重做 absmax、per-output-channel，結果是確定的；直接對 464M 模型跑 `distill2/quantize.py` 也會得到同樣的量化權重。

## 這些模型能做、不能做什麼

- 學生模型保留 VGGT 的相機姿態與深度路徑；track head 已移除，point head 有建立但沒有訓練、也沒有評估。
- 精度（論文 Table 2，對照 CO3D ground truth，65 個測試場景 × 10 幀）：842M 的 RRA@15 0.964、AUC@30 0.876；464M 的 RRA@15 0.883、AUC@30 0.767。這是 INT8 之前的數字。INT8 只對照教師模型檢查過旋轉誤差（5.20° → 5.21°）。
- 詳細結果與設定見 [`docs/RESULTS.md`](../docs/RESULTS.md)。

## 授權

這些權重是 Meta VGGT-1B 的衍生模型（前端直接繼承自 VGGT 的 DINOv2 ViT-L/14），依 [VGGT License](../LICENSE.txt) 散布；訓練資料是 VGGT 對 CO3D 推論得到的 soft label（CO3D 為 CC BY-NC 4.0）。僅供研究與非商業用途。
