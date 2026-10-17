# distill2 — VGGT 知識蒸餾（乾淨重建版）

把 VGGT-1B（teacher）蒸餾成輕量 student 的完整、自包含管線。
核心貢獻：**診斷出相機的 camera token cross-frame collapse，並證明 structured teacher-init 可解**。

> 這是作者在 `distill2/` 資料夾裡的重現筆記（內部用語如「Path B」＝論文的 teacher-initialized student、「P0」＝內部的實驗編號）。
> 公開版的完整說明請看根目錄的 [`README.md`](../README.md) 與 [`docs/`](../docs/)；改動歷程見 [`CHANGES.md`](CHANGES.md) 與 [`docs/HISTORY.md`](../docs/HISTORY.md)。

## 儲存庫結構

```
distill2/
├── *.py                  程式碼（build_student / verify_teacher_copy / data / loss(2) / train(2) /
│                         eval / eval_gt / measure_efficiency / quantize / run_ablation / visualize /
│                         slicegpt_* / train_sliced …）
├── reports/              實驗報告（效率 / 量化 / 消融 / 相對位姿；md·csv·json）
├── viz/                  視覺化圖（深度 / 點雲 / 軌跡 / 直方圖 / success-failure）
├── README.md             本檔（重現說明）
└── CHANGES.md            改動說明（歷史紀錄）
```
> 大型檔（`*.pt` 權重、`runs/`、`abl/`、`*.npy`）已由 `.gitignore` 排除，不進版控；
> 例外：論文用到的三個模型放在 repo 根目錄的 `weights/`（Git LFS，見 [`../weights/README.md`](../weights/README.md)）。

---

## 一、核心發現（一句話）

小 student 從零蒸餾時，深度可學、但**相機旋轉塌縮**（camera token 跨幀 cosine 0.998、rot_err ≈82°≈隨機）。
蒸餾（輸出 / 特徵 / 結構 KD）皆不足以解決；**唯有複製 teacher 權重（patch embed + aggregator + camera head 三者缺一不可，且凍結前端）**才能打破塌縮 → rot_err 4.9°、RRA@15 0.97（20 epochs 的 `runs\distill_b1_full`、65 場景；論文 checkpoint 見「四」）。

---

## 二、環境與依賴

- Python 3.10 以上、PyTorch（CUDA；論文環境為 2.10.0+cu128）、`numpy`、`opencv-python`、`matplotlib`、`fvcore`。
- **`vggt` 套件**：就是本 repo 根目錄的 `vggt/`（VGGT 原始碼加上本專案的 `VGGTStudent`），在 repo 根目錄執行 `pip install -e .` 安裝。論文所有結果都是用這一版（作者私有 fork 的 commit `3c14206`，另加 4 個空的 `__init__.py`）。
- 選用：`torchao`（GPU int8 量化）、`pypdf`（讀 PDF）。
- 硬體（實測）：RTX 5080 16GB。
- ⚠ 不要另外把其他 `vggt` 副本放進 `sys.path` 前面，否則會載到錯版本的 `vggt`（舊版 camera head 只有 2 層，見 CHANGES）。

---

## 三、資料

teacher soft labels（用 VGGT-1B 對 CO3D 推理產生），目錄：
```
<DATA_ROOT>\{train,test}\<category__seqid>\
   images/  depth.npy  depth_conf.npy  extrinsics.npy  intrinsics.npy
   pose_enc.npy  point_mask.npy  world_points.npy  meta.json
```
train 427 / test 65 場景。由 `../scripts/export_co3d_labels.py` 產生。

---

## 四、主線流程（重現 Path B）

```powershell
cd distill2

# 1) 從 teacher 切出 student（複製全部元件）→ student_init.pt
python build_student.py --out student_init.pt --copy all

# 2) 訓練（凍結 patch embed；定期 eval 存 best）——train.py 預設 40 epochs × 500 steps、img_per_seq 4
python train.py --student_init student_init.pt `
    --data_root <DATA_ROOT> `
    --out_dir runs\distill_b1

# 3) 評估（rot_err / RRA / abs_rel）
python eval.py --ckpt runs\distill_b1\best.pt `
    --data_root <DATA_ROOT>\test --num_scenes 65
```

VRAM 不足：把 `--img_per_seq` 降到 2、或加 `--throttle 0.05`（降 GPU 佔用避免黑頻）。

**實際存在的兩個 Path B 模型**（同一個 `student_init.pt`、同一份 train.py，只差訓練長度）：

| 目錄 | 訓練設定 | best | 用在哪裡 |
|---|---|---|---|
| `runs\distill_b1` | 上面的指令，train.py 預設（40 epochs；當時指令沒有另外存檔，依 checkpoint 的 `epoch` 推定） | epoch 37（`last.pt` = epoch 39） | **切片、464M 模型、CO3D GT 評估（Path B 0.964 / 0.876）、量化、效率表**的基底 |
| `runs\distill_b1_full` | `--epochs 20 --eval_every 2 --img_per_seq 2 --throttle 0.1` | epoch 19 | 「七」裡 rotation 82.7° → **4.9°**（65 場景）這一行 |

論文第 7 頁寫 "the main run is 20 epochs of 500 steps"，但切片與 GT 表格用的其實是 40 epochs 的 `runs\distill_b1`。

---

## 四之二、權重複製（teacher-init）：程式、版本與驗證

Path B 的權重複製全部在 **`build_student.py`**。teacher（`facebook/VGGT-1B`，aggregator 24 層）→
student（寬度同為 1024、aggregator 12 層），`--copy all` 時的對應：

| student | ← teacher | 張量數 | 備註 |
|---|---|---|---|
| `aggregator.patch_embed`（DINOv2 ViT-L/14-reg） | 同名，整個複製 | 344 | 訓練時凍結 |
| `aggregator.camera_token` / `register_token` | 同名 | 1 + 1 | |
| `aggregator.frame_blocks[i]`（i = 0…11） | `frame_blocks[2i]` | 216 | 每隔一層取一層 |
| `aggregator.global_blocks[i]` | `global_blocks[2i]` | 216 | 同上 |
| `camera_head` | 同名，整個複製 | 69 | pose 能力的關鍵 |
| `depth_head` / `point_head` | **不複製**（隨機初始化，由蒸餾學） | 62 + 62 | |

其他選項（消融用）：`--copy none`、或逗號分隔 `patch_embed,aggregator,camera_head` 任意組合。

**版本說明**：`student_init.pt`（建立 2026-06-07、最後寫入 06-08）早於本 repo 第一個 commit `48f0412`（06-09），
所以產生它的那份程式沒有被精確版控；它的 `meta` 格式與 `48f0412` 版完全一致（沒有 `copy` 欄位）。
目前版本（`a2e2ead`，2026-07-07）為了消融加了 `--copy` 參數。逐行比對兩版：`--copy all` 時複製的元件、token、
層對應（`i ← 2i`）、`load_state_dict(strict=False)` 都相同。其餘差異不影響被複製的張量：teacher/student 建構順序
對調（隨機初始化 head 時的 RNG 序不同）、移除層數整除 assert、`embed_dim`/`num_heads` 改為寫死 1024/16、
存檔 `meta` 多一個 `copy` 欄位。depth/point head 是不固定 seed 的隨機初始化，兩版都不會逐位元重現。
不論是哪一版產生的，複製結果以下面的逐位元驗證為準。

**驗證**（`verify_teacher_copy.py`；只讀 checkpoint 與 teacher 的 safetensors，不需 GPU，記憶體需求低）：

```powershell
python verify_teacher_copy.py --ckpt student_init.pt runs\distill_b1\best.pt
```

2026-10-17 實跑結果：

| 檔案 | 應複製的 847 個張量 vs teacher | depth / point head（124 個） |
|---|---|---|
| `student_init.pt` | **全部逐位元相同**（0 個不同） | 不同（未複製，符合預期） |
| `runs/distill_b1/best.pt`（訓練後） | `patch_embed` 344 個**仍相同**（證實凍結生效）；其餘 503 個已被訓練更新 | 不同 |

---

## 五、補充實驗（內部編號 P0）

```powershell
# P0-1 嚴謹相對位姿評估（RRA/RTA/AUC；至少 vs teacher，有 CO3D 標註可對 GT）
python eval_gt.py --ckpt runs\distill_b1\best.pt --data_root ...\test --num_scenes 20
python eval_gt.py --ckpt runs\distill_b1\best.pt --data_root ...\test --ref co3d --co3d_root <CO3D_ROOT>

# P0-2 效率 / 壓縮表（params / FLOPs / VRAM / latency / throughput）
python measure_efficiency.py --frames 4 --img_size 518 --runs 20 --pathb_ckpt runs\distill_b1\best.pt
# P0-2 量化（int8，測壓縮不掉精度）
python quantize.py --ckpt runs\distill_b1\best.pt --data_root ...\test --num_scenes 8

# P0-4 消融（teacher-init 各元件的貢獻；建議過夜）
python run_ablation.py --stage all --epochs 12 --img_per_seq 2 --throttle 0.05
python run_ablation.py --stage collect      # 只收表

# P0-5 視覺化（深度圖 / point cloud / camera trajectory / rotation histogram / success-failure）
python visualize.py --ckpt runs\distill_b1\best.pt --data_root ...\test `
    --collapsed_ckpt runs\abl\random\best.pt --out_dir viz
```

（探索）二次蒸餾——把 Path B 當 teacher、蒸餾更小 student（目前仍塌縮，屬負面結果 / 未來工作）：
```powershell
python train2.py --teacher_ckpt runs\distill_b1\best.pt --data_root ...\ `
    --out_dir runs\distill2_s --img_per_seq 2 --feature_weight 3 --struct_weight 4
```

---

## 五之二、層剪枝與 SliceGPT 寬度切片

```powershell
# 層剪枝（保持寬度 1024、只砍層數；build → train d8/d6/d4 → 收表）
python run_layer_pruning.py --stage all --depths 8,6,4 --epochs 12 --img_per_seq 2 --throttle 0.05
python run_layer_pruning.py --stage collect --depths 8,6,4

# SliceGPT 光譜探針（hook 投影、不動權重；判斷寬度可否事後切片）
python slicegpt_probe.py --ckpt runs\distill_b1\best.pt --data_root ...\ --ks 1024,896,768,640,512,384,256

# SliceGPT 真切片（權重手術）：先跑等價性測試（CPU），再校準+掃 k+評估
python slicegpt_slice.py --check
python slicegpt_slice.py --run --ks 512,384,256 --bench --save_k 512

# 切片模型恢復蒸餾（k<512 崩掉時救回；--k_pe/--k_ch 支援三流組合）
python train_sliced.py --k 384 --q_ckpt runs\sliced\sliced_k512.pt --data_root ...\ --img_per_seq 2 --throttle 0.05
python train_sliced.py --k 256 --k_ch 1024 --q_ckpt "runs\sliced\sliced_agg256+pe512+ch1024.pt" --data_root ...\ --img_per_seq 2 --throttle 0.05
```

結論速覽：**深度阈值 8 層**（d8 可用 1.7×、d6 以下崩）；**寬度可事後切片**——
k=512 零重訓 5.62°；**k=384/256 + 6 epochs 恢復蒸餾 → 4.18° / 4.95°（殘差流 1/4 寬仍達 Path B 品質）**。
先前「變窄訓練全塌」是優化問題而非容量問題（探針 k=256 仍 4.9°）。
三流擴展：**camera trunk 可切**（agg256+ch1024 → **5.36° @ 516M，vs teacher 2.43×**）；
**DINOv2 PE 流不可切**（k_pe=512 連 --train_pe 都塌——前端損傷＝已知塌縮域，重要負面結果）。
報告見 `reports/layer_pruning_report.md`、`reports/efficiency_prune.md`、`reports/slicegpt_probe.md`、
`reports/slicegpt_slice.md`、`reports/slicegpt_slice_combo.md`。

---

## 五之三、論文主模型（464M）：權重、產生方式、載入

**權重**：repo 根目錄的 `weights/vggt_sliced_464M_agg256_ch512.pt`（Git LFS）

| 項目 | 內容 |
|---|---|
| 來源 | `runs/sliced/recover_agg256_ch512/best.pt` 原檔複製（best = 第 6 個 epoch，`epoch=5`） |
| 大小 / SHA256 | 1,908,372,429 bytes / `FE6E2CF735DCCB993560C98D21B2E3DC1357F2EE0186986CB6BFE7C7E8F1B352` |
| 結構 | Path B（`runs\distill_b1`，40 epochs）→ SliceGPT 切 aggregator 流 k=256、camera trunk 流 k=512；DINOv2 前端未切 |
| 參數 | **464.16M 參數**；state_dict 464.43M（論文寫的 464.4M 是這個數），多出的 0.28M 是 buffer（`deslice_Q` 0.262M + SlicedNorm `v` 0.017M）；teacher 1256.5M → **2.71×** |
| 精度（vs teacher） | rot_err **5.20°**；逐幀旋轉誤差 <15° 的比例 0.969（存檔內 `eval`：quick_eval、8 個 test 場景、相對第 0 幀，**不是**下面 GT 的成對 RRA） |
| 精度（CO3D GT） | 65 場景×10 幀、成對相對位姿：RRA@15 **0.883**、AUC@30 **0.767**（`reports/eval_gt_co3d10_ch512.md`，`eval_gt.py --ckpt runs\sliced\recover_agg256_ch512\best.pt --ref co3d --num_scenes 65 --frames 10`） |
| 存檔內容 | `model`、`meta`、旋轉基底 `Q_agg`（1024²）/ `Q_pe`（1024²）/ `Q_ch`（2048²）與 `k_agg=256` / `k_pe=0` / `k_ch=512` |

**下載**：裝好 git-lfs 後 `git clone` 會自動抓 `weights/` 的三個模型（共約 5.8 GB；只抓主模型約 1.9 GB，見 [`../weights/README.md`](../weights/README.md)）。只要程式不要權重：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/yoyo6108876/VGGT-final.git
```

之後要權重再 `git lfs pull --include "weights/*.pt"`。（LFS 下載量計入 repo 擁有者的每月流量額度。）

**載入**（切片後的結構，**必須用 `load_sliced()`**：它先依存檔裡的 Q/k 重建切片結構再灌權重；
直接 `VGGTStudent` + `load_state_dict` 會對不上）：

```python
import torch
from slicegpt_slice import load_sliced          # 在 distill2/ 下執行

model = load_sliced("../weights/vggt_sliced_464M_agg256_ch512.pt", "cuda")
images = torch.rand(1, 4, 3, 518, 518, device="cuda")   # [B, S, 3, H, W]，值域 0~1
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(images)        # out["pose_enc"]、out["depth"]、out["depth_conf"] …
```

**產生方式**（實際執行過的指令）：

```powershell
# 1) Path B：build_student.py → train.py（預設 40 epochs）→ runs\distill_b1\best.pt（見「四」）

# 2) 在 Path B 上校準三流旋轉基底並存檔（含完整 Q_agg / Q_pe / Q_ch）
#    當時用 --out reports\slicegpt_slice_combo.md；那份報告之後有手動整理過，重跑請改輸出檔名以免蓋掉
python slicegpt_slice.py --run --ks 0,256 --k_pe 512 --k_ch 1024 --bench --save_k 256 `
    --out reports\slicegpt_slice_combo_rerun.md
#    → runs\sliced\sliced_agg256+pe512+ch1024.pt（下一步只取其中的完整 Q，k 重新指定）

# 3) 切 aggregator 256 + camera trunk 512（PE 不切），恢復蒸餾 6 epochs
python train_sliced.py --k 256 --k_ch 512 --q_ckpt "runs\sliced\sliced_agg256+pe512+ch1024.pt" `
    --data_root <DATA_ROOT> `
    --epochs 6 --img_per_seq 2 --eval_every 2 --throttle 0.05
#    → runs\sliced\recover_agg256_ch512\best.pt
```

程式版本：`slicegpt_slice.py` 最後修改於 2026-07-13 18:41，之後沒有再動（同日 20:01 提交為 `c314bcd`）；
步驟 2 的 Q 檔寫於 18:46、步驟 3 的模型寫於 23:34，所以兩步用的都是這份程式（步驟 2 若在 18:41 前就已啟動則無法確認，
但 Q 基底已存在權重檔裡，載入與重切都不必重新校準）。`train_sliced.py` 之後只多了兩個選用參數
（`--gt_pose`、`--resume_sliced`），不加時流程與當時相同。
腳本的預設路徑是 Windows 寫法（例如 `runs\sliced\...`）；在 Linux 上請明確給 `--out_dir` / `--save_dir`。

int8 版（同一模型，537MB、5.21°）在 repo 根目錄的 `weights/vggt_sliced_464M_int8.pt`，載入方式見 [`../weights/README.md`](../weights/README.md)；
也可以從上面的權重重新量化（需 `torchao` 與 GPU；`quantize.py` 偵測到存檔含 `Q_agg` 會自動改用 `load_sliced`；
輸出要放到已存在的資料夾，例如 `--out quantized_sliced_int8_rerun.pt`，因為 quantize.py 不會自動建立資料夾）：

```powershell
python quantize.py --ckpt ..\weights\vggt_sliced_464M_agg256_ch512.pt `
    --data_root <DATA_ROOT>\test --num_scenes 8 `
    --out quantized_sliced_int8_rerun.pt --report reports\quantize_sliced_report_rerun.md
#    原始結果見 reports\quantize_sliced_report.md（不給 --out/--report 會寫到目前目錄的
#    quantized_int8.pt / quantize_report.md，會蓋掉 Path B 的 int8 檔）
```

---

## 六、檔案總覽

| 檔案 | 用途 |
|---|---|
| `build_student.py` | 從 teacher 選擇性複製權重（`--copy all/none/patch_embed,...`）|
| `verify_teacher_copy.py` | 驗證 student 權重與 teacher 逐位元相同（見「四之二」）|
| `data.py` | Dataset：讀 teacher labels、first-frame canonical + scale 正規化 |
| `loss.py` | 主 loss：depth + camera（rotation `1-\|dot\|`、scalar-last、獨立 EMA）|
| `loss2.py` | 二次蒸餾 loss：+ camera-token feature KD + 跨幀結構 KD |
| `train.py` | 主訓練（Path B）|
| `train2.py` | 二次蒸餾（線上 Path B teacher + 小 student）|
| `eval.py` | 基本評估（rot_err / abs_rel）；`load_student` 自動偵測架構 |
| `eval_gt.py` | **P0-1** 相對位姿（RRA/RTA/AUC；vs teacher 或 CO3D GT）|
| `measure_efficiency.py` | **P0-2** 效率 / 壓縮表 |
| `quantize.py` | **P0-2** int8 量化 |
| `run_ablation.py` | **P0-4** 消融 orchestrator |
| `run_layer_pruning.py` | 層剪枝 orchestrator（d8/d6/d4，Path B 配方）|
| `slicegpt_probe.py` | SliceGPT 光譜探針（hook 投影截斷，判斷可行性）|
| `slicegpt_slice.py` | SliceGPT 真切片（LN/LayerScale 摺疊 + 共享基底；`load_sliced` 載入）|
| `train_sliced.py` | 切片模型恢復蒸餾 |
| `visualize.py` | **P0-5** 視覺化（5 張圖）|

---

## 七、關鍵結果

| 項目 | 數字 |
|---|---|
| rotation（vs teacher soft label） | 82.7° → **4.9°**；RRA@15 0.13 → **0.97**（此行來自 20 epochs 的 `runs\distill_b1_full`、65 場景，不是論文用的 checkpoint；論文的 `runs\distill_b1`（40 epochs）quick_eval 為 3.42°（8 場景）/ 2.87°（12 場景），見「四」）|
| 相對位姿（vs 原版 VGGT） | RRA@15 **0.98**、RTA@15 0.98、AUC@30 **0.89** |
| **CO3D ground-truth**（65 場景×10 幀） | teacher RRA@15 1.000 / AUC@30 0.976；**Path B 0.964 / 0.876**；k384 0.905 / 0.792；agg256+ch512 0.883 / 0.767 |
| 實測加速（S=4, RTX 5080） | teacher 384.6ms → Path B 233.9ms（**1.64×**）→ 切片版 201.1ms（**1.91×**） |
| depth | abs_rel 1.04 → **0.18**、δ<1.25 0.40 → 0.70 |
| 參數 | teacher 1256M / Path B 842M（**架構壓縮 1.5×**）|
| int8 量化 | 3369MB → **915MB（3.68×）**，rot_err 零損失 |
| 消融 | 只有「複製全部 + 凍結 PE」= 4.3°；其餘皆 ~75°（**三者缺一不可**）|
| 層剪枝 | **深度阈值 8 層**：d8 = 9.45°/741M/1.7×；d6、d4 崩 |
| SliceGPT | k=384/256 恢復後 4.18°/4.95°；**agg256+trunk512 → 5.20° @ 464.2M 參數（state_dict 464.4M；2.71×）**；PE 流不可切（512/768 皆塌）|
| 切片×int8 | **1858MB → 537MB（3.46×）零損失** → vs teacher fp32 部署大小 **≈9.4×** |

**限制**：(1) 解法脆（前端一微調即塌，且 PE 流不可切）；(2) 壓縮模型對 GT 的 RRA@15 較 Path B 低約 8 個百分點（0.964 → 0.883）；(3) 跨資料集泛化未驗證。

---

## 八、重現性備註

- 隨機性：`train.py`、`data.py` 都沒有固定 seed，也沒做過同設定的重複實驗。現有幾個「複製全部」的 run 設定不同：
  主線 `runs\distill_b1`（40 ep）quick_eval 3.42°、`runs\distill_b1_full`（20 ep）4.44°、消融 full（12 ep）4.32°，僅供參考。
- 產物：checkpoint（`runs/**/best.pt`）、報告（`*.md/*.csv`）、圖（`viz/*.png`）。
- 大檔（`*.pt` / `*.npy` / `runs/`）已由 `.gitignore` 排除；資料集在 repo 外，不進版控。
  例外：`weights/*.pt` 走 Git LFS（`.gitattributes`）。
- 早期內部報告（`reports/*.md`、`measure_vram_curve.py` 註解）寫的「466M」是同一個模型（`recover_agg256_ch512`）的錯誤數字：
  正確為 **464.2M 參數 / 464.4M state_dict**（論文用後者）。
