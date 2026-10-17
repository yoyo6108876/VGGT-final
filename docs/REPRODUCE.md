# 重現指南（單張 GPU）

> **對象**：只有一張 GPU、想從 CO3D 原始資料開始重跑論文每個表格的人。
> **論文**：*Compressing a 3D Foundation Model: Teacher-Initialized Distillation and Global Projection Slicing for Deployable VGGT*。作者 Tzu-Chi Huang、Kuan-You Chen、Chung-Wei Lin、Chow-Sing Lin（國立臺南大學）。已被 ICS 2026 接受，收錄於 Springer CCIS，尚未出版。
> **原則**：每個數字後面都標出處。用算術推出、沒有直接記錄的數字標「推算」。指令如果在紀錄裡找不到原文，而是依 argparse 預設值、run 目錄名稱或報告標頭重建的，標「重建」。論文和實驗紀錄不一致的地方，兩邊都列出來，並註明哪個是論文、哪個是紀錄（第 16 節彙整）。

**記號與慣例**

| 記號 | 意思 |
|---|---|
| `<REPO>` | 本 repo 的根目錄 |
| `<DATA_ROOT>` | teacher soft-label 資料集的根目錄，底下有 `train/` 和 `test/`（第 4 節產生） |
| `<CO3D_ROOT>` | CO3D 原始資料的根目錄，底下是各類別資料夾 |
| （論文 §x.x / Table n / App. X） | 出處是論文本文 |
| （作者開發紀錄） | 只記在作者的私人實驗筆記，或只能從原始機器上的檔案讀到（checkpoint 欄位、檔案大小、檔案時間），不在本 repo 裡。本文只保留核對過的內容 |

- 除非另外註明，指令都在 `<REPO>/distill2/` 下執行。原因有兩個：腳本之間用 `from eval import ...` 互相匯入；輸出的 `runs/`、`reports/`、`abl/` 也都放在 `distill2/` 底下（來源：distill2/README.md 五之三「在 distill2/ 下執行」）。所以從 `distill2/` 指到 repo 裡的權重時，路徑要寫成 `../weights/...`。
- 原始實驗是在 Windows 的 PowerShell 上跑的（來源：distill2/README.md 四、五之三的 PowerShell 指令和反斜線路徑；作者開發紀錄）。本文的指令用 bash 寫法和正斜線路徑。在 PowerShell 上，把行尾的 `\` 換成反引號 `` ` ``，把 `export X=1` 換成 `$env:X="1"`。

---

## 0. 開始前必讀

1. **本 repo 只收錄論文用到的三個模型，其他 checkpoint 和資料都不在這裡。** `weights/`（Git LFS）裡的三個檔都是訓練時存下的原檔，沒有重新存過（來源：weights/README.md、weights/SHA256SUMS.txt）：

   | 本 repo 的檔案（從 `distill2/` 看的路徑） | 原始訓練路徑（本文指令裡的名稱） | 內容 |
   |---|---|---|
   | `../weights/vggt_teacher_init_842M.pt.part0` + `.part1`，先合併成 `../weights/vggt_teacher_init_842M.pt`（§3.2） | `runs/distill_b1/best.pt` | 842M teacher-initialized student，合併後 3,368,981,430 bytes |
   | `../weights/vggt_sliced_464M_agg256_ch512.pt` | `runs/sliced/recover_agg256_ch512/best.pt` | 464M 論文主模型，1,908,372,429 bytes |
   | `../weights/vggt_sliced_464M_int8.pt` | `runs/sliced/quantized_sliced_int8.pt` | 464M 的 INT8 weight-only 版，537,215,008 bytes；載入方式見 [weights/README.md](../weights/README.md) |

   - 其他 checkpoint（消融、層剪枝、其他切片設定、GT 微調等，也就是 `runs/`、`abl/` 底下的其餘檔案）和 teacher soft-label 資料集都不在 repo 裡。`.gitignore` 排除了 `*.pt`（`weights/*.pt` 例外）、`*.npy`、`runs/`、`abl/`、`data/`，第 1 行也註明了這一點（來源：.gitignore）。
   - 所以：資料集一定要自己產生（§4）；用到 842M 或 464M 的步驟，可以直接拿 `weights/` 的檔案代替本文指令裡的 `runs/...` 路徑；其餘模型要照本文自己訓練。
   - `weights/` 的檔案保留了原始 checkpoint 的欄位，可以直接讀出 epoch 和訓練中 eval 的結果（不會把權重全部讀進記憶體）：
     ```bash
     python -c "import torch; ck=torch.load('../weights/vggt_sliced_464M_agg256_ch512.pt',map_location='cpu',mmap=True,weights_only=False); print(ck['epoch'], ck['eval'], ck['k_agg'], ck['k_pe'], ck['k_ch'])"
     ```
2. **資料集沒辦法逐位元重建。** 論文用的是 427 個 train、65 個 test 場景，涵蓋 33 個類別，訓練時完全不用 ground truth（來源：論文 §4.1）。但產生這份資料集的確切參數（類別清單、`--max_scenes`、`--max_images_per_scene`）沒有留下紀錄（見第 4.2 節）。重建出來的 split 會不一樣，絕對數字也會跟著不一樣。
3. **沒有固定 seed，也沒做過同設定的重複實驗。** `train.py` 和 `data.py` 都沒有固定 seed（來源：distill2/README.md 八）。depth head 和 point head 用的是不固定 seed 的隨機初始化（來源：distill2/README.md 四之二）。現有三個「複製全部」的 run 設定各不相同，quick_eval（前 8 個 test 場景）的結果分別是 `runs/distill_b1`（40 epochs）3.42°、`runs/distill_b1_full`（20 epochs）4.44°、消融的 `full`（12 epochs）4.32°（來源：distill2/README.md 八）。論文也說，相差 1 到 2 度的列不應過度解讀（來源：論文 §5 Limitations）。所以重現時，差 ±1–2° 算正常。
4. **同一個 checkpoint，換個評估子集，數字就不一樣。** 各腳本預設的子集不同（第 15.5 節）：
   - 訓練中的 quick_eval：排序後的前 8 個 test 場景
   - 探針和零重訓切片：前 12 個
   - `eval.py --num_scenes 65`：全部

   例如同一個 `runs/distill_b1/best.pt`（＝`../weights/vggt_teacher_init_842M.pt`），在前 8 個場景是 3.42°，在前 12 個場景是 2.87°（來源：distill2/reports/quantize_report.md；distill2/reports/slicegpt_probe.md）。
5. **「RRA@15」有兩種，意義不同。**
   - **vs teacher**（Tables 1、3、4）：對照 teacher soft label。程式算的是「逐幀旋轉誤差（以第 0 幀為座標原點）小於 15° 的比例」（來源：distill2/eval.py 第 88、98 行；distill2/README.md 五之三）。
   - **vs GT**（Table 2）：對照 CO3D 官方標註，算的是成對相對旋轉。

   同一個 464M 模型，前者是 0.969（前 8 個 test 場景，也就是權重檔 `eval` 欄位存的 quick_eval 值），後者是 0.883、AUC@30 0.767（來源：論文 Table 4、Table 2）。

---

## 1. 論文的每個表和圖由哪個腳本產生

| 論文 | 內容 | 腳本（`distill2/`） | 紀錄檔（`distill2/reports/`） | 本文 |
|---|---|---|---|---|
| Table 1 | teacher-init 消融 | `run_ablation.py` | `ablation_report.md/.csv` | §6 |
| Table 2 | 相對位姿 vs CO3D GT | `eval_gt.py --ref co3d` | `eval_gt_co3d10_{pathb,k384,ch512}.md` | §11 |
| Table 3 | 層剪枝 | `run_layer_pruning.py`；latency 用 `measure_unified_bench.py` | `layer_pruning_report.md/.csv`、`unified_bench.json` | §7、§12 |
| Table 4 上半 | aggregator 的探針、零重訓、恢復 | `slicegpt_probe.py`、`slicegpt_slice.py`、`train_sliced.py` | `slicegpt_probe.md`、`slicegpt_slice.md` | §8、§9 |
| Table 4 下半 | 三流切片 | `slicegpt_slice.py`（三流校準）、`train_sliced.py` | `slicegpt_slice_combo.md` | §9 |
| Table 5 | 效率 | `measure_unified_bench.py`、`measure_budget_incl_context.py`（frozen 模式，取 GFLOPs）、`measure_flops_extra.py`、`measure_vram_curve.py`、`measure_vram_int8.py`、`measure_efficiency.py`（† 那一列） | `unified_bench.json`、`budget_incl_context_frozen.json`、`flops_extra.json`、`vram_curve.json`、`vram_curve_int8.json`、`efficiency_report.md` | §12 |
| Table 6 | 各元件參數量 | 沒有專用腳本，從 state_dict 計數（§9.7 附指令） | — | §9.7 |
| Table 7 | 模擬記憶體預算下能處理的幀數 | `measure_budget_incl_context.py`（paper、frozen 兩種模式） | `budget_incl_context_{paper,frozen}.json` | §12 |
| Fig. 2 | 峰值記憶體 vs 幀數 | `measure_vram_curve.py` | `vram_curve.json` | §12 |
| §4.7 | INT8 與儲存大小 | `quantize.py` | `quantize_sliced_report.md` | §10 |
| §4.1 | 隨機取幀的穩健性檢查 | `eval_gt.py --random_frames 0` | `eval_gt_co3d10rand_*.md` | §11.2 |
| §4.3 | 直接用 GT 微調 | `make_gt_labels.py`，加上 `train.py` / `train_sliced.py --gt_pose 1` | `gtft_report.md` | §11.3 |
| §3.5、App. C | 切片等價性測試 | `slicegpt_slice.py --check` | `slicegpt_slice_combo.md` 標頭 | §9.1 |
| App. D | CPU 延遲 | `measure_budget_incl_context.py`（frozen 模式） | `budget_incl_context_frozen.json` | §12 |
| §3.2、Fig. 1 的 82°（146.4M scratch student） | 舊管線的初步診斷 | 這個 run 早於 distill2 管線，本 repo 無法重跑 | `eval_results_before.json`（舊評估腳本的輸出） | §15.1 |

---

## 2. 硬體、磁碟與耗時

**硬體**
- 所有實驗都在單張 RTX 5080（16 GB）上完成（來源：論文 §4.1；distill2/README.md 二）。
- CPU 延遲是在同一台桌機上用 8 threads 量的（來源：論文 App. D；distill2/reports/budget_incl_context_frozen.json 的 `cpu_latency.threads`）。
- 系統 RAM 和訓練時的峰值 VRAM 都沒有紀錄。

**推論記憶體**（S=4、518×518、bf16、no_grad、參數維持預設的 trainable 狀態）：

| 模型 | 峰值 VRAM |
|---|---|
| teacher | 9.21 GB |
| 842M | 5.85 GB |
| 464M | 3.89 GB |
| 464M + INT8 | 1.81 GB |

（來源：distill2/reports/unified_bench.json）

**磁碟**
- 一個 842M 等級的 checkpoint 是 3,368,981,430 bytes（來源：weights/README.md）。`train.py` 每次 eval 都會寫 `last.pt`，有進步時再寫 `best.pt`，所以每個 run 會留兩份（來源：distill2/train.py 第 135–139 行）。
- 原始機器上，`distill2/runs/` 共 119 GB，`distill2/abl/`（消融和剪枝的初始檔）共 27 GB（作者開發紀錄：2026-10-17 對這兩個目錄執行 `du -sh`）。
- 如果要全部重跑，請預留 150 GB 以上（推算：119 + 27，還要再加 teacher 權重和資料集）。clone 本 repo 時，LFS 權重另外約 5.8 GB（來源：weights/README.md）。

**耗時**（只列有紀錄、或能從檔案時間推得的）

| 階段 | 設定 | 耗時 | 依據 |
|---|---|---|---|
| 恢復蒸餾 | 6 epochs × 400 steps | 約 25 分鐘 | 論文 §3.5 |
| 消融，每組 | 12 epochs × 400 steps，`--img_per_seq 2 --throttle 0.05` | 約 50–60 分鐘；`full_ft_pe` 那組約 7.6 小時 | 推算（作者開發紀錄：原始 `abl/*_init.pt` 和 `runs/abl/*/last.pt` 的檔案修改時間） |
| 消融，六組全部 | 同上 | README 建議過夜 | distill2/README.md 五 |
| 層剪枝 | 12 epochs × 400 steps | d8 約 49 分鐘、d6 約 31 分鐘、d4 約 26 分鐘 | 推算（作者開發紀錄：原始 `abl/prune_*_init.pt` 和 `runs/prune/*/last.pt` 的檔案修改時間） |
| 主線訓練 | 40 epochs × 500 steps | 沒有 log。`runs/distill_b1` 從 epoch 37 的 `best.pt` 到 epoch 39 的 `last.pt` 相隔約 68 分鐘（約 34 分鐘/epoch），可當量級參考 | 推算（作者開發紀錄：檔案修改時間） |
| `make_gt_labels.py` | 492 個場景 | 約數分鐘 | distill2/make_gt_labels.py docstring |

其他階段（匯出資料、探針、切片、量測）的耗時都沒有紀錄。

---

## 3. 環境

### 3.1 紀錄中的版本

| 項目 | 紀錄值 | 來源 |
|---|---|---|
| OS / shell | Windows、PowerShell | distill2/README.md 四、五之三（PowerShell 指令）；作者開發紀錄 |
| GPU | RTX 5080 16 GB | 論文 §4.1 |
| PyTorch | 2.10.0+cu128 | 論文 §4.7；`reports/budget_incl_context_*.json` 的 `"torch"` 欄 |
| torchao | 0.17（`Int8WeightOnlyConfig`） | 論文 §4.7 |
| Python | 確切版本沒有紀錄；`pyproject.toml` 要求 `>= 3.10` | pyproject.toml |
| vggt 套件的依賴 | `numpy<2`、Pillow、huggingface_hub、einops、safetensors、opencv-python | pyproject.toml `dependencies` |
| distill2 的依賴 | numpy、opencv-python、matplotlib、fvcore；torchao 為選用 | distill2/README.md 二 |
| torchvision | 版本沒有紀錄。`vggt/utils/load_fn.py` 會 import 它，匯出 soft label 時會用到 | vggt/utils/load_fn.py 第 9 行 |
| nvidia-smi | `measure_budget_incl_context.py` 用它讀 GPU 已用記憶體，必須在 PATH 上 | distill2/measure_budget_incl_context.py 第 26–29 行 |

### 3.2 安裝

```bash
# 要先裝 git-lfs，clone 時才會一起下載 weights/ 的三個模型（約 5.8 GB）
git clone https://github.com/yoyo6108876/VGGT-final.git
# 只要程式、不要權重：GIT_LFS_SKIP_SMUDGE=1 git clone ...，之後再 git lfs pull --include "weights/<檔名>"
cd VGGT-final

# 842M 的檔案超過 GitHub LFS 單檔 2 GB 上限，切成兩段，用之前先合併（Windows cmd 用 copy /b，見 weights/README.md）
cat weights/vggt_teacher_init_842M.pt.part0 weights/vggt_teacher_init_842M.pt.part1 > weights/vggt_teacher_init_842M.pt
sha256sum weights/vggt_teacher_init_842M.pt     # 應和 weights/SHA256SUMS.txt 的 df35c85a…a05 相同

python -m venv .venv
source .venv/bin/activate          # Windows PowerShell：.venv\Scripts\Activate.ps1

# PyTorch：紀錄的版本是 2.10.0+cu128，下面是對應這個版本的安裝寫法
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# 本 repo 的 vggt 套件（含 VGGTStudent），用 editable 模式安裝
pip install -e .

# distill2 需要的其他套件（torchao 只有 INT8 和效率量測才用到，載入 INT8 權重檔也需要）
pip install matplotlib fvcore "torchao==0.17.*"
```

- 權重的下載、合併、載入方式見 [weights/README.md](../weights/README.md)。
- `pip install -e .` 會依 `pyproject.toml` 安裝 `vggt*` 套件（`[tool.setuptools.packages.find] include = ["vggt*"]`）和 3.1 表列的依賴。
- teacher `facebook/VGGT-1B` 第一次使用時會從 Hugging Face 下載：
  - `VGGT.from_pretrained` 用於 `build_student.py`、匯出腳本和各量測腳本
  - `verify_teacher_copy.py` 用的是 `hf_hub_download("facebook/VGGT-1B", "model.safetensors")`
  - fp32 大小約 5025 MB（來源：論文 §1）
- 消融中有三組不複製前端（random、agg_only、head_only）。這三組會用 `torch.hub.load_state_dict_from_url` 下載官方的 DINOv2 ViT-L/14-reg 權重，所以需要網路（來源：vggt/models/aggregator.py 第 201–212 行；distill2/build_student.py 第 55 行）。

### 3.3 import 路徑陷阱（一定要檢查）

**紀錄中發生過的問題**
- 開發機上同時有兩份 `vggt`：
  - editable install 指向的那份是正確版，camera head 的 `trunk_depth=4`。
  - 專案根目錄下另有一份舊版，用 conv patch embed，`trunk_depth=2`。
- 只要某個腳本把舊版所在目錄插到 `sys.path` 最前面，`import vggt` 就會在沒有任何提示的情況下載到舊版。舊版會把訓練時的 4 層 camera head 截成 2 層，旋轉評估因此失真（來源：distill2/CHANGES.md §3(C)；作者開發紀錄）。
- `distill2/` 的腳本都不改 `sys.path`（來源：distill2/measure_efficiency.py 第 30 行註解；`distill2/*.py` 裡沒有任何 `sys.path` 操作）。

**本 repo 的情況**
- `vggt/` 只有一份，就是產生論文結果時用的那份模型程式（來源：distill2/README.md 二：作者 fork 的 commit `3c14206`，另加 4 個空的 `__init__.py`）。
- 這 4 個 `__init__.py` 在本 repo 已經有了：`vggt/__init__.py`、`vggt/heads/__init__.py`、`vggt/models/__init__.py`、`vggt/utils/__init__.py`，都是 0 bytes。
- `scripts/export_co3d_labels.py` 會把 repo 根目錄插到 `sys.path` 最前面（第 48 行）。在本 repo 裡它指向的就是同一份 `vggt/`，沒有問題。但如果你把腳本搬到別的專案，或環境裡另外裝了別的 `vggt`，就可能重演上面的問題。

**檢查指令**

```bash
python -c "import vggt, inspect, vggt.models.vggt_student as s; print(vggt.__file__); print('trunk_depth=4' in inspect.getsource(s))"
```

- 預期第一行是 `<REPO>/vggt/__init__.py`，第二行是 `True`。
- 載入 checkpoint 時，`eval.load_student` 應該印出 `[載入 OK] 完全匹配`。如果印出 `[警告] missing: ...` 或 `[警告] unexpected: ...`，代表架構對不上（來源：distill2/eval.py 第 39–47 行）。可以拿 `../weights/vggt_teacher_init_842M.pt` 直接試：`python -c "from eval import load_student; load_student('../weights/vggt_teacher_init_842M.pt','cpu')"`。

---

## 4. 資料準備

### 4.1 CO3D 原始資料

各腳本預期的目錄結構如下：

```
<CO3D_ROOT>/
└── <category>/                   # 例如 apple
    ├── frame_annotations.jgz     # eval_gt.py / make_gt_labels.py 從這裡讀 GT 外參
    └── <sequence_name>/
        └── images/               # export_co3d_labels.py 從這裡讀影像
            ├── frame000001.jpg
            └── ...
```

（來源：scripts/export_co3d_labels.py docstring 和 `main()`；distill2/eval_gt.py 第 92–116 行）

- 本 repo 不提供 CO3D，請自行下載後放到 `<CO3D_ROOT>`。
- 各腳本 docstring 和 help 裡寫的 co3d_root 範例路徑只是範例，並不是原始機器上的實際位置，實際資料夾用的是別的名稱（作者開發紀錄）。用你自己的路徑即可。

### 4.2 產生 teacher soft labels：`scripts/export_co3d_labels.py`

在 `<REPO>` 下執行：

```bash
python scripts/export_co3d_labels.py \
    --co3d_root <CO3D_ROOT> \
    --categories <cat_1> <cat_2> ... \
    --out_root <DATA_ROOT> \
    --save_points \
    --max_images_per_scene 20 \
    --test_ratio 0.1 \
    --seed 42
```

**腳本的行為**（來源：scripts/export_co3d_labels.py）

- 一個序列要有 `images/`，而且影像數不少於 `--min_images`（預設 5），才會被處理（第 310 行）。
- 每個類別打散後切出 test：`n_test = max(1, round(n × test_ratio))`（第 322–323 行）。所以每個類別至少有 1 個 test 場景。打散用的是同一個 `random.Random(--seed)`，只在開頭建立一次（第 288 行），依 `--categories` 的順序依序使用；所以類別的順序不同，split 也會不同。
- 每個場景均勻抽出 `--max_images_per_scene` 張影像（`subsample_images()`）。預設是 10；help 文字建議 RTX 3050 用 8、RTX 5080 用 20–30。
- 抽出的影像預處理成 518×518 後，一次送進 VGGT-1B 推論。輸入影像在 compute capability ≥ 8 時轉成 bf16，否則轉成 fp16（第 275–279 行）；腳本沒有開 autocast，模型權重維持 fp32。
- 已經有 `meta.json` 的場景會跳過（第 127 行），所以可以中斷後續跑。失敗的場景會刪掉整個輸出資料夾（第 362 行）。
- 場景資料夾命名為 `<category>__<sequence>`（第 333 行）。`eval_gt.py` 靠這個名字回頭找 CO3D 標註，請不要改名。

**參數怎麼選**
- **一定要加 `--save_points`。** `data.py` 用 `world_points.npy` 做尺度正規化，沒有這個檔時 scale 會固定為 1（來源：distill2/data.py 第 54–61、113–114 行）。論文 §4.1 也把 world points 列為儲存的預測之一。
- **`--max_images_per_scene` 至少設 12。** teacher-relative 指標每個場景取 12 個等距幀，GT 指標取 10 個（來源：論文 §4.1；distill2/eval.py 第 70–74 行）。另外，`slicegpt_slice.py --check` 會用到第一個 test 場景的第 0 和第 5 幀（來源：distill2/slicegpt_slice.py 第 332 行）。
- **論文資料集的原始參數沒有紀錄。** 用了哪 33 個類別、`--max_images_per_scene` 設多少、有沒有用 `--max_scenes`，紀錄裡都找不到。

### 4.3 單一場景：`scripts/export_teacher_labels.py`

這支腳本處理你自己的一個影像資料夾，和 4.2 的差別如下（來源：scripts/export_teacher_labels.py）：
- 輸出到 `<out_root>/<scene_name>/`，**不分 train/test**。要用於訓練或評估，請自己把它移到 `train/` 或 `test/` 底下。
- **不抽樣。** 資料夾裡的所有影像一次送進 teacher，張數太多會 OOM。
- 輸出資料夾如果已存在，會先整個刪除（`shutil.rmtree`）。
- `scene_name` 預設取資料夾名稱（資料夾本身叫 `images` 時改取上一層的名稱）。如果要做 GT 評估，名字必須是 `<category>__<sequence>`。

### 4.4 產出的資料夾結構

```
<DATA_ROOT>/
├── train/                          # 論文：427 個場景
│   └── <category>__<sequence>/
│       ├── images/0000.png ...     # 預處理後的 518×518 影像
│       ├── depth.npy               # [S,H,W]   teacher 深度
│       ├── depth_conf.npy          # [S,H,W]
│       ├── extrinsics.npy          # [S,3,4]   由 pose_enc 解碼
│       ├── intrinsics.npy          # [S,3,3]
│       ├── pose_enc.npy            # [S,9]     (t, q, FoV)；四元數為 scalar-last (x,y,z,w)
│       ├── point_mask.npy          # [S,H,W]   depth > 1e-8
│       ├── world_points.npy        # [S,H,W,3] 需要 --save_points
│       ├── world_points_conf.npy   # [S,H,W]   需要 --save_points
│       ├── original_coords.npy     # [S,6]
│       ├── meta.json               # scene_name、num_images、image_paths、source_images、teacher_model、target_size、image_hw、save_points、files
│       └── gt_extrinsics.npy       # 選用，由 make_gt_labels.py 產生（§11.3）
└── test/                           # 論文：65 個場景
```

（來源：scripts/export_co3d_labels.py `process_scene()`；distill2/data.py 第 4–13、28–29 行；train/test 場景數見論文 §4.1、distill2/CHANGES.md §1；scalar-last 見 distill2/loss.py 第 8 行）

- `data.py` 只收齊下面 7 個檔的場景：`meta.json`、`depth.npy`、`depth_conf.npy`、`extrinsics.npy`、`intrinsics.npy`、`pose_enc.npy`、`point_mask.npy`。缺任何一個，該場景就會被**直接略過，不報錯**（來源：distill2/data.py 第 28–39 行）。
- 訓練時的 pose 目標由 `data.py` 重新計算：先把第一個抽到的幀設為 identity，再做尺度正規化，然後重算 `pose_enc`。評估時（`eval.py`、`eval_gt.py`）則直接拿存檔的 `pose_enc.npy` 當參考（來源：distill2/data.py `_normalize()`；distill2/eval.py 第 81 行；distill2/eval_gt.py 第 136–139 行）。

### 4.5 檢查資料集

```bash
cd distill2
python -c "from data import TeacherDataset; d=TeacherDataset('<DATA_ROOT>','train',2,10); b=d[0]; print({k: tuple(v.shape) for k, v in b.items()})"
```

- 預期先印出 `[TeacherDataset:train] N scenes`。論文的資料集是 427，test 是 65。
- 接著印出 shape：`images` (2,3,518,518)、`depth` (2,518,518)、`depth_conf` (2,518,518)、`pose_enc` (2,9)、`point_mask` (2,518,518)（來源：distill2/data.py docstring 第 8–13 行）。

**注意排序。** 切片校準用的是字典序排在前 8 個的 train 場景。在論文的資料集裡，這 8 個全部是 apple 類別（來源：論文 §3.5）。你自己匯出的資料集，類別組成不同，校準場景也就不同。

### 4.6 各腳本 `--data_root` 要給哪一層

| 給 `<DATA_ROOT>`（含 `train/`、`test/`） | 給 `<DATA_ROOT>/test` |
|---|---|
| `train.py`、`train2.py`、`train_sliced.py`、`run_ablation.py`、`run_layer_pruning.py`、`slicegpt_probe.py`、`slicegpt_slice.py`、`make_gt_labels.py` | `eval.py`、`eval_gt.py`、`quantize.py`、`visualize.py` |

（來源：各腳本 argparse 的 help 和使用方式）`run_ablation.py`、`run_layer_pruning.py`、`slicegpt_probe.py`、`slicegpt_slice.py` 的 `--data_root` 預設值是原開發機的絕對路徑，**一定要自己指定**。

---

## 5. 主線：teacher-initialized student（842M）

> **不想重訓的話**：`../weights/vggt_teacher_init_842M.pt`（合併後）就是下面 §5.3 那個 run 的 `best.pt`（epoch 37），可以直接拿來做 §5.4 以後的所有步驟。要自己重訓，照下面的順序做。

### 5.1 從 teacher 切出 student

```bash
cd distill2
python build_student.py --out student_init.pt --copy all
```

**`--copy all` 時的權重對應**（來源：distill2/README.md 四之二；distill2/build_student.py）

| student | 來自 teacher 的 | 張量數 |
|---|---|---|
| `aggregator.patch_embed`（DINOv2 ViT-L/14-reg，訓練時凍結） | 同名，整個複製 | 344 |
| `aggregator.camera_token` / `register_token` | 同名 | 1 + 1 |
| `aggregator.frame_blocks[i]` / `global_blocks[i]`，i = 0…11 | `[2i]`，每隔一層取一層 | 216 + 216 |
| `camera_head` | 同名，整個複製 | 69 |
| `depth_head` / `point_head` | **不複製**，隨機初始化 | 62 + 62 |

- 寬度維持 1024，16 個 attention heads，aggregator 12 層。不建 track head（`enable_track=False`）。
- **預期輸出**：`patch_embed`、每個 `frame[i]<-t[j]` / `global[i]<-t[j]` block、`camera_head` 各印一行 `[複製 ...] N keys → OK`（camera/register token 直接複製，不印），最後印 `✓ 已存 student_init.pt｜842.2M｜copy=['aggregator', 'camera_head', 'patch_embed']`（來源：distill2/build_student.py 第 26–30、72–83 行）。842.2M 是推算值，和 distill2/reports/efficiency_report.md 裡同架構的 842.2M 一致。原始檔大小是 3,369,037,438 bytes（作者開發紀錄）。
- **版本說明**：原始的 `student_init.pt` 不是由目前這份 `build_student.py` 產生的。不過兩版在 `--copy all` 時複製的元件、層對應和 `load_state_dict(strict=False)` 的做法都相同（來源：distill2/README.md 四之二）。

### 5.2 驗證權重複製（不需要 GPU）

```bash
python verify_teacher_copy.py --ckpt student_init.pt
# 訓練完可以一起驗：
python verify_teacher_copy.py --ckpt student_init.pt runs/distill_b1/best.pt
# 只想驗 repo 裡的 842M（和 runs/distill_b1/best.pt 是同一個檔）：
python verify_teacher_copy.py --ckpt ../weights/vggt_teacher_init_842M.pt
```

這支腳本用 `torch.load(mmap=True)` 讀 student，用 safetensors 逐張量讀 teacher，記憶體需求低。2026-10-17 實跑的結果如下（來源：distill2/README.md 四之二）：

| 檔案 | 應複製的 847 個張量 vs teacher | depth / point head（124 個） |
|---|---|---|
| `student_init.pt` | 全部逐位元相同（0 個不同） | 不同（沒有複製，符合預期） |
| 訓練後的 `runs/distill_b1/best.pt`（＝`../weights/vggt_teacher_init_842M.pt`） | `patch_embed` 的 344 個仍然相同（證明凍結生效），其餘 503 個已被訓練更新 | 不同 |

### 5.3 訓練

```bash
python train.py --student_init student_init.pt --data_root <DATA_ROOT> \
    --out_dir runs/distill_b1 --epochs 40 --steps_per_epoch 500 --eval_every 2
# 可選：--throttle 0.05（防止螢幕黑屏，見 §15.7）；--img_per_seq 2（16 GB 不夠時）
```

> **論文與紀錄的差異：訓練長度**
> - **論文** §4.1 寫 "the main run is 20 epochs of 500 steps"。作者開發紀錄裡較早的指令寫的也是 `--epochs 20`。
> - **紀錄**：論文用到的 842M checkpoint `runs/distill_b1/best.pt`（＝本 repo 的 `weights/vggt_teacher_init_842M.pt`），其實來自 **40 epochs × 500 steps** 的訓練。證據是 `best.pt` 的 `epoch` 欄位是 37（可以從 repo 的權重檔讀出：先照 §3.2 合併，再把 §0 第 1 點指令的路徑換成 `../weights/vggt_teacher_init_842M.pt`，並拿掉 `ck['k_agg'], ck['k_pe'], ck['k_ch']`），`last.pt` 的 `epoch` 欄位是 39（作者開發紀錄）；而 `train.py` 的預設值正是 `--epochs 40 --steps_per_epoch 500`（第 41–42 行）。distill2/README.md 四也把 distill_b1 記為「train.py 預設（40 epochs）」，並註明當時的指令沒有另外存檔，是依 checkpoint 的 `epoch` 推定的。
> - 另外有一個 20-epoch 的 run，`runs/distill_b1_full`（`--epochs 20 --eval_every 2 --img_per_seq 2 --throttle 0.1`，best epoch 19）。它只用在 README 的「82.7° → 4.9°（65 場景）」那一行，**沒有**用在切片、GT 評估、量化和效率的結果（來源：distill2/README.md 四、七）。
>
> **論文與紀錄的差異：每段序列的幀數**
> - **論文** §4.1 寫 "two frames per sampled sequence"。
> - **紀錄**：README 四把 distill_b1 記為「train.py 預設」，而 `--img_per_seq` 的預設值是 4（train.py 第 43 行）。checkpoint 不存訓練參數，所以 distill_b1 實際用了幾幀，**沒有辦法核對**。作者開發紀錄裡那條較早的 distill_b1 指令寫的是 `--img_per_seq 2`（和論文一致），但同一條指令的 `--epochs 20` 和 checkpoint 不符，所以也不能當成實際設定。

**訓練配方**（train.py 預設值，與論文 §4.1、App. B 相同）

| 項目 | 值 | 程式位置 |
|---|---|---|
| optimizer | AdamW，lr 5e-5，weight decay 0.05 | train.py 第 45–46、80 行 |
| 學習率排程 | 前 5% 線性 warm-up，之後 cosine | 第 29–33、47 行 |
| 精度 | bf16 autocast；grad clip 1.0 | 第 103、110 行 |
| 前端 | 凍結（`--freeze_patch_embed 1`） | 第 51、69–74 行 |
| loss 權重 | rotation 4.0、translation+FoV 2.0、depth 1.0 | loss.py 第 74–75 行 |
| rotation loss | `1 − \|dot\|`，絕對旋轉與相鄰幀相對旋轉相加成一項；rotation、translation+FoV、depth 三項各自用 EMA 正規化（momentum 0.99，warm-up 100 步）；相機 loss（旋轉與平移/FoV）在 refinement stage 間以 γ=0.6 衰減 | loss.py |
| depth loss | 以 teacher confidence 加權的 MSE | loss.py |
| eval | 每 `--eval_every` 個 epoch（以及最後一個 epoch）對前 `--eval_scenes` 個 test 場景（預設 8）跑一次 quick_eval；rot_err 創新低就存 `best.pt` | train.py 第 129–140 行 |

**預期輸出**
- 每 50 步印一行 `[e{epoch} s{step}] lr=... loss_rot=... loss_transfov=... loss_depth=... objective=...`。
- 每次 eval 印一行 `[EVAL e..] rot_err=... rra15=... abs_rel=... d1=...`。
- 啟動時印 `已凍結 patch_embed：304.4M 參數`（推算：§9.7 的前端是 304.37M）。
- 原始 run 的 `best.pt` 在 epoch 37：rot_err 3.42°、rra_15 1.000、abs_rel 0.177、δ1 0.752（8 個場景；來源：`weights/vggt_teacher_init_842M.pt` 的 `eval` 欄位，也就是原始 `runs/distill_b1/best.pt`；3.42° 也見 distill2/reports/quantize_report.md）。
- 如果 eval 的 rot_err 一直停在 75–82°，就是塌縮了，請看 §15.1。

### 5.4 評估 vs teacher soft labels

```bash
python eval.py --ckpt runs/distill_b1/best.pt --data_root <DATA_ROOT>/test --num_scenes 65
# 用 repo 裡的權重：--ckpt ../weights/vggt_teacher_init_842M.pt
```

先印 `[載入 OK] 完全匹配`，再印 `mean_rot_err`、`rra_15`、`abs_rel`、`delta_1`。每個場景最多取 12 個等距幀（eval.py 第 70–74 行）。已有紀錄的對照值（在論文的資料集上；你自己匯出的資料集數字會不同，見 §0 第 2 點）：

| checkpoint | `--num_scenes` | rot_err | RRA@15（逐幀） | 來源 |
|---|---|---|---|---|
| `runs/distill_b1/best.pt`（＝`../weights/vggt_teacher_init_842M.pt`） | 8 | 3.42° | 1.000 | 權重檔的 `eval` 欄位；reports/quantize_report.md |
| 同上 | 12 | 2.87° | 1.000 | reports/slicegpt_probe.md 的 baseline |
| `runs/distill_b1_full/best.pt`（20 epochs，不在 repo） | 65 | 4.89° | 0.968 | distill2/CHANGES.md §4；distill2/README.md 四、七 |

`runs/distill_b1` 在全部 65 個場景上的值沒有紀錄。

### 5.5 相對位姿 vs teacher（不在論文的表格裡）

```bash
python eval_gt.py --ckpt runs/distill_b1/best.pt --data_root <DATA_ROOT>/test --out_prefix reports/eval_gt_report_rerun
```

預設是 20 個場景、每場景 8 幀（eval_gt.py 第 146–147 行）。`--out_prefix` 請用新檔名，`reports/eval_gt_report` 會蓋掉原始報告。紀錄結果：RRA@15 0.984、RRA@30 0.988、RTA@15 0.977、RTA@30 0.986、AUC@30 0.888（來源：distill2/reports/eval_gt_report.md）。

---

## 6. 消融（Table 1）

```bash
export PYTHONUTF8=1      # 原因見 §15.6
python run_ablation.py --stage all --data_root <DATA_ROOT> \
    --epochs 12 --img_per_seq 2 --throttle 0.05 --out_prefix reports/ablation_report_rerun
# 只收表：python run_ablation.py --stage collect --out_prefix reports/ablation_report_rerun
# 只跑部分組合：--only head_only,full
```

**流程**（來源：distill2/run_ablation.py）
- `build`：對每組呼叫 `build_student.py`，產生 `abl/<name>_init.pt`。
- `train`：呼叫 `train.py`，固定 `--eval_every 3`，輸出到 `runs/abl/<name>/`。
- `collect`：從各組 `best.pt` 的 `eval` 欄位收成表。
- `--out_prefix` 是相對 `distill2/` 的路徑。預設是 `ablation_report`，會寫在 `distill2/` 根目錄而不是 `reports/`。原始報告是 `reports/ablation_report.md/.csv`，重跑時請用新檔名，以免蓋掉。
- 這六組的 checkpoint 都不在 repo 裡，要自己跑。

**參數說明**
- `--epochs` 的預設是 15，README 的指令用 12（distill2/README.md 五）。原始六組的 `last.pt` 都是 `epoch=11`，也就是全部跑了 12 epochs × 400 steps（作者開發紀錄：原始 `runs/abl/*/last.pt` 的 `epoch` 欄位）。論文 §4.1 寫的是 "12–15 epochs of 400 steps"。

| 組合（腳本名稱） | 論文的列名 | `--copy` | 凍結前端 | rot_err | RRA@15 | abs_rel | δ1 |
|---|---|---|---|---|---|---|---|
| random | random | none | ✓ | 76.18° | 0.188 | 0.235 | 0.642 |
| patch_only | FE only | patch_embed | ✓ | 75.59° | 0.198 | 0.260 | 0.616 |
| agg_only | agg. only | aggregator | ✓ | 75.58° | 0.208 | 0.265 | 0.574 |
| head_only | head only | camera_head | ✓ | 75.48° | 0.208 | 0.247 | 0.578 |
| full | full | 三者都複製 | ✓ | 4.32° | 1.000 | 0.189 | 0.746 |
| full_ft_pe | full, FE tuned | 三者都複製 | — | 75.68° | 0.219 | 0.248 | 0.632 |

（來源：distill2/reports/ablation_report.md；與論文 Table 1 一致）

**注意**
- 這些值都是訓練中 quick_eval 在前 8 個 test 場景上的結果（train.py `--eval_scenes` 預設 8；run_ablation.py 第 54 行）。論文沒有寫 Table 1 用了幾個場景（論文 §4.1 只寫每個場景取 12 個等距幀）。
- 不複製前端的組合（random、agg_only、head_only），前端是凍結的**官方** DINOv2，需要網路下載。FE only 那組複製的是 teacher 經 VGGT 訓練調整過的前端，兩者不是同一組權重（來源：論文 Table 1 caption）。
- `runs/abl/full` 就是 Table 1 和 Table 3 的 4.32° 那一列。它和後面切片所用的 `runs/distill_b1`（40 epochs，即 `weights/vggt_teacher_init_842M.pt`）是**不同的 checkpoint**（來源：論文 Table 4 caption）。
- `runs/abl/random/best.pt` 之後還會在 §13 被當成「塌縮模型」的對照。

---

## 7. 層剪枝（Table 3）

**前置條件**：`collect` 會讀 `runs/abl/full/best.pt` 當 12 層的基準（run_layer_pruning.py 第 83 行），所以至少要先跑完 §6 的 `full` 那組。剪枝的 checkpoint 都不在 repo 裡。

```bash
export PYTHONUTF8=1
python run_layer_pruning.py --stage all --depths 8,6,4 --data_root <DATA_ROOT> \
    --epochs 12 --img_per_seq 2 --throttle 0.05 --out_prefix reports/layer_pruning_report_rerun
# 只收表：python run_layer_pruning.py --stage collect --depths 8,6,4 --out_prefix reports/layer_pruning_report_rerun
```

**流程**
- 初始檔寫到 `abl/prune_d{d}_init.pt`，訓練輸出到 `runs/prune/d{d}/`。報告預設寫到 `reports/layer_pruning_report.md/.csv`（`--out_prefix` 預設值），會蓋掉原始報告，所以上面的指令改用新檔名。
- teacher 的 24 層按 stride 取層：`j = i × (24 // depth)`。其他配方和消融完全相同（來源：distill2/run_layer_pruning.py docstring）。

| 層數 | 參數 | 壓縮倍數 | rot_err | RRA@15 | abs_rel | δ1 | latency（S=4） |
|---|---|---|---|---|---|---|---|
| 12（基準） | 842M | 1.49× | 4.32° | 1.000 | 0.189 | 0.746 | 219.1 ms |
| 8 | 741M | 1.69× | 9.45° | 0.781 | 0.201 | 0.744 | 180.1 ms |
| 6 | 691M | 1.82× | 56.78° | 0.208 | 0.214 | 0.712 | 161.4 ms |
| 4 | 641M | 1.96× | 75.74° | 0.198 | 0.227 | 0.658 | 144.0 ms |

（來源：精度取自 distill2/reports/layer_pruning_report.md；latency 取自 distill2/reports/unified_bench.json，見 §12；與論文 Table 3 一致）

- Table 3 的 latency 是 `measure_unified_bench.py` 在同一個 session 裡量的。`reports/efficiency_prune.md` 的 222.8 / 186.2 / 165.3 / 145.9 ms 是較早另一個 session 的值，不要混用（見 §16）。
- **延長訓練**（不在論文中）：d8 用 24 epochs 訓練（`runs/prune/d8_long`，`last.pt` 的 epoch=23），得到 9.94°，仍在 9.5–10° 的平台期（作者開發紀錄；原始 checkpoint 的 `epoch`、`eval` 欄位）。
  - 重建的指令：`python train.py --student_init abl/prune_d8_init.pt --data_root <DATA_ROOT> --out_dir runs/prune/d8_long --epochs 24 --steps_per_epoch 400 --img_per_seq 2 --eval_every 3 --throttle 0.05`

---

## 8. 光譜探針（Table 4 的 Probe 欄）

```bash
python slicegpt_probe.py --ckpt runs/distill_b1/best.pt --data_root <DATA_ROOT> \
    --ks 1024,896,768,640,512,384,256 --out reports/slicegpt_probe_rerun.md
# 用 repo 裡的權重：--ckpt ../weights/vggt_teacher_init_842M.pt
```

**做法**
- 用 forward hook 把每個 aggregator block 的輸出投影到前 k 維：`μ + (y−μ) Q_k Q_kᵀ`。這裡的基底是**每個 block 各自的、中心化的** PCA，**完全不動權重**。
- 校準：前 8 個 train 場景 × 6 幀（`--calib_scenes 8 --calib_frames 6`）。評估：前 12 個 test 場景（`--eval_scenes 12`）（來源：distill2/slicegpt_probe.py）。
- `--out` 的預設值是 `reports\slicegpt_probe.md`（原始報告），重跑時請用新檔名。
- 這個基底只用來診斷，和 §9 切片用的「共享、未中心化」基底不同。Table 4 的 PCA 能量欄描述的是探針基底，不是切片基底保留的能量（來源：論文 §3.4–3.5、Table 4 caption）。

| k | 平均 PCA 能量 | rot_err | RRA@15 |
|---|---|---|---|
| 不截斷 / 1024 | 100.00% | 2.87° | 1.000 |
| 896 | 99.93% | 2.99° | 1.000 |
| 768 | 99.82% | 3.06° | 1.000 |
| 640 | 99.65% | 3.22° | 0.993 |
| 512 | 99.38% | 3.45° | 0.993 |
| 384 | 98.91% | 3.86° | 0.986 |
| 256 | 97.98% | 4.91° | 0.958 |

（來源：distill2/reports/slicegpt_probe.md）

---

## 9. 全域投影切片（Table 4）

**共同注意事項**
- `slicegpt_slice.py` 和 `train_sliced.py` 的預設路徑都是 Windows 反斜線寫法；`slicegpt_slice.py` 的 `--data_root` 預設值是原機的絕對路徑，`train_sliced.py` 的 `--data_root` 則是必填。下面的指令把 `--ckpt`、`--data_root`、`--save_dir`、`--out`、`--base_ckpt`、`--q_ckpt`、`--out_dir` 全部寫明，Linux 和 Windows 都能用（來源：distill2/README.md 五之三）。
- `--out` 會**覆寫**報告。`reports/slicegpt_slice_combo.md` 在原始實驗後有手動整理過，所以重跑時請改用新的輸出檔名（來源：distill2/README.md 五之三）。
- 下面的 `--ckpt runs/distill_b1/best.pt` / `--base_ckpt runs/distill_b1/best.pt` 都可以換成 `../weights/vggt_teacher_init_842M.pt`（同一個檔）。切片中間檔（`runs/sliced/sliced_*.pt`）和 agg384、agg256、三流其他組合的恢復模型都不在 repo 裡，要照下面的步驟自己產生；只有論文主模型（§9.5）和它的 INT8 版（§10）在 `weights/`。

### 9.1 等價性測試（一定先做）

```bash
python slicegpt_slice.py --check --ckpt runs/distill_b1/best.pt --data_root <DATA_ROOT>
```

**做法**：在 CPU 上用 fp32 跑。三條殘差流（前端 1024、camera trunk 2048、aggregator 1024）都取 k = 滿維，再各套一個**隨機正交** Q，切片後的模型必須重現原模型的輸出。測試用第一個 test 場景的第 0、5 幀（slicegpt_slice.py 第 326–348 行）。

**判定**：腳本會提示 max|Δ| 在 1e-3 以下就算通過。紀錄值是 pose_enc max|Δ| = 3.0e-7、depth max|Δ| = 2.6e-6（來源：distill2/reports/slicegpt_slice_combo.md 標頭；論文 §3.5、App. C）。

**沒通過的話不要往下做**。可能的原因是 `vggt` 版本不對（§3.3），或程式被改過。

### 9.2 aggregator-only，零重訓掃 k

```bash
python slicegpt_slice.py --run --ckpt runs/distill_b1/best.pt --data_root <DATA_ROOT> \
    --ks 512,384,256 --bench --save_k 512 --save_dir runs/sliced \
    --out reports/slicegpt_slice_rerun.md
```

**做法**：用前 8 個 train 場景 × 6 幀，校準未中心化的二階矩，得到共享基底。權重手術在 CPU 上以 float64 完成後才移到 GPU。評估用前 12 個 test 場景。`--bench` 會額外量 S=4 的延遲（來源：distill2/slicegpt_slice.py）。

| k | 共享基底能量 | rot_err（零重訓） | RRA@15 | storage | latency |
|---|---|---|---|---|---|
| 原模型 | 100.00% | 2.87° | 1.000 | 842M | 223 ms |
| 512 | 99.52% | 5.62° | 0.958 | 692M | 208 ms |
| 384 | 99.12% | 34.60° | 0.382 | 654M | 203 ms |
| 256 | 98.30% | 49.74° | 0.264 | 616M | 199 ms |

（來源：distill2/reports/slicegpt_slice.md；storage 是 state_dict 的總元素數，包含 304M 的凍結前端和 de-slice 矩陣）

**輸出檔名**：照目前的程式，`--save_k 512` 寫出的檔名是 `runs/sliced/sliced_agg512.pt`，因為 label 是 `agg{k}`（slicegpt_slice.py 第 421、437 行）。紀錄和 `train_sliced.py` 預設值裡的 `runs/sliced/sliced_k512.pt` 是舊版腳本的檔名，下一步請自己指定 `--q_ckpt`。

### 9.3 k=384 和 k=256 的恢復蒸餾

```bash
python train_sliced.py --base_ckpt runs/distill_b1/best.pt --q_ckpt runs/sliced/sliced_agg512.pt \
    --k 384 --data_root <DATA_ROOT> --out_dir runs/sliced/recover_k384 --img_per_seq 2 --throttle 0.05
python train_sliced.py --base_ckpt runs/distill_b1/best.pt --q_ckpt runs/sliced/sliced_agg512.pt \
    --k 256 --data_root <DATA_ROOT> --out_dir runs/sliced/recover_k256 --img_per_seq 2 --throttle 0.05
```

**做法**
- `--q_ckpt` 只取存檔裡的完整 Q 和 meta，k 由 `--k` 重新指定。
- 預設配方：6 epochs × 400 steps、lr 2e-5、凍結 DINOv2 本體、每 2 個 epoch 對前 8 個 test 場景做 quick_eval（來源：distill2/train_sliced.py 第 41–49 行；論文 §3.5）。

| k | 零重訓 | 恢復後 | RRA@15 | abs_rel | storage |
|---|---|---|---|---|---|
| 384 | 34.60° | 4.18° | 1.000 | 0.181 | 654M |
| 256 | 49.74° | 4.95° | 0.990 | 0.188 | 616M |

（來源：distill2/reports/slicegpt_slice.md「恢復蒸餾 eval 8 test scenes」；原始兩個 run 的 best 都在 epoch 5（作者開發紀錄：原始 checkpoint 的 `epoch` 欄位））

### 9.4 三流校準（產生主模型要用的 Q）

```bash
python slicegpt_slice.py --run --ckpt runs/distill_b1/best.pt --data_root <DATA_ROOT> \
    --ks 0,256 --k_pe 512 --k_ch 1024 --bench --save_k 256 --save_dir runs/sliced \
    --out reports/slicegpt_slice_combo_rerun.md
```

- 產出 `runs/sliced/sliced_agg256+pe512+ch1024.pt`。存檔裡有完整的 `Q_agg`（1024²）、`Q_pe`（1024²）、`Q_ch`（2048²）。下一步只取其中的完整 Q，k 會重新指定（來源：distill2/README.md 五之三）。
- 原始檔大小是 1,509,565,442 bytes（作者開發紀錄）。
- 零重訓結果：`agg0+pe512+ch1024` 72.82°、`agg256+pe512+ch1024` 74.75°，前端一切就崩。原模型在這個 session 是 224 ms（來源：distill2/reports/slicegpt_slice_combo.md）。
- 切片的順序固定為前端 → camera trunk → aggregator（來源：distill2/slicegpt_slice.py docstring；論文 App. C）。
- 只想重切、不想重新校準的話：`weights/vggt_sliced_464M_agg256_ch512.pt` 裡也存了同一次校準的完整 `Q_agg`、`Q_pe`、`Q_ch`（distill2/README.md 五之三「存檔內容」），可以拿來當 `--q_ckpt ../weights/vggt_sliced_464M_agg256_ch512.pt`。

### 9.5 論文主模型：agg256 + trunk512（464M）

```bash
python train_sliced.py --base_ckpt runs/distill_b1/best.pt \
    --q_ckpt "runs/sliced/sliced_agg256+pe512+ch1024.pt" --k 256 --k_ch 512 \
    --data_root <DATA_ROOT> --out_dir runs/sliced/recover_agg256_ch512 \
    --epochs 6 --img_per_seq 2 --eval_every 2 --throttle 0.05
```

這是實際執行過的指令（來源：distill2/README.md 五之三），這裡只多寫了 `--base_ckpt` 和 `--out_dir`，值都和預設相同。原始 run 的 `best.pt` 就是本 repo 的 `weights/vggt_sliced_464M_agg256_ch512.pt`，不想重訓可以直接用。

| 項目 | 紀錄值 | 來源 |
|---|---|---|
| best | epoch=5（第 6 個 epoch） | distill2/README.md 五之三；權重檔的 `epoch` 欄位 |
| rot_err vs teacher | 5.20°（前 8 個 test 場景） | 同上；reports/slicegpt_slice_combo.md |
| 逐幀 RRA@15 vs teacher | 0.969（前 8 個 test 場景；不是 §11 的成對 RRA，GT 成對 RRA@15 是 0.883） | 同上 |
| abs_rel | 0.193 | reports/quantize_sliced_report.md；權重檔的 `eval` 欄位 |
| 參數 | 464.16M；state_dict 464.43M（論文的 464.4M 是後者） | distill2/README.md 五之三 |
| 檔案 | 1,908,372,429 bytes；SHA256 `FE6E2CF735DCCB993560C98D21B2E3DC1357F2EE0186986CB6BFE7C7E8F1B352` | 同上；weights/SHA256SUMS.txt |

- 存檔內容：`model`、`meta`、`Q_agg`、`Q_pe`、`Q_ch`，以及 `k_agg=256`、`k_pe=0`、`k_ch=512`。
- 重新訓練**不會**得到相同的 SHA256，因為沒有固定 seed。這個雜湊只用來核對你拿到的是不是原始檔。
- 早期報告寫的「466M」是**同一個 checkpoint 的錯誤數字**，正確是 464.4M（state_dict）或 464.16M（參數）。本 repo 裡寫錯的地方是 reports/slicegpt_slice_combo.md、gtft_report.md、slicegpt_d8_combo.md 和 measure_vram_curve.py 的註解（來源：distill2/README.md 八），作者開發紀錄裡也有同樣的錯誤。

### 9.6 Table 4 下半的其他組合（含前端的負面結果）

除了 agg256+ch1024 那一行來自 distill2/README.md 五之二，其餘指令都是**重建**的：依 `train_sliced.py` 預設的 run 目錄命名 `recover_agg{k}_pe{k_pe}_ch{k_ch}` 和報告推回。共同參數：`--base_ckpt runs/distill_b1/best.pt --q_ckpt "runs/sliced/sliced_agg256+pe512+ch1024.pt" --data_root <DATA_ROOT> --img_per_seq 2 --throttle 0.05`。

| 組合 | 額外參數 | `--out_dir` | 恢復後 rot_err（RRA@15） | storage |
|---|---|---|---|---|
| agg256 + trunk1024 | `--k 256 --k_ch 1024` | `runs/sliced/recover_agg256_ch1024` | 5.36°（0.948） | 516M |
| agg256 + trunk1024 + fe512，凍結前端 | `--k 256 --k_pe 512 --k_ch 1024` | `runs/sliced/recover_agg256_pe512_ch1024` | 69.4°（0.177） | 365M |
| agg256 + trunk1024 + fe512，前端也訓練，8 epochs | 上一列再加 `--train_pe --epochs 8` | `runs/sliced/recover_triple_trainpe` | 77.2°（0.188），見下方說明 | 365M |
| agg256 + trunk512 + fe768，凍結前端 | `--k 256 --k_pe 768 --k_ch 512` | `runs/sliced/recover_agg256_pe768_ch512` | 48.1°（0.208） | 約 390M |

（來源：distill2/reports/slicegpt_slice_combo.md；論文 Table 4；全部是前 8 個 test 場景的結果）

- 原始 run 的 `last.pt` 證實了 epochs 數：`--train_pe` 那組是 8 epochs（epoch=7），其餘是 6 epochs（epoch=5）。存檔裡的 k 值也和上表相同（作者開發紀錄：原始 checkpoint 的 `epoch` 與 `k_*` 欄位）。
- `recover_triple_trainpe/best.pt` 存的 `eval` 是 75.26°（0.198，epoch 3），和報告與論文寫的 77.2°（0.188）不同（作者開發紀錄：原始 checkpoint 的 `eval` 欄位）。紀錄裡找不到 77.2° 是哪一次 eval 的值。兩個值都在塌縮區，結論不受影響（見 §16）。
- `--train_pe` 會讓前端約 304M 的參數也進入訓練，所需記憶體比凍結時多。

### 9.7 載入切片模型、計算參數量（Table 6）

切片後的結構**必須用 `load_sliced()` 載入**：它先依存檔裡的 Q 和 k 重建切片結構，再把權重灌進去。直接用 `VGGTStudent` 加 `load_state_dict` 會對不上（來源：distill2/README.md 五之三）。`eval.py` 和 `visualize.py` 用的是 `load_student`，**讀不了切片檔**。`quantize.load_any()` 會自動判斷該用哪一種（quantize.py 第 28–36 行），`eval_gt.py` 也是用它載入（eval_gt.py 第 161 行）。

```bash
# 重新評估任一 checkpoint（一般檔或切片檔都可以）；最後一個參數 8 是場景數
python -c "import torch; from quantize import load_any; from eval import quick_eval; m=load_any('../weights/vggt_sliced_464M_agg256_ch512.pt','cuda'); print(quick_eval(m,'<DATA_ROOT>/test',8,'cuda',torch.bfloat16))"

# Table 6：依元件統計 state_dict 元素數（用 mmap 讀，不會把權重全部載入記憶體）
python -c "import torch,collections; sd=torch.load('../weights/vggt_sliced_464M_agg256_ch512.pt',map_location='cpu',mmap=True,weights_only=False)['model']; g=collections.Counter(); [g.update({('aggregator.patch_embed' if k.startswith('aggregator.patch_embed') else 'aggregator(other)' if k.startswith('aggregator.') else k.split('.')[0]): v.numel()}) for k,v in sd.items()]; print({k: round(v/1e6,2) for k,v in g.items()}, round(sum(g.values())/1e6,2))"
```

用 842M（`runs/distill_b1/best.pt`，即 `../weights/vggt_teacher_init_842M.pt`）和 464M（`../weights/vggt_sliced_464M_agg256_ch512.pt`）跑第二段指令（2026-10-17 實測）：

| 元件 | 842M | 464M | 論文 Table 6（T.-init / Sliced） |
|---|---|---|---|
| 前端（`aggregator.patch_embed`） | 304.37M | 304.63M | 304.4M / 304.6M（含 0.26M adapter） |
| aggregator 其餘部分 | 302.37M | 75.96M | 302.4M / 76.0M |
| camera_head | 216.17M | 64.59M | 216.2M / 64.6M |
| depth_head | 9.62M | 9.62M | 9.6M / 9.6M |
| point_head | 9.62M | 9.62M | 9.6M / 9.6M |
| 合計 | 842.16M | 464.43M | 842.2M / 464.4M |

---

## 10. INT8 weight-only（§4.7）

```bash
python quantize.py --ckpt runs/sliced/recover_agg256_ch512/best.pt \
    --data_root <DATA_ROOT>/test --num_scenes 8 \
    --out runs/sliced/quantized_sliced_int8_rerun.pt --report reports/quantize_sliced_report_rerun.md
# 用 repo 裡的權重：--ckpt ../weights/vggt_sliced_464M_agg256_ch512.pt
```

**做法**
- 需要 GPU 和 torchao。torchao 不能用時，會退回 PyTorch 的 CPU 動態量化，只量化 `nn.Linear`（quantize.py 第 67–89、126–132 行）。
- 不給 `--out` 和 `--report` 的話，結果會寫到目前目錄的 `quantized_int8.pt` 和 `quantize_report.md`，**蓋掉 Path B 的 INT8 結果**（來源：distill2/README.md 五之三）。
- 量化規格：torchao 0.17 的 `Int8WeightOnlyConfig`，對稱、per-output-channel、以權重本身的絕對最大值做校準，不使用 activation 資料。所有 `nn.Linear` 都量化，包括凍結的前端；卷積、normalization、bias、token、de-slice 矩陣不量化；activation 維持 bf16（來源：論文 §4.7）。

| | 大小 | rot_err | abs_rel | VRAM | latency |
|---|---|---|---|---|---|
| 量化前 | 1858 MB | 5.20° | 0.193 | 3.89 GB | 223.9 ms |
| torchao int8 weight-only | 537 MB | 5.21° | 0.193 | 3.68 GB | 248.8 ms |

（來源：distill2/reports/quantize_sliced_report.md；前 8 個場景、S=4）

- **原始 INT8 檔就在本 repo**：`weights/vggt_sliced_464M_int8.pt`（537,215,008 bytes，原始路徑 `runs/sliced/quantized_sliced_int8.pt`）。它是 `torch.save({"model": state_dict, "quant": 方法名稱})` 存的（quantize.py 第 145 行），載入時先用 `load_sliced()` 建出 fp32 切片結構，再用 torchao 的 `quantize_(m, Int8WeightOnlyConfig())` 轉成同樣的量化結構，最後 `load_state_dict(ck["model"])`（完整程式見 [weights/README.md](../weights/README.md)）。
- 537 MB 是 `torch.save` 寫出 state_dict 的實際 bytes 數，不是理論上的權重 bytes（來源：論文 §4.7；quantize.py 的 `size_mb()`，以 10^6 bytes 為 1 MB）。teacher fp32 5025 MB ÷ 537 MB ≈ 9.4×（論文 §4.7）。
- 這份報告的 VRAM 3.68 GB **不是** INT8 模型單獨的峰值。`quantize.py` 量量化後的模型時，量化前的基準模型還留在 GPU 上，程式沒有釋放它（quantize.py 第 115、127、142 行）。粗算 3.68 ≈ 1.81 + 1.86（推算：1.86 GB 是 fp32 基準模型的權重）。論文 Table 5 的 1.81 GB 來自 `measure_unified_bench.py`，那裡是單獨載入的（§12）。
- INT8 比 bf16 慢（248.8 vs 223.9 ms；Table 5 是 229.9 vs 201.3 ms）。原因是 torch 2.10.0+cu128 沒有 fused INT8 kernel，權重是在執行時即時反量化（來源：論文 §4.7、Table 5 註腳 ‡）。
- 論文說量化後沒有重量 GT 指標和 depth 指標（論文 §4.7）；紀錄裡其實有 INT8 對 teacher 的 abs_rel（0.193，上表，前 8 個場景），但沒有任何對 GT 的指標，INT8 的 GT 準確率是未知。目前的 `eval_gt.py` 也不能直接拿 INT8 檔當 `--ckpt`：`load_any()` 認不得只有 `model`、`quant` 兩個鍵的格式。要評估的話，得先照上面的方式載入 INT8 模型，再自己呼叫 eval_gt.py 裡的函式。
- Path B（842M）的 INT8 結果（不在論文中）：
  - 指令：`python quantize.py --ckpt runs/distill_b1/best.pt --data_root <DATA_ROOT>/test --num_scenes 8 --report reports/quantize_report_rerun.md`（`--ckpt` 也可以用 `../weights/vggt_teacher_init_842M.pt`；量化檔會寫到目前目錄的 `quantized_int8.pt`，不在 repo 裡）
  - 紀錄：3369 MB → 915 MB（3.68×），3.42° → 3.42°（來源：distill2/reports/quantize_report.md）

---

## 11. CO3D ground-truth 評估（Table 2）

### 11.1 每場景 10 個等距幀（論文用這組）

```bash
python eval_gt.py --ckpt runs/distill_b1/best.pt --data_root <DATA_ROOT>/test \
    --num_scenes 65 --frames 10 --ref co3d --co3d_root <CO3D_ROOT> --out_prefix reports/eval_gt_co3d10_pathb_rerun
python eval_gt.py --ckpt runs/sliced/recover_k384/best.pt --data_root <DATA_ROOT>/test \
    --num_scenes 65 --frames 10 --ref co3d --co3d_root <CO3D_ROOT> --out_prefix reports/eval_gt_co3d10_k384_rerun
python eval_gt.py --ckpt runs/sliced/recover_agg256_ch512/best.pt --data_root <DATA_ROOT>/test \
    --num_scenes 65 --frames 10 --ref co3d --co3d_root <CO3D_ROOT> --out_prefix reports/eval_gt_co3d10_ch512_rerun
```

- 第三條指令（ch512）在 distill2/README.md 五之三有原文。另外兩條是依報告標頭「參考：co3d｜場景 65｜每場景 10 幀」和 argparse **重建**的。
- 842M 那一列用的是 `runs/distill_b1`（來源：distill2/README.md 四）。
- 第一條和第三條的 `--ckpt` 可以換成 `../weights/vggt_teacher_init_842M.pt` 和 `../weights/vggt_sliced_464M_agg256_ch512.pt`。agg384（`recover_k384`）不在 repo 裡，要先跑 §9.3。

**`eval_gt.py` 的行為**
- 直接讀 `<CO3D_ROOT>/<category>/frame_annotations.jgz`，**不需要**先跑 `make_gt_labels.py`。
- 用 `flip_xy` 慣例把 PyTorch3D 相機轉成 OpenCV 相機。
- teacher 那一列取自 soft label 的 `pose_enc.npy`，**不會**重跑 teacher 模型。
- 找不到標註的場景會被略過，並計入 skip（來源：distill2/eval_gt.py）。

| 模型 | 參數 | RRA@15 | RRA@30 | RTA@15 | RTA@30 | AUC@30 |
|---|---|---|---|---|---|---|
| VGGT-1B teacher | 1256M | 1.000 | 1.000 | 0.930 | 0.957 | 0.976 |
| Teacher-initialized | 842M | 0.964 | 0.981 | 0.893 | 0.941 | 0.876 |
| Sliced, agg384 | 654M | 0.905 | 0.939 | 0.826 | 0.910 | 0.792 |
| Sliced, agg256+trunk512 | 464M | 0.883 | 0.940 | 0.814 | 0.908 | 0.767 |

（來源：distill2/reports/eval_gt_co3d10_{pathb,k384,ch512}.md；與論文 Table 2 一致）

- AUC@30 只算旋轉，是 RRA 曲線在 1–30° 之間的面積，不能和同時考慮旋轉與平移的文獻數字比較（來源：論文 §4.1；distill2/eval_gt.py 第 65–70 行）。
- 本文的 split 比文獻常用的協定（41 類別、隨機 10 幀）容易：teacher 的 AUC@30 在這裡是 0.976，文獻是 0.882（來源：論文 §4.1）。
- 較早的 20 場景 × 8 幀版本（distill2/reports/eval_gt_co3d_{pathb,k384,ch512}.md）是初步結果，論文不引用（作者開發紀錄）。

### 11.2 隨機取幀的檢查

在上面三條指令後面加 `--random_frames 0`，並把 `--out_prefix` 改成 `reports/eval_gt_co3d10rand_{pathb,k384,ch512}_rerun`，改用隨機 10 幀（seed 0，作者開發紀錄）：

| 模型 | RRA@15 | AUC@30 | teacher（同協定） |
|---|---|---|---|
| 842M | 0.965 | 0.882 | 1.000 / 0.979 |
| agg384 | 0.923 | 0.813 | 1.000 / 0.979 |
| 464M | 0.875 | 0.762 | 1.000 / 0.979 |

（來源：distill2/reports/eval_gt_co3d10rand_{pathb,k384,ch512}.md）

和等距取幀相比，RRA@15 最多差 0.018，AUC@30 最多差 0.021（來源：論文 §4.1）。

### 11.3 直接用 GT 微調（選用，§4.3）

```bash
# 1) 為每個場景寫入 gt_extrinsics.npy（flip_xy；492 個場景約數分鐘）
python make_gt_labels.py --data_root <DATA_ROOT> --co3d_root <CO3D_ROOT>
# 預期每個 split 印一行 "[train] 已產生 N，skip M"；原始資料 492 個場景全部覆蓋

# 2) 842M：從 best 接著訓練 4 epochs、lr 1e-5，pose 目標改用 GT（重建）
python train.py --student_init runs/distill_b1/best.pt --data_root <DATA_ROOT> \
    --out_dir runs/gtft/pathb --epochs 4 --lr 1e-5 --img_per_seq 2 --gt_pose 1

# 3) 464M：用 --resume_sliced 接著訓練（重建）。--k 是必填參數；
#    存檔時寫入的 k 值取自命令列，所以 --k 256 --k_ch 512 一定要和原模型一致
python train_sliced.py --resume_sliced runs/sliced/recover_agg256_ch512/best.pt --k 256 --k_ch 512 \
    --data_root <DATA_ROOT> --out_dir runs/gtft/sliced --epochs 4 --lr 1e-5 --img_per_seq 2 --gt_pose 1
```

- 步驟 2、3 的起點可以分別換成 `../weights/vggt_teacher_init_842M.pt` 和 `../weights/vggt_sliced_464M_agg256_ch512.pt`。GT 微調後的模型不在 repo 裡。
- 步驟 1 的說明來自 distill2/make_gt_labels.py docstring 和 distill2/reports/gtft_report.md。
- 重建的依據：gtft_report.md 記錄「4 epochs、lr 1e-5、從各自最佳權重續訓」；原始兩個 run 的 `last.pt` 都是 epoch=3；`runs/gtft/sliced` 存檔的 k 值是 (256, 0, 512)（作者開發紀錄：原始 checkpoint 的欄位）。`--steps_per_epoch` 沒有紀錄，上面的指令用的是預設值。
- `--resume_sliced` 時 `--k` 必填（train_sliced.py 第 36 行），而存檔的 k 值取自命令列參數（第 151–155 行）。

結果：
- 842M：RRA@15 / RTA@15 / AUC@30 從 0.964 / 0.893 / 0.876 變成 0.960 / 0.898 / 0.856
- 464M：從 0.883 / 0.814 / 0.767 變成 0.883 / 0.821 / 0.770
- 沒有實質增益。評估用的是 `runs/gtft/{pathb,sliced}/last.pt`（來源：distill2/reports/gtft_report.md）
- 論文的說法：RRA@15 最多變動 0.004、AUC@30 最多變動 0.020，正負號不一致（論文 §4.3）

---

## 12. 效率、VRAM、記憶體預算、FLOPs（Table 5、Table 7、Fig. 2、App. D）

**前置條件**
- 量測腳本讀的是寫死的 checkpoint 路徑：`runs/distill_b1/best.pt`、`runs/prune/d{8,6,4}/best.pt`、`runs/sliced/recover_agg256_ch512/best.pt`。
  - 842M 和 464M 可以用 repo 的權重：把 `../weights/vggt_teacher_init_842M.pt` 複製成 `runs/distill_b1/best.pt`，把 `../weights/vggt_sliced_464M_agg256_ch512.pt` 複製成 `runs/sliced/recover_agg256_ch512/best.pt`（或直接改腳本裡的路徑）。
  - 剪枝模型 `runs/prune/d{8,6,4}/best.pt` 不在 repo 裡，要先跑 §7。需要它們的是 `measure_unified_bench.py`（Table 3、Table 5 的剪枝列）和 `measure_flops_extra.py`（d8 的 GFLOPs）；`measure_vram_curve.py`、`measure_vram_int8.py`、`measure_budget_incl_context.py` 只用到 842M 和 464M。
  - INT8 模型由腳本在執行時對 464M 模型自己量化，不讀 `weights/vggt_sliced_464M_int8.pt`（measure_unified_bench.py 第 29–32 行；measure_vram_int8.py 第 27–28 行；measure_budget_incl_context.py 第 42–44 行）。
- 要能下載 teacher，並裝好 torchao。
- `measure_unified_bench.py`、`measure_vram_curve.py`、`measure_vram_int8.py`、`measure_flops_extra.py`、`measure_budget_incl_context.py` 的 checkpoint 和輸出 JSON 路徑都**寫死成 Windows 反斜線**（例如 `r"runs\distill_b1\best.pt"`、`r"reports\unified_bench.json"`），在 Linux 上要先改成正斜線。這些腳本也會**直接改寫** `reports/` 裡的原始 JSON，執行前請先備份。`measure_efficiency.py` 的路徑則是參數。
- 量測時請關掉其他用到 GPU 的程式。跨 session 的延遲差異可達約 15%（見 §16），只比較同一個 session 量出來的數字。

```bash
python measure_unified_bench.py      # Table 5 / Table 3 的 latency 與 VRAM → reports/unified_bench.json
python measure_vram_curve.py         # Fig. 2 與 Max frames → reports/vram_curve.json
python measure_vram_int8.py          # INT8 的 Max frames → reports/vram_curve_int8.json
python measure_flops_extra.py        # d8 與 scratch conv-512 的 GFLOPs → reports/flops_extra.json
# Table 7：第一個參數是模型清單，第二個是模式（paper = 參數維持預設 trainable 狀態，frozen = requires_grad=False）
python measure_budget_incl_context.py sliced_int8,sliced,pathb,teacher paper    # → reports/budget_incl_context_paper.json
python measure_budget_incl_context.py sliced_int8,sliced,pathb,teacher frozen   # → reports/budget_incl_context_frozen.json（含 GFLOPs 與 CPU 延遲）
# Table 5 的 † 列（scratch student）與舊的效率表；FLOPs 用 fvcore 的 GMACs，和上面的 GFLOPs 不是同一種量
python measure_efficiency.py --frames 4 --img_size 518 --runs 20 --pathb_ckpt runs/distill_b1/best.pt --out_prefix reports/efficiency_report_rerun
```

### Table 5

| 模型 | 參數 | GFLOPs | VRAM（S=4） | latency（S=4） | Max frames（≤16 GB） |
|---|---|---|---|---|---|
| VGGT-1B teacher | 1256.5M | 16814.3 | 9.21 GB | 388.1 ms | 24 |
| Scratch student（conv, 512）† | 146.4M | 2399.8 | 1.76 GB | 123.8 ms | — |
| Teacher-initialized | 842.2M | 9919.2 | 5.85 GB | 219.1 ms | 48 |
| ├ pruned to 8 blocks | 741.4M | 8194.1 | 5.07 GB | 180.1 ms | — |
| Sliced（agg256 + trunk512） | 464M | 7496.8 | 3.89 GB | 201.3 ms | 64 |
| └ + INT8 weight-only | 464M | 7496.8 | 1.81 GB | 229.9 ms | 80 |

（來源：latency 和 VRAM 取自 reports/unified_bench.json；GFLOPs 取自 reports/budget_incl_context_frozen.json 的 `gflops_S4` 和 reports/flops_extra.json；† 列的 latency 和 VRAM 取自 reports/efficiency_report.md；Max frames 取自 reports/vram_curve.json 和 vram_curve_int8.json；與論文 Table 5 一致）

**怎麼讀 Max frames**
- 判定標準是峰值 ≤ 16 GB。teacher 在 S=24 時是 16.00 GB，S=28 是 17.08 GB；842M 在 S=48 是 13.35 GB，S=64 是 16.27 GB；464M 在 S=64 是 13.92 GB，S=80 是 16.72 GB；INT8 在 S=80 是 14.65 GB，S=96 是 17.47 GB（來源：reports/vram_curve*.json）。
- JSON 裡的 `max_frames` 和 `max_frames_alloc` 欄位，記的是「沒有丟出 OOM 的最大 S」（64 / 96 / 96 / 96），裡面包含了 Windows 共享記憶體溢出才跑得起來的情況，**不是**論文的數字（§15.4）。
- `measure_vram_int8.py` 結尾會直接印出「16GB 預算內最大 S」。

**怎麼讀 VRAM**
- VRAM 是 `torch.no_grad()` 下的峰值 allocated memory，不含 CUDA context（來源：論文 Table 5 caption、Fig. 2 caption）。
- 參數維持預設的 trainable 狀態，因為這幾支腳本都沒有呼叫 `requires_grad_(False)`。
- frozen 狀態下，除了 INT8 模型以外，峰值都會更低（來源：論文 Table 7 caption；measure_budget_incl_context.py 第 47–49 行）。

### Table 7（模擬預算，含約 303 MiB 的 CUDA context）

| 模型 | Default 4 / 6 / 8 / 12 GiB | Frozen 4 / 6 / 8 / 12 GiB |
|---|---|---|
| teacher | 0 / 0 / 1 / 8 | 0 / 1 / 4 / 12 |
| Teacher-initialized | 0 / 4 / 12 / 32 | 1 / 4 / 20 / 48 |
| Sliced | 2 / 8 / 24 / 48 | 4 / 16 / 32 / 48 |
| Sliced + INT8 | 8 / 20 / 32 / 56 | 8 / 20 / 32 / 56 |

（來源：reports/budget_incl_context_paper.json 對應 Default 欄，budget_incl_context_frozen.json 對應 Frozen 欄；`context_mib` = 303.0；與論文 Table 7 一致）

- JSON 裡另有 16 GiB 的欄位，論文刻意不列（論文 App. D：超過卡上實際可用的記憶體）。
- 腳本開始前會先檢查其他程式佔用的 GPU 記憶體，超過 4500 MiB 就中止。原始量測時其他程式佔了 1271–1316 MiB（來源：measure_budget_incl_context.py 第 82–85 行；兩個 JSON 的 `other_processes_mib`）。
- CPU 延遲只在 frozen 模式、而且命令列沒有加 `nocpu` 時才量。條件是 fp32、S=2，每個模型先暖機 1 次再量 3 次：teacher 16233.7 ms、sliced 7403.1 ms，約 2.19×（來源：budget_incl_context_frozen.json `cpu_latency`；measure_budget_incl_context.py 第 132–150 行；論文 App. D）。
- 儲存大小：teacher fp32 5025 MB（論文 §1、§4.7；用同一算法推算的值見 §16）、sliced fp32 1858 MB、INT8 537 MB（reports/quantize_sliced_report.md）。論文沒有給 842M 的儲存大小；紀錄中是 3369 MB（reports/quantize_report.md）。

**剪枝模型的效率表**（不在論文中，重建）：

```bash
python measure_efficiency.py --no_teacher --pathb_ckpt runs/distill_b1/best.pt \
    --extra_ckpts "PathB-d8 (1024, 8層)=runs/prune/d8/best.pt;PathB-d6 (1024, 6層)=runs/prune/d6/best.pt;PathB-d4 (1024, 4層)=runs/prune/d4/best.pt" \
    --out_prefix reports/efficiency_prune_rerun
```

標籤裡有逗號時，項目之間要用 `;` 分隔（measure_efficiency.py 第 105–110 行）。紀錄結果見 reports/efficiency_prune.md。

---

## 13. 視覺化

```bash
python visualize.py --ckpt runs/distill_b1/best.pt --data_root <DATA_ROOT>/test \
    --collapsed_ckpt runs/abl/random/best.pt --out_dir viz
```

- 會在 `viz/` 產生 5 張圖：`rotation_histogram.png`、`camera_trajectory.png`、`depth_comparison.png`、`pointcloud.png`、`success_failure.png`（來源：distill2/README.md 五；distill2/visualize.py；distill2/viz/ 裡的原始圖）。
- `--ckpt` 可以換成 `../weights/vggt_teacher_init_842M.pt`。`--collapsed_ckpt`（塌縮模型的對照）是選填；`runs/abl/random/best.pt` 不在 repo 裡，要先跑 §6，或者不加這個參數。
- `--num_scenes` 預設 12，用在直方圖和 success/failure 圖。
- `visualize.py` 用 `load_student` 載入，所以**不能**直接畫切片模型。
- 論文裡沒有任何定性圖（點雲、深度圖、相機軌跡）；論文只有兩張圖：Fig. 1 流程總覽、Fig. 2 峰值記憶體曲線。

---

## 14. 論文以外的紀錄實驗（選用）

**剪枝 × 切片疊加**
- 論文把「層剪枝加寬度切片」列為未來工作（論文 §5 Limitations），但紀錄中其實做過一次（來源：distill2/reports/slicegpt_d8_combo.md）。
- 重建的指令（依 run 目錄 `runs/sliced_d8/` 和檔名 `sliced_agg256+ch512.pt` 推回）：
  ```bash
  python slicegpt_slice.py --run --ckpt runs/prune/d8/best.pt --data_root <DATA_ROOT> \
      --ks 256 --k_ch 512 --bench --save_k 256 --save_dir runs/sliced_d8 --out reports/slicegpt_d8_combo_rerun.md
  python train_sliced.py --base_ckpt runs/prune/d8/best.pt --q_ckpt "runs/sliced_d8/sliced_agg256+ch512.pt" \
      --k 256 --k_ch 512 --data_root <DATA_ROOT> --out_dir runs/sliced_d8/recover --img_per_seq 2 --throttle 0.05
  ```
  `--base_ckpt` 一定要指向 d8 的 checkpoint（不在 repo 裡，要先跑 §7）。
- 紀錄結果：
  - d8 原模型在前 12 個場景是 9.06°（和 Table 3 的 9.45° 是不同子集）
  - 零重訓 52.79°
  - 恢復後 9.08°（前 8 個場景，和 Table 3 的 9.45° 同一子集），439M，178 ms
  - GT 評估 RRA@15 0.645、AUC@30 0.564

**二次蒸餾、投影初始化等「先變窄再訓練」的嘗試**
- 結果全部塌縮：DINOv2-S 從零訓練約 82°、camera-token feature KD 約 75°、跨幀結構 KD 約 77°、SVD 投影初始化約 77°（作者開發紀錄；各實驗的細節見 [RESULTS.md](RESULTS.md) §9.2–9.4）。
- 指令：
  - `python train2.py --teacher_ckpt runs/distill_b1/best.pt --data_root <DATA_ROOT> --out_dir runs/distill2_s --img_per_seq 2 --feature_weight 3 --struct_weight 4`（distill2/README.md 五）
  - `python proj_init.py --teacher_ckpt runs/distill_b1/best.pt --out proj_student.pt`，接著用 `train.py` 訓練（distill2/proj_init.py docstring）
  - 兩條指令的 `--teacher_ckpt` 都可以用 `../weights/vggt_teacher_init_842M.pt`。
- 紀錄的結論是：要拿到較窄的模型，一律走「事後切片＋恢復蒸餾」，不要再試「先變窄再訓練」（作者開發紀錄）。

**舊版評估腳本** `scripts/eval_student_vs_teacher.py`：這是舊 Hydra 管線的評估腳本，已修好架構偵測（distill2/CHANGES.md §3(C)）。重現 distill2 的結果用不到它。

---

## 15. 疑難排解

### 15.1 相機姿態塌縮（camera-token collapse）

**症狀**
- quick_eval 的 rot_err 停在 75–82° 左右，RRA@15 約 0.1–0.2，但 abs_rel 正常（塌縮的消融組合是 0.235–0.265）（來源：reports/ablation_report.md）。
- 模型內部：各幀預測的四元數幾乎一樣（跨幀標準差約 0.01，teacher 約 0.5）；camera token 的跨幀 cosine 是 0.998；`camera_token` 參數幾乎沒離開初始值（來源：論文 §3.2；distill2/CHANGES.md §2）。

**紀錄中會導致塌縮的情況**

| 情況 | 紀錄的結果 | 來源 |
|---|---|---|
| 沒有同時複製前端、aggregator、camera head | 75–76° | reports/ablation_report.md |
| 三者都複製，但前端也拿去訓練（`--freeze_patch_embed 0`） | 75.68° | 同上 |
| 寬度變窄或從零訓練（conv-512、DINOv2-S、二次蒸餾、投影初始化） | 約 75–82° | 論文 §3.2（conv-512 的 82.3°）；作者開發紀錄（其餘） |
| aggregator 少於 8 層 | d6 56.78°、d4 75.74° | reports/layer_pruning_report.md |
| 切到前端那條流（k_pe 為 512 或 768，凍結或訓練都一樣） | 48–77° | reports/slicegpt_slice_combo.md |
| 四元數順序弄錯（VGGT 是 scalar-last `(x,y,z,w)`，曾誤用 scalar-first） | 相對旋轉項的梯度變成雜訊 | distill2/CHANGES.md §3(A)；論文 App. B |

`distill2/loss.py` 已經修好四元數順序和 EMA buffer 的 device 問題（loss.py 第 8 行、第 92–93 行），請不要改回去。`train.py` 和 `train_sliced.py` 預設都凍結前端，請保持預設。

**82° 的 scratch student 無法用本 repo 重跑**：論文 §3.2 的 82.3°、RRA@15 0.098 是 146.4M scratch student 的結果，用舊腳本在 10 個場景上評估，早於 distill2 管線（來源：論文 §3.2；數值也見 distill2/reports/eval_results_before.json）。在目前管線裡對應的塌縮參考是 Table 1 的 random 列（76.18°）（來源：論文 §3.2）。distill2/CHANGES.md §4 的「起點 82.7°、RRA@15 0.135」是另一份舊評估的數字。

### 15.2 載到錯的 vggt

**症狀**
- `load_student` 印出 `[警告] missing/unexpected`。
- 旋轉評估異常變差（舊版會把 camera head 截成 2 層）。
- 或者 `import vggt.models.vggt_student` 失敗。

**處理**：照 §3.3 檢查 `vggt.__file__`，並確認 `trunk_depth=4` 那行輸出 `True`。不要在任何腳本裡把別的 `vggt` 所在目錄插到 `sys.path` 最前面（來源：distill2/CHANGES.md §3(C)；作者開發紀錄）。

### 15.3 OOM 與記憶體

- **訓練**：用 `--img_per_seq 2`（distill2/README.md 四；作者開發紀錄）。`train.py` 的預設是 4，`train_sliced.py`、`run_ablation.py`、`run_layer_pruning.py` 的預設是 2。
- **`--train_pe`** 會讓前端約 304M 的參數也進入訓練，需要更多記憶體。
- **評估**：`eval.py` 每個場景最多取 12 幀，就是為了避免大模型整段 50 幀 OOM（eval.py 第 73 行註解）。
- **匯出資料**：用 `--max_images_per_scene` 控制每次送進 teacher 的張數（§4.2）。`export_teacher_labels.py` 不抽樣，影像太多就會 OOM。
- `--throttle` **不會**降低記憶體用量，它只是在每一步之後多睡一下（§15.7）。

### 15.4 Windows 上超過 16 GB 時的共享記憶體溢出

- 在原始的 Windows 機器上，峰值超過 16 GB 實體 VRAM 時並不會丟出 OOM，而是溢出到 Windows 共享記憶體繼續跑，速度掉幾十倍，實務上不能用（作者開發紀錄）。
- 這就是為什麼 `reports/vram_curve.json` 裡會出現 16 GB 卡跑出 27.19 GB（teacher，S=64）這種值。
- 因此：
  - Max frames 一律用「峰值 ≤ 16 GB」判定，不要用 JSON 的 `max_frames` 欄位（§12）。
  - 如果訓練或推論突然慢了好幾十倍，先檢查是不是發生了同樣的溢出。紀錄中前端也一起訓練的 `full_ft_pe` 那組花了約 7.6 小時，其他組約 50–60 分鐘（推算，見 §2）；原因沒有紀錄。

### 15.5 quick_eval 預設只評前 8 個場景

`quick_eval()` 取的是排序後 test 場景的前 `num_scenes` 個（eval.py 第 102–107 行）。各腳本的預設不同：

| 用途 | 腳本與參數 | 場景數 |
|---|---|---|
| 訓練中 eval，存進 `best.pt` 的 `eval` 欄位 | `train.py`、`train_sliced.py`、`run_ablation.py`、`run_layer_pruning.py` 的 `--eval_scenes` | 8 |
| 量化前後比較 | `quantize.py --num_scenes`（預設 8，README 也用 8） | 8 |
| 探針與零重訓切片 | `slicegpt_probe.py`、`slicegpt_slice.py` 的 `--eval_scenes` | 12 |
| `eval.py` | `--num_scenes` 預設 | 10 |
| `eval_gt.py` | 預設 20 個場景 × 8 幀；Table 2 用 65 × 10 | — |

**和論文 Table 4 的關係**（來源：distill2/reports/slicegpt_slice.md、slicegpt_slice_combo.md、quantize_sliced_report.md；distill2/train_sliced.py 第 49 行）
- 論文說 Table 4 全部用前 12 個 test 場景（7 個類別）（論文 §4.1、Table 4 caption）。
- 紀錄顯示，只有探針、零重訓和 2.87° 是 12 個場景。
- 恢復後的值（4.18、4.95、5.36、5.20、69.4、77.2、48.1）和 INT8 的 5.20° → 5.21° 都是**前 8 個場景**。前 8 個場景只有 4 個類別（apple、backpack、ball、banana）（推算：distill2/reports/eval_results_before.json 依序列出的 test 場景名稱，假設它用的是同一個排序過的 test split）。

所以要重現 Table 4 的恢復值，請維持預設的 8。如果改用 12 個場景重新評估，數字會不同。例如同一個 842M checkpoint，12 個場景是 2.87°，8 個場景是 3.42°。

### 15.6 Windows 上子行程因 cp950 編碼錯誤死掉

在 Windows 上，子行程的輸出被重導向時會退回 cp950 編碼，印 `✓` 之類的字元時會丟出 `UnicodeEncodeError`，訓練中途就死了。`run_layer_pruning.py` 已經替子行程設了 `PYTHONUTF8=1`（第 34–37 行），`run_ablation.py` 沒有。所以跑消融前請先 `export PYTHONUTF8=1`；PowerShell 用 `$env:PYTHONUTF8="1"`。

### 15.7 GPU 滿載時螢幕黑屏

- 加 `--throttle 0.05`。它會在每一步之後 `torch.cuda.synchronize()` 再 sleep 指定的秒數，讓出時間給顯示驅動（train.py 第 52–53、113–117 行）。help 建議 0.05–0.15。代價是每一步多 0.05 秒，4800 步（12 epochs × 400 steps）約多 4 分鐘（推算）。
- 紀錄中的另一個做法：用系統管理員權限執行 `nvidia-smi -pl 288`，把功耗上限壓到 80%（作者開發紀錄）。這會改變系統的 GPU 設定，請自行判斷是否要用。

### 15.8 切片 checkpoint 的載入與檔名

- 切片檔一律用 `slicegpt_slice.load_sliced()` 或 `quantize.load_any()` 載入（§9.7）。INT8 檔的載入方式不同，見 §10 和 [weights/README.md](../weights/README.md)。
- `--save_k` 存出的檔名是 `sliced_agg{k}{+pe..}{+ch..}.pt`。`sliced_k512.pt` 是舊檔名（§9.2）。
- 原始的 `recover_k384`、`recover_k256`（不在 repo）是舊格式，切片資訊只存在 `Q`、`k` 兩個鍵（沒有 `Q_agg`/`k_agg`），`apply_slices_()` 也能處理（slicegpt_slice.py 第 355–366 行）；用目前的 `train_sliced.py` 重跑 §9.3 會存成新格式。

### 15.9 在 Linux 上執行

- 有些腳本的預設路徑是反斜線寫法，有些寫死在程式裡：`measure_*.py` 的 checkpoint 和 JSON 路徑，`train_sliced.py`、`slicegpt_*.py` 的 `--base_ckpt`、`--q_ckpt`、`--out_dir`、`--save_dir`、`--out` 預設值。
- 能用參數指定的，請照本文的指令全部寫明；寫死的要改程式（來源：distill2/README.md 五之三）。
- 這些腳本只在 Windows 上實際跑過（作者開發紀錄）。

### 15.10 GT 評估的 teacher 分數異常低

如果 teacher 對 GT 的 RRA@15 低於 0.3，`eval_gt.py` 會提示座標轉換可能不對，請改試 `--co3d_convention flip_yz` 或 `none`。正確的慣例 `flip_xy` 已經用「teacher RRA@15 = 1.000」驗證過（來源：distill2/eval_gt.py 第 209–210 行；distill2/make_gt_labels.py docstring）。如果出現「沒有可評估的場景」，請確認 `<CO3D_ROOT>/<category>/frame_annotations.jgz` 存在，而且場景資料夾名稱是 `<category>__<sequence>`。

---

## 16. 論文與紀錄的差異（彙整，2026-10-17 核對）

| # | 項目 | 論文 | 紀錄 | 來源 |
|---|---|---|---|---|
| 1 | 842M 主模型的訓練長度 | §4.1："20 epochs of 500 steps" | `runs/distill_b1`（論文所有下游結果的基底，即 `weights/vggt_teacher_init_842M.pt`）是 40 epochs × 500 steps（`best.pt` epoch=37、`last.pt` epoch=39）。20-epoch 的是另一個 run `runs/distill_b1_full`，沒有用在論文結果裡 | 權重檔的 `epoch` 欄位；`last.pt` 見作者開發紀錄；train.py 第 41–42 行；distill2/README.md 四 |
| 2 | 主線每段序列的幀數 | §4.1："two frames per sampled sequence" | README 把 distill_b1 記為 train.py 預設（`--img_per_seq 4`），但沒有可核對的紀錄。作者開發紀錄裡那條較早的 distill_b1 指令寫的是 `--img_per_seq 2`（和論文一致），但同一條指令的 `--epochs 20` 和 checkpoint 不符，所以也不能當成實際設定。`distill_b1_full` 確定是 2 | distill2/README.md 四；train.py 第 43 行 |
| 3 | Table 4 的評估子集 | 全部是前 12 個 test 場景（§4.1、Table 4 caption） | 只有探針、零重訓和 2.87° 是 12 個；恢復值（4.18、4.95、5.36、5.20、69.4、77.2、48.1）和 INT8 的 5.20 → 5.21 是前 8 個。Tables 1、3 也是 8 個（論文沒有寫場景數）。所以論文 §4.5 拿 4.95°（8 個場景）和探針的 4.91°（12 個場景）相比，是跨了兩個子集 | reports/slicegpt_slice.md、slicegpt_slice_combo.md、quantize_sliced_report.md；train_sliced.py、run_ablation.py、run_layer_pruning.py 的 `--eval_scenes 8` |
| 4 | 主模型參數量 | 464M / 464.4M | 一致。早期報告寫的 466M 是同一個 checkpoint 的錯誤數字；正確是 464.4M（state_dict）或 464.16M（參數） | distill2/README.md 五之三、八 |
| 5 | 跨 session 的延遲漂移 | §4.7："within about 7%" | 最多約 15–18%（依計算方向）：scratch student 123.8 ms vs 105.2 ms（推算：慢的比快的多 17.7%，快的比慢的少 15.0%）；sliced 223.9 ms vs 201.3 ms（推算：11.2% / 10.1%） | reports/efficiency_report.md、efficiency_prune.md、unified_bench.json、quantize_sliced_report.md |
| 6 | Table 5 † 列 | "from an earlier, unrepeated batch" | scratch student 其實在另一個 session 重測過（105.2 ms，快約 15%） | reports/efficiency_prune.md |
| 7 | teacher fp32 大小 | 5025 MB（§1、§4.7） | 用和 537、1858 MB 相同的算法（`torch.save` state_dict 的 bytes，10^6 bytes = 1 MB），1256.5M 個 fp32 參數約 5026 MB（推算：1256.5M × 4 bytes）。差約 1 MB，不影響 9.4×（推算：5026 / 537 ≈ 9.36） | 論文 Table 6 的 1256.5M；quantize.py `size_mb()` |
| 8 | 前端也訓練的三流組合（8 epochs） | 77.2°（0.188） | 報告同樣是 77.2°，但 `recover_triple_trainpe/best.pt` 存的 `eval` 是 75.26°（0.198，epoch 3）；77.2° 的出處檔找不到。兩者都是塌縮 | reports/slicegpt_slice_combo.md；作者開發紀錄（原始 checkpoint 的 `eval` 欄位） |
| 9 | 剪枝 × 切片 | 列為未來工作（§5 Limitations） | 做過一次（d8 + agg256+ch512：9.08°、439M、GT RRA@15 0.645），沒有寫進論文 | reports/slicegpt_d8_combo.md |
| 10 | Table 1/3/4 的「RRA@15 vs teacher」 | 稱為 RRA@15 | 程式算的是逐幀（以第 0 幀為座標原點）旋轉誤差小於 15° 的比例，不是成對相對旋轉 | distill2/eval.py 第 88、98 行；distill2/README.md 五之三 |
| 11 | 摘要和 §4.7 的「teacher 最多 1 幀」 | "against at most one for the teacher" | 只在 Table 7 的 Default 欄成立；Frozen 欄的 teacher 在 6 GiB 能處理 1 幀、8 GiB 能處理 4 幀 | 論文 Table 7；reports/budget_incl_context_frozen.json |
| 12 | INT8 模型的 depth 指標 | §4.7：量化後沒有重量 GT 指標和 depth 指標 | 有 INT8 對 teacher 的 abs_rel：0.193（前 8 個場景，和量化前相同）；對 GT 的指標確實沒有，INT8 的 GT 準確率未知 | reports/quantize_sliced_report.md |
