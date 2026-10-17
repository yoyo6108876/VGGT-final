# 實驗結果總覽（docs/RESULTS.md）

本文件整理論文 *Compressing a 3D Foundation Model: Teacher-Initialized Distillation and Global Projection Slicing for Deployable VGGT*（Tzu-Chi Huang, Kuan-You Chen, Chung-Wei Lin, Chow-Sing Lin；ICS 2026，Springer CCIS，尚未出版）用到的每一個實驗數字，也收錄論文沒有放進去的其他結果。

寫法約定：

- 論文表格照 camera-ready 版印出的值轉錄，來源寫成「論文 Table N」或「論文 §x.y」。
- 紀錄中的數字附上本 repo 的檔案和行號，例如（來源：distill2/reports/ablation_report.md L7）。路徑都相對於 repo 根目錄，檔案在 [`distill2/reports/`](../distill2/reports/)、[`distill2/`](../distill2/) 的程式、[`distill2/README.md`](../distill2/README.md)、[`distill2/CHANGES.md`](../distill2/CHANGES.md) 與 [`scripts/`](../scripts/)。
- 只出現在作者私人開發筆記、本 repo 沒有收錄的事實，標「（作者開發紀錄）」；這些都已和原筆記核對過。
- 用算術推得、論文和紀錄都沒有直接寫的數字，一律標「推算」。
- 論文和紀錄不一致時，兩邊都列出來，並註明哪個是論文、哪個是紀錄。
- `runs/...` 是原始實驗目錄的名稱。本 repo 的 [`weights/`](../weights/README.md)（Git LFS）收了其中三個模型：842M 主 checkpoint `runs/distill_b1/best.pt`（拆成 `vggt_teacher_init_842M.pt.part0` / `.part1` 兩段，合併方法見 weights/README.md）、464M 論文主模型 `runs/sliced/recover_agg256_ch512/best.pt`（`vggt_sliced_464M_agg256_ch512.pt`）、464M INT8 `runs/sliced/quantized_sliced_int8.pt`（`vggt_sliced_464M_int8.pt`）。其他 checkpoint（`runs/`、`abl/` 下的其餘檔案）不在本 repo。本文件引用的 checkpoint 欄位（`epoch`、`eval`）是 2026-10-17 只解析 checkpoint 裡的 pickle metadata 讀到的，沒有載入權重。
- 在 `distill2/` 底下執行指令時，權重路徑寫成 `../weights/...`；載入方式（含 INT8 檔）見 [`weights/README.md`](../weights/README.md)，重現步驟見 [`REPRODUCE.md`](REPRODUCE.md)。
- 資料路徑寫成 `<DATA_ROOT>`（teacher soft labels）和 `<CO3D_ROOT>`（CO3D 官方標註）。

---

## 0. 閱讀須知

### 0.1 兩種評估：指標同名，但不能混用

| | vs teacher（teacher-relative） | vs GT（CO3D ground truth） |
|---|---|---|
| 程式 | `distill2/eval.py` 的 `quick_eval` / `eval_scene` | `distill2/eval_gt.py --ref co3d` |
| 參考值 | teacher soft labels（`pose_enc.npy`、`depth.npy`） | `<CO3D_ROOT>/<category>/frame_annotations.jgz` 的外參，用 flip_xy 轉成 OpenCV |
| rotation 指標 | 每幀**絕對**旋轉誤差的平均（°；姿態以第 0 幀為參考座標，所以是各幀相對第 0 幀的旋轉），RRA@15 = 誤差 < 15° 的幀比例 | 所有幀對 (i<j) 的**相對**旋轉誤差：RRA@15/30；相對平移方向：RTA@15/30；AUC@30 = RRA 在 1…30° 各門檻的平均（只看旋轉） |
| depth 指標 | abs-rel、δ<1.25（對 teacher depth） | 無（論文只拿 depth 跟 teacher 比） |
| 場景 | test split 依名稱排序後的前 N 個（N = 8 或 12） | 65 個 test 場景全部 |
| 幀 | 每場景等距 12 幀（含第 0 幀） | 每場景等距 10 幀（另有隨機 10 幀版本） |

來源：distill2/eval.py L70–74（等距 12 幀）、L81–88（每幀旋轉誤差）、L98（`rot < 15`）、L102–107（排序取前 N 個）；distill2/eval_gt.py L45–70（指標）、L120–125（`frame_ids`）、L81–83（flip_xy）。

所以「vs teacher 的 RRA@15」和「vs GT 的 RRA@15」連定義都不一樣：前者是每幀絕對誤差，後者是幀對相對誤差。不能拿來互比。

### 0.2 N = 8 還是 12：論文寫法和紀錄不一致

| 結果 | 實際評估子集（紀錄） | 論文寫法 |
|---|---|---|
| Table 1（消融） | 前 8 個 test 場景 | 沒寫場景數 |
| Table 3（層剪枝） | 前 8 個 | 沒寫場景數 |
| Table 4 probe 欄、no-retraining 欄、k=1024 的 2.87° | 前 12 個 | 前 12 個 |
| Table 4 recovered 值（4.18、4.95、5.36、5.20、69.4、77.2、48.1） | **前 8 個** | 前 12 個（§4.1、Table 4 caption） |
| INT8 檢查 5.20° → 5.21° | 前 8 個 | 沒寫 |

根據：

- 預設值：train.py L50、train_sliced.py L49、run_ablation.py L54、run_layer_pruning.py L54、quantize.py L99 都是 `--eval_scenes`/`--num_scenes` = 8；slicegpt_probe.py L126、slicegpt_slice.py L471 是 12（以上都在 distill2/ 下）。
- 報告本身就寫了：distill2/reports/slicegpt_slice.md L15「恢復蒸餾 … eval 8 test scenes」；distill2/reports/slicegpt_slice_combo.md L3–4「zero-shot eval 12 test scenes；恢復蒸餾 eval 8 test scenes」；distill2/reports/quantize_sliced_report.md L5「rot_err 取 8 場景平均」。
- RRA@15 的值本身也對得上（推算）：8 場景 × 12 幀 = 96 幀，所以值會是 1/96 的倍數，例如 0.96875 = 93/96、0.9479 = 91/96、0.78125 = 75/96、0.9896 = 95/96，這些都不是 1/144 的倍數。12 場景 × 12 幀 = 144 幀，值會是 1/144 的倍數，例如 0.993 = 143/144、0.3819 = 55/144、0.2639 = 38/144，這些都不是 1/96 的倍數。有些值兩邊都成立，不能拿來區分，例如 0.1875 = 18/96 = 27/144、0.9583 = 92/96 = 138/144。
- 同一個 842M checkpoint（`runs/distill_b1/best.pt`，即本 repo 的 `weights/vggt_teacher_init_842M.pt.part0` + `.part1` 合併後的檔案）在 12 場景是 2.87°（distill2/reports/slicegpt_probe.md L9），在 8 場景是 3.42°（distill2/reports/quantize_report.md L9；checkpoint 存的 `eval` 也是 3.4232°）。
- 依名稱排序後，前 8 個 test 場景是 apple、backpack、ball、banana 各 2 個，只涵蓋 4 個類別；前 12 個再加上 baseballbat 1 個、baseballglove 2 個、bench 1 個，共 7 個類別，和論文 §4.1 的「first 12 test scenes (7 categories)」一致。這是用和 eval.py L103–104 相同的排序得到的（依資料集 test split 目錄與 meta.json 列出；資料集不在本 repo）。distill2/reports/eval_results_before.json 列出的前 10 個場景名稱（L22–139）順序也相同。

### 0.3 best.pt 怎麼選的

train.py L130–139 和 train_sliced.py L146–160 每隔幾個 epoch 會對 **test split 的前 N 個場景**跑一次 `quick_eval`，rot_err 最低的那次存成 `best.pt`，同一次的 `eval` 也一起存進去。報告印的就是這個值（run_ablation.py L84–91 直接讀 `ck["eval"]`）。這裡沒有獨立的 validation split，所以從訓練 run 的 best.pt 讀出的 vs-teacher 數字（Table 1、Table 3、Table 4 的 recovered 值與各 checkpoint 的 eval）都是「同一子集上幾次評估中最好的一次」；probe、no-retraining、INT8 檢查則是對固定 checkpoint 的單次評估（但這個 checkpoint 本身是用 8 場景選出來的，見 §0.2）。

論文主要用到的幾個 run，best 恰好就是最後一次評估：

- `runs/abl/full`：epoch 11 / 0–11
- `runs/prune/d8`：epoch 11
- `runs/sliced/recover_*`：epoch 5 / 0–5（`recover_triple_trainpe` 除外，它是 epoch 3 / 0–7，見 §4）

`runs/distill_b1` 的 best 則是 epoch 37 / 0–39。GT 評估的 65 個場景也包含這 8 / 12 個場景。

### 0.4 單位

| 量 | 定義 | 來源 |
|---|---|---|
| VRAM（Table 5、Fig. 2） | `torch.cuda.max_memory_allocated() / 1e9`，單位 GB（十進位），不含 CUDA context | measure_unified_bench.py L60；measure_vram_curve.py L44；measure_vram_int8.py L23；measure_efficiency.py L73 |
| 記憶體預算（Table 7） | GiB = 1024³ bytes；JSON 的 key 叫 `budgets_gb`，但值是 GiB | measure_budget_incl_context.py L23 |
| 檔案大小 | MB = 1e6 bytes，量的是 `torch.save(model.state_dict())` 寫出的位元組數（論文 §4.7 也這樣定義 537 MB） | quantize.py L39–42 |
| Latency | 暖機後 N 次的平均（ms）。各腳本的 N 不同：20 次 + 5 次暖機（measure_unified_bench.py L13、measure_efficiency.py L132–133）；15 + 4（quantize.py L46）；10 + 3（slicegpt_slice.py L304） | 同左 |

### 0.5 checkpoint 與數字的對照

| 代稱 | 原始路徑（本 repo 收錄的檔案） | 訓練紀錄（checkpoint metadata） | 用到它的結果 |
|---|---|---|---|
| 842M 主 checkpoint（紀錄裡叫 Path B） | `runs/distill_b1/best.pt`（本 repo：`weights/vggt_teacher_init_842M.pt.part0` + `.part1`，合併後 3,368,981,430 bytes） | `last.pt` 的 epoch = 39，所以至少跑了 40 個 epoch；best 是 epoch 37；8 場景 eval 3.42° | Table 2 的 842M 列（依 distill2/README.md L79）、Table 4 的 k=1024 列（2.87°）、所有切片的基底、Table 5 / Fig. 2 / Table 7 的量測、842M 的 INT8 |
| 842M 消融配方 | `runs/abl/full/best.pt` | 12 epochs；best = epoch 11；4.317° | Table 1 full 列；Table 3「12 (ref.)」的精度 |
| 654M（agg k=384） | `runs/sliced/recover_k384/best.pt` | 恢復 6 epochs；best = epoch 5；4.178° | Table 4；Table 2 的 agg384 列（GT md 沒寫路徑） |
| 616M（agg k=256） | `runs/sliced/recover_k256/best.pt` | 同上；4.953° | Table 4 |
| 516M | `runs/sliced/recover_agg256_ch1024/best.pt` | 同上；5.361° | Table 4 |
| **464M 主模型** | `runs/sliced/recover_agg256_ch512/best.pt`（本 repo：`weights/vggt_sliced_464M_agg256_ch512.pt`，1,908,372,429 bytes，SHA256 `FE6E2CF735DCCB993560C98D21B2E3DC1357F2EE0186986CB6BFE7C7E8F1B352`） | 恢復 6 epochs；best = epoch 5；5.1988°，RRA 0.96875（quick_eval 的每幀 RRA，8 場景；不是 eval_gt 的幀對 RRA，GT 幀對是 0.883 / AUC@30 0.767，見 §2） | Table 2、4、5、7、Fig. 2、INT8 |
| 464M INT8 | `runs/sliced/quantized_sliced_int8.pt`（本 repo：`weights/vggt_sliced_464M_int8.pt`，537,215,008 bytes） | quantize.py 存的 `{"model": state_dict, "quant": ...}`（quantize.py L145），沒有 `epoch` / `eval` 欄位 | 論文的 537 MB；5.20° → 5.21° 是同一次 quantize.py 執行量的（distill2/reports/quantize_sliced_report.md） |
| front end 有切的三個變體 | `runs/sliced/recover_agg256_pe512_ch1024`、`recover_triple_trainpe`、`recover_agg256_pe768_ch512` | 見 §4 | Table 4 下半部 |
| 層剪枝 | `runs/prune/{d8,d6,d4}/best.pt` | 12 epochs | Table 3；Table 5 的 pruned 列 |

用到哪個 checkpoint，依據各腳本或報告寫死的路徑：measure_unified_bench.py L27、L36–39；measure_vram_curve.py L28、L31；measure_vram_int8.py L27；measure_budget_incl_context.py L38、L41；distill2/reports/slicegpt_probe.md L3；distill2/reports/slicegpt_slice.md L3；distill2/reports/slicegpt_slice_combo.md L3；distill2/reports/quantize_report.md L3；distill2/reports/quantize_sliced_report.md L3。主模型的大小與 SHA256 來自 distill2/README.md L190 和 weights/SHA256SUMS.txt。

Table 5、Table 7 和 `vram_curve_int8.json` 的 INT8 數字不是載入 INT8 檔量的，而是把 464M 模型載入後在記憶體裡直接量化（measure_unified_bench.py L29–32、measure_budget_incl_context.py L42–44、measure_vram_int8.py L27–28）。weights/README.md L68 說明這種 weight-only 量化的結果是確定的。

另外有 `runs/distill_b1_full`：20 epochs（`last.pt` epoch = 19，best 也是 epoch 19），8 場景 eval 4.44°。上面列的量測腳本都沒有用它。常被引用的「82.7° → 4.9°」（distill2/CHANGES.md L66–71 的 4.89°）來自這個 20-epoch run，不是論文用的 `runs/distill_b1`（§9.5）。

### 0.6 論文與紀錄不一致一覽

| # | 項目 | 論文 | 紀錄 | 細節 |
|---|---|---|---|---|
| 1 | Table 4 recovered 值與 INT8 檢查的場景數 | 前 12 個 test 場景 | 前 8 個 | §0.2、§4 |
| 2 | §4.5「4.95° almost exactly the probe's prediction (4.91°)」 | 當成同一子集比較 | 4.95° 是 8 場景，4.91° 是 12 場景，跨了子集 | §4 |
| 3 | 842M 主 checkpoint 的訓練長度 | 「main run is 20 epochs of 500 steps」（§4.1） | `runs/distill_b1/last.pt` 的 epoch = 39，至少 40 epochs；train.py L41 預設 `--epochs 40`，distill2/README.md L63、L79–82 也記錄 distill_b1 用的是 train.py 預設（寫 `--epochs 20` 的是另一個 run `runs/distill_b1_full`，L80）。作者開發紀錄裡有一條寫 `--epochs 20 --img_per_seq 2` 的 distill_b1 指令，和 checkpoint 不符；當時實際的 `img_per_seq` 無法從紀錄確認（train.py 預設 4，L43；論文寫每序列 2 幀） | §0.5 |
| 4 | 主模型參數 | 464M / 464.4M | 舊報告寫 466M，是同一個 checkpoint 的錯誤數字（distill2/README.md L306–307） | §4、§6 |
| 5 | 跨 session latency 漂移 | 「within about 7%」（§4.7） | 同一架構最多約 15%（scratch student：123.8 → 105.2 ms，同一支 measure_efficiency.py，推算） | §9.11 |
| 6 | Table 4「FE tuned (8 ep.)」 | 77.2° (0.188) | 報告也是 77.2° (0.188)，但 `best.pt` 存的 eval 是 75.26° (0.198) | §4 |
| 7 | 消融 / 剪枝的訓練長度 | 12–15 epochs of 400 steps | 都是 12 epochs | §1、§3 |
| 8 | teacher fp32 大小 | 5025 MB | 1,256,537,516 參數 × 4 B ≈ 5026 MB（推算下限） | §5 |
| 9 | 加速倍數 | 1.77× / 1.93×（同一 session） | 舊報告 1.64× / 1.91×（混用不同 session） | §5、§10 |
| 10 | Table 7 的文字描述 | 「against at most one for the teacher」、「counts are lower than … Fig. 2」 | 只在 Default 欄成立；Frozen 欄 teacher 在 8 GiB 是 4 幀 | §7 |

---

## 1. Table 1：teacher initialization 消融

論文原表（論文 Table 1，p.7）：

| Config. | Copied | FE frozen | Rot. err. | RRA@15 | abs-rel | δ1 |
|---|---|---|---|---|---|---|
| random | — | ✓ | 76.18° | 0.188 | 0.235 | 0.642 |
| FE only | front end | ✓ | 75.59° | 0.198 | 0.260 | 0.616 |
| agg. only | aggregator | ✓ | 75.58° | 0.208 | 0.265 | 0.574 |
| head only | camera head | ✓ | 75.48° | 0.208 | 0.247 | 0.578 |
| **full** | **all three** | ✓ | **4.32°** | **1.000** | **0.189** | **0.746** |
| full, FE tuned | all three | — | 75.68° | 0.219 | 0.248 | 0.632 |

數字來源（六列都和紀錄四捨五入後相同）：

| 論文列 | 紀錄名稱 | 來源 | 原始值（rot / RRA / abs-rel / δ1） |
|---|---|---|---|
| random | `random` | distill2/reports/ablation_report.md L7；ablation_report.csv L2 | 76.1809 / 0.1875 / 0.2350 / 0.6423 |
| FE only | `patch_only` | ablation_report.md L8；csv L3 | 75.5937 / 0.1979 / 0.2600 / 0.6162 |
| agg. only | `agg_only` | ablation_report.md L9；csv L4 | 75.5850 / 0.2083 / 0.2653 / 0.5740 |
| head only | `head_only` | ablation_report.md L10；csv L5 | 75.4751 / 0.2083 / 0.2468 / 0.5782 |
| full | `full` | ablation_report.md L11；csv L6 | 4.3172 / 1.0000 / 0.1894 / 0.7458 |
| full, FE tuned | `full_ft_pe` | ablation_report.md L12；csv L7 | 75.6819 / 0.21875 / 0.2475 / 0.6317 |

評估設定：

- 全部是 vs teacher：前 8 個 test 場景 × 等距 12 幀（run_ablation.py L54、L79；eval.py L70–74）。論文沒寫場景數。
- 訓練：每個 epoch 400 steps，每 3 個 epoch 評估一次（run_ablation.py L52、L79），共 12 epochs（distill2/README.md L138 `--epochs 12`；`runs/abl/*/last.pt` 的 epoch = 11）。論文 §4.1 寫的是「12–15 epochs」；run_ablation.py 的預設是 15（L51），但實際跑的是 12。
- csv 的 `epoch` 欄是 best.pt 的 epoch 編號：2, 5, 5, 2, 11, 2。塌縮的幾列最好的一次評估都出現得很早。
- random 列的 patch embed 是 hub 上的官方預訓練 DINOv2，有凍結；aggregator 和 camera head 是隨機初始化（run_ablation.py L8；論文 Table 1 caption）。
- 「FE」在紀錄裡叫 `patch_embed`，指整個 DINOv2 ViT-L/14。

論文明確不主張的事（論文 §4.2）：每個元件都「個別必要」。三種兩兩組合沒有跑過。舊紀錄寫的「三者缺一不可」（distill2/README.md L29、L290）比論文說得更強，以論文為準。

---

## 2. Table 2：對 CO3D ground truth 的相對位姿

論文原表（論文 Table 2，p.8）：

| Model | Params | RRA@15 | RRA@30 | RTA@15 | RTA@30 | AUC@30 |
|---|---|---|---|---|---|---|
| VGGT-1B teacher | 1256M | 1.000 | 1.000 | 0.930 | 0.957 | 0.976 |
| Teacher-initialized | 842M | 0.964 | 0.981 | 0.893 | 0.941 | 0.876 |
| Sliced, agg384 | 654M | 0.905 | 0.939 | 0.826 | 0.910 | 0.792 |
| Sliced, agg256+trunk512 | 464M | 0.883 | 0.940 | 0.814 | 0.908 | 0.767 |

數字來源（全部與紀錄相同）：

| 列 | 精度來源 | Params 來源 |
|---|---|---|
| teacher | distill2/reports/eval_gt_co3d10_pathb.md L8（k384、ch512 兩個檔的 L8 也相同） | distill2/reports/efficiency_report.md L10（1256.5M） |
| 842M | eval_gt_co3d10_pathb.md L7 | efficiency_report.md L13（842.2M） |
| 654M | eval_gt_co3d10_k384.md L7 | distill2/reports/slicegpt_slice.md L10（storage 654M） |
| 464M | eval_gt_co3d10_ch512.md L7 | distill2/README.md L192（state_dict 464.43M；參數 464.16M） |

評估設定：

- 三個 md 檔的 L3 都寫「參考：co3d｜場景 65｜每場景 10 幀」。用 `eval_gt.py --ref co3d --num_scenes 65 --frames 10`，等距取幀，flip_xy。
- 每場景 10 幀，也就是 45 個幀對；65 場景共 2925 個幀對（推算）。
- 權重都沒有量化。reports/ 裡沒有任何 INT8 模型的 GT 評估檔。
- 保留率（論文 §4.3）：RRA@15 96.4% / 88.3%，AUC@30 89.8% / 78.6%。驗算（推算）：0.876 / 0.976 = 0.8975，0.767 / 0.976 = 0.7859。

注意事項：

- md 檔沒有寫 checkpoint 路徑。0.883 / 0.767 對應 `recover_agg256_ch512` 是根據 distill2/README.md L194（附了當時的 `eval_gt.py --ckpt` 指令）。842M 列在紀錄裡叫「Path B」；distill2/README.md L79 說 CO3D GT 評估（0.964 / 0.876）用的是 `runs/distill_b1`。論文沒寫是哪個 checkpoint。
- **teacher 列的來源**：`eval_gt.py` 的 `teacher_extri`（L136–139）讀的是預先匯出的 teacher soft labels（`pose_enc.npy`），沒有在同樣 10 幀上重跑 teacher。匯出腳本對每個場景的所有選取影像只做一次 forward（scripts/export_co3d_labels.py L131、L145）。eval.py L73 的註解說一個場景有 50 幀；distill2/reports/eval_results_before.json 的每場景 RRA 都是 0.02 的倍數，也和 50 幀相符（推算）；資料集裡 test 場景的 `meta.json` 記的影像數多數是 50、最少 29（依資料集 test split 目錄與 meta.json 列出；資料集不在本 repo）。所以 teacher 產生 labels 時看到的幀數，可能比 student 評估時的 10 幀多。這是推論，影響沒有量測過。論文寫「teacher and students under an identical protocol」，是指取幀和指標程式相同。
- 協定比文獻容易：這裡 teacher AUC@30 = 0.976，文獻是 0.882（論文 §4.1）。0.882 出自 VGGT 原論文（論文的參考文獻 [6]）的 Table 1，每場景隨機 10 幀（作者開發紀錄）。論文 §4.1 本身沒有標這個數字的出處。
- 早期還有一版「20 場景 × 8 幀」的 GT 評估，見 §9.6。論文一律用 65 場景版。

---

## 3. Table 3：固定寬度 1024 的層剪枝

論文原表（論文 Table 3，p.8）：

| Blocks | Params | Compr. | Rot. err. | RRA@15 | abs-rel | δ1 | Latency (S=4) |
|---|---|---|---|---|---|---|---|
| 12 (ref.) | 842M | 1.49× | 4.32° | 1.000 | 0.189 | 0.746 | 219.1 ms |
| 8 | 741M | 1.69× | 9.45° | 0.781 | 0.201 | 0.744 | 180.1 ms |
| 6 | 691M | 1.82× | 56.78° | 0.208 | 0.214 | 0.712 | 161.4 ms |
| 4 | 641M | 1.96× | 75.74° | 0.198 | 0.227 | 0.658 | 144.0 ms |

數字來源：

| 欄 | 來源 |
|---|---|
| Params、Compr.、Rot. err.、RRA@15、abs-rel、δ1 | distill2/reports/layer_pruning_report.md L9–12；layer_pruning_report.csv L2–5 |
| Latency | distill2/reports/unified_bench.json L6–21（`pathb_d12` 219.1、`d8` 180.1、`d6` 161.4、`d4` 144.0） |

評估設定：

- vs teacher，前 8 個 test 場景 × 12 幀（run_layer_pruning.py L54）。
- Path B 配方，按 stride 取 block，12 epochs × 400 steps，`img_per_seq=2`（layer_pruning_report.md L3–4）。
- Latency 是 S=4、518×518、bf16 autocast、no_grad，暖機 5 次後取 20 次平均，同一 session 量完（measure_unified_bench.py L4–5、L13）。

注意事項：

- 「12 (ref.)」的精度數字和 Table 1 full 列是同一個 checkpoint（`runs/abl/full`；layer_pruning_report.csv L2 的值與 ablation_report.csv L6 完全相同）。但 219.1 ms 是在 `runs/distill_b1/best.pt` 上量的（measure_unified_bench.py L36）。兩者架構相同、權重不同。
- 較早一個 session 量的 latency：222.8 / 186.2 / 165.3 / 145.9 ms（distill2/reports/efficiency_prune.md L12–15）。
- 紀錄裡有、論文沒放的：d6、d4 的 VRAM 4.67 / 4.28 GB（unified_bench.json L14–21）。
- d8 延長到 24 epochs 的結果見 §9.8。論文沒有任何剪枝模型的 GT 評估；紀錄裡只有 d8 加切片的 GT（§9.9）。

---

## 4. Table 4：三條殘差流的寬度切片

論文原表（論文 Table 4，p.9）：

| Streams sliced | Energy | Probe | No retr. / recov. | Params |
|---|---|---|---|---|
| agg k=1024 (none) | 100.00% | 2.87° | — | 842M |
| agg k=512 | 99.38% | 3.45° | 5.62° / — | 692M |
| agg k=384 | 98.91% | 3.86° | 34.60° / 4.18° | 654M |
| agg k=256 | 97.98% | 4.91° | 49.74° / 4.95° | 616M |
| *（下半部：aggregator 固定 k=256；括號內是 teacher-relative RRA@15）* | | | | |
| agg256 + trunk1024 | — | — | 5.36° (0.948) | 516M |
| **agg256 + trunk512** | — | — | **5.20° (0.969)** | **464M** |
| agg256+trunk1024+fe512, FE frozen | — | — | 69.4° (0.177) | 365M |
| agg256+trunk1024+fe512, FE tuned (8 ep.) | — | — | 77.2° (0.188) | 365M |
| agg256+trunk512+fe768, FE frozen | — | — | 48.1° (0.208) | ∼390M |

數字來源與實際評估子集：

| 列 | 論文值 | 紀錄來源 | 子集（紀錄） |
|---|---|---|---|
| k=1024 | 100.00% / 2.87° / 842M | slicegpt_probe.md L9–10；slicegpt_slice.md L8 | 12 |
| k=512 | 99.38% / 3.45° / 5.62° / 692M | slicegpt_probe.md L14；slicegpt_slice.md L9 | 12 |
| k=384 | 98.91% / 3.86° / 34.60° / 4.18° / 654M | probe L15；slice L10（no-retrain）；slice L19（recovered） | probe 與 no-retrain 是 12；recovered 是 **8** |
| k=256 | 97.98% / 4.91° / 49.74° / 4.95° / 616M | probe L16；slice L11；slice L20 | 同上 |
| agg256+trunk1024 | 5.36° (0.948) / 516M | slicegpt_slice_combo.md L21 | **8** |
| agg256+trunk512 | 5.20° (0.969) / 464M | slicegpt_slice_combo.md L22（紀錄寫 **466**） | **8** |
| fe512, frozen | 69.4° (0.177) / 365M | slicegpt_slice_combo.md L19 | **8** |
| fe512, tuned | 77.2° (0.188) / 365M | slicegpt_slice_combo.md L20 | **8** |
| fe768, frozen | 48.1° (0.208) / ∼390M | slicegpt_slice_combo.md L23 | **8** |

以上檔案都在 distill2/reports/ 下。紀錄的命名：`pe` = front end（FE），`ch` = camera-head trunk，`agg` = aggregator。

評估與方法設定：

- **Probe**：用 hook 把每個 block 的輸出投影到它自己的 centered PCA 前 k 維，不改任何權重。校準用 train split 依名稱排序的前 8 個場景，每場景 6 幀（slicegpt_probe.py L81–82、L124–125 的預設值）。論文 §3.4 寫「eight calibration scenes」，§3.5 寫明是依字典序的前 8 個訓練場景、全部屬於 apple 類別；每場景 6 幀只寫在程式預設值裡。
- **Energy 欄**是 probe 用的 per-block centered 基底保留的能量。真正切片用的是全域共享、uncentered 的基底，它的能量論文沒有印出，紀錄裡有：k = 512 / 384 / 256 分別是 99.52% / 99.12% / 98.30%（slicegpt_slice.md L9–11）。
- **Params**：用的是 storage = state_dict 總元素數，包含 304M 凍結的 patch embed 和 de-slice 矩陣（slicegpt_slice.md L4）。
- **Recovery**：6 epochs × 400 steps，lr 2e-5，`img_per_seq` 2，每 2 個 epoch 評估一次，eval 8 個場景，DINOv2 凍結（train_sliced.py L41–49（數值預設）、L8 與 L51–53（`--train_pe` 預設關，DINOv2 凍結）；slicegpt_slice.md L13）。「約 25 分鐘」是論文 §3.5 的說法，reports/ 裡沒有對應的計時 log。
- **等價性檢查**：用隨機正交 Q、k = 滿維時，pose 的 max|Δ| = 3.0e-7，depth 是 2.6e-6（slicegpt_slice_combo.md L5；論文 §3.5、Appendix C）。

論文沒印、紀錄裡有的 RRA@15：

- probe：1.000 / 0.993 / 0.986 / 0.958（k = 1024 / 512 / 384 / 256，依序是 slicegpt_probe.md L10、L14、L15、L16）
- no-retraining：0.958 / 0.382 / 0.264（k = 512 / 384 / 256；slicegpt_slice.md L9–11）
- recovered：k384 1.000、k256 0.990（slicegpt_slice.md L19–20）

與紀錄的差異：

1. **場景數**：論文說 Table 4 全部是前 12 個 test 場景（§4.1、Table 4 caption），但 recovered 值是前 8 個（§0.2）。所以 §4.5「recovered k=256 的 4.95° 幾乎等於 probe 預測的 4.91°」是拿 8 場景和 12 場景的值在比。
2. **2.87° 和 4.32° 是不同 checkpoint**：2.87° 是 `runs/distill_b1/best.pt`（40-epoch run）在 12 場景的值；4.32° 是 `runs/abl/full`（12-epoch 消融配方）在 8 場景的值。論文的 caption 有說是不同 checkpoint，但沒說評估子集也不同。
3. **464M 和 466M**：slicegpt_slice_combo.md L22、L28 寫 466M，distill2/README.md L306–307 說明這是同一模型的錯誤數字，正確是 464.4M（state_dict 464.43M，參數 464.16M；distill2/README.md L192）。L28 的「vs teacher 2.70×」也是用 466M 算的；用 464.4M 是 1256.5 / 464.4 = 2.71×（推算）。
4. **FE tuned (8 ep.)**：checkpoint `runs/sliced/recover_triple_trainpe/best.pt` 的 best 是 epoch 3（共 8 epochs，`last.pt` epoch = 7），存的 eval 是 75.26°、RRA 0.198。論文和報告都是 77.2°、0.188（報告 L3–4 說恢復蒸餾 eval 8 個場景；0.188 ≈ 18/96，推算）。從 checkpoint 找不回 77.2° 的出處，因為 `last.pt` 不存 eval。兩個值都落在塌縮區（約 75–77°），不影響結論。
5. **front end 有切的三列是不是 8 場景**：checkpoint eval 和 8 場景相符（推算）。`recover_agg256_pe512_ch1024` 是 69.41° / 0.1771 = 17/96，不是 1/144 的倍數；`recover_agg256_pe768_ch512` 是 48.14° / 0.2083 = 20/96，但它也等於 30/144，單看這個值不能區分。

論文的界定（論文 §4.5、§4.6）：

- 沒有做 identity（直接截斷通道）和隨機正交基底的對照，所以只主張共享基底「足以」在 k=256 恢復，不主張它比簡單基底好。
- front end 是「在測過的設定下」不可切；per-site 基底沒試過。

---

## 5. Table 5：效率

論文原表（論文 Table 5，p.10）：

| Model | Params | GFLOPs | VRAM | Latency | Max frames |
|---|---|---|---|---|---|
| VGGT-1B teacher | 1256.5M | 16814.3 | 9.21 GB | 388.1 ms | 24 |
| Scratch student (conv, 512)† | 146.4M | 2399.8 | 1.76 GB | 123.8 ms | — |
| Teacher-initialized | 842.2M | 9919.2 | 5.85 GB | 219.1 ms | 48 |
| — pruned to 8 blocks | 741.4M | 8194.1 | 5.07 GB | 180.1 ms | — |
| Sliced (agg256 + trunk512) | 464M | 7496.8 | 3.89 GB | 201.3 ms | 64 |
| — + INT8 weight-only | 464M | 7496.8 | 1.81 GB | 229.9 ms‡ | 80 |

註腳：†是初步診斷 run，pose 已塌縮，latency 和 VRAM 來自較早的一批量測。‡這個 torch 版本沒有 INT8 kernel，所以比 bf16 慢。

數字來源：

| 列 | Params | GFLOPs | VRAM / Latency | Max frames |
|---|---|---|---|---|
| teacher | efficiency_report.md L10（1256.5；csv L2 是 1256.537516） | budget_incl_context_frozen.json L32 | unified_bench.json L2–5 | vram_curve.json L48–49（S=24 → 16.00 GB；S=28 → 17.08） |
| scratch† | efficiency_report.md L11 | flops_extra.json L3 | efficiency_report.md L11（1.76 GB、123.8 ms） | — |
| 842M | efficiency_report.md L13（842.2） | budget_incl_context_frozen.json L24 | unified_bench.json L6–9 | vram_curve.json L33–34（S=48 → 13.35；S=64 → 16.27） |
| pruned 8 | efficiency_prune.md L13（741.4） | flops_extra.json L2 | unified_bench.json L10–13 | — |
| sliced | distill2/README.md L192 | budget_incl_context_frozen.json L16 | unified_bench.json L22–25 | vram_curve.json L15–16（S=64 → 13.92；S=80 → 16.72） |
| sliced + INT8 | 同上 | 同 sliced（weight-only，FLOPs 不變） | unified_bench.json L26–29 | vram_curve_int8.json L15–16（S=80 → 14.65；S=96 → 17.47） |

除了 sliced 兩列的 Params（distill2/README.md）之外，其他來源檔都在 distill2/reports/ 下。

量測設定：

- **Latency / VRAM**：S=4、518×518、bf16 autocast、no_grad、暖機 5 次、20 次平均，同一 session（measure_unified_bench.py L4–5、L13、L45–61）。參數維持載入時的狀態（requires_grad=True），也就是 Table 7 的「Default」狀態（measure_budget_incl_context.py L47–49 的註解說明了兩種模式）。
- **GFLOPs**：`torch.utils.flop_counter.FlopCounterMode`，S=4（measure_budget_incl_context.py L104–109；measure_flops_extra.py L14–20）。scratch student 用隨機初始化建出來，並設 `enable_track=False`（measure_flops_extra.py L28），因為 FLOPs 和權重值無關。
- **Max frames**：在 `vram_curve*.json` 上取峰值 ≤ 16 GB 的最大 S，只是格點值，不是精確上限。這是論文 Table 5 caption 的定義（「the largest S staying within 16 GB」）；INT8 的腳本把這條規則寫成程式（measure_vram_int8.py L42，`g <= 16.0`），另外三個模型的 measure_vram_curve.py 沒有寫，是從曲線讀出來的。JSON 裡的 `max_frames` 欄（96 / 96 / 64）是「沒有 OOM 的最大 S」（measure_vram_curve.py L57、L63），不是論文的值。超過 16 GB 的點是靠 Windows 共享記憶體溢出才跑完的（作者開發紀錄）。另外，budget JSON 記錄這張卡的總量是 16303 MiB（budget_incl_context_frozen.json L14），約 17.09 GB（推算）。
- **加速倍數**（論文 §4.7）：1.77× = 388.1 / 219.1，1.93× = 388.1 / 201.3（推算驗證）。

與紀錄的差異和注意事項：

- scratch student 的 123.8 ms / 1.76 GB 出自 2026-07-01 那批量測（efficiency_report.md；日期是檔案時間，作者開發紀錄）。2026-07-12 的 efficiency_prune.md L10 用同一支腳本重測同一架構（measure_efficiency.py L86–88 用隨機初始化建出 conv-512 student，不載入權重），得到 105.2 ms / 1.76 GB，快了約 15%（推算：(123.8 − 105.2) / 123.8）。論文 † 說「earlier, unrepeated batch」，但紀錄顯示有重測過。
- quantize 報告裡的 INT8 VRAM 和 Table 5 不同：distill2/reports/quantize_sliced_report.md L10 是 3.68 GB，Table 5 是 1.81 GB。quantize.py 量 INT8 模型時，基準模型還留在 GPU 上：L115 先載入 `base`，L127 又載入第二份，`base` 一直沒有釋放。所以 3.68 GB 很可能包含了基準模型（推論，未重測）。latency 也不同（248.8 vs 229.9 ms），而且 quantize.py 是 15 次平均。論文用的是 unified_bench.json。
- **GFLOPs 和 GMACs 不能互比**：distill2/reports/efficiency_report.md 和 efficiency_prune.md 的 GMACs 是 fvcore 算的 MAC，報告自己註明注意力等算子可能被低估（efficiency_report.md L6）。例如 842M 是 3666.1 GMACs，同一模型 Table 5 是 9919.2 GFLOPs。
- teacher 的 1256.5M 含 65.9M track head，student 都不含（論文 Table 6）。不含 track head 時，比例是 1.41× / 2.56×（推算，論文沒給）。
- 論文沒給 GFLOPs 比例。teacher / sliced = 2.24×（推算）。

---

## 6. Table 6：各元件參數預算

論文原表（論文 Table 6，p.12）：

| Component | Teacher | T.-init. | Sliced | Treatment in the students |
|---|---|---|---|---|
| Front end (ViT-L/14) | 304.4M | 304.4M | 304.6M^a | inherited, frozen, not sliced |
| Aggregator | 604.7M | 302.4M | 76.0M | every other block, k=256 |
| Camera head | 216.2M | 216.2M | 64.6M | inherited, trunk to 512 |
| Depth head | 32.7M | 9.6M | 9.6M | narrowed, from scratch |
| Point head | 32.7M | 9.6M | 9.6M | narrowed, unsupervised |
| Track head | 65.9M | — | — | removed |
| **Total** | **1256.5M** | **842.2M** | **464.4M** | |

^a 含 aggregator 介面上 1024 × 256 的 adapter（0.26M）。

來源與核對：

- distill2/reports/ 裡**沒有**逐元件參數的檔案。論文 caption 說這些數字是從 checkpoint 的 state dict 數出來的。兩個學生的 checkpoint 就是本 repo weights/ 裡的 842M 與 464M 檔。
- 總數和紀錄一致：1256.5M（efficiency_report.md L10）、842.2M（efficiency_report.md L13）、464.4M（distill2/README.md L192，state_dict 464.43M）。
- 加總核對（推算）：
  - T.-init.：304.4 + 302.4 + 216.2 + 9.6 + 9.6 = 842.2
  - Sliced：304.6 + 76.0 + 64.6 + 9.6 + 9.6 = 464.4
  - Teacher：304.4 + 604.7 + 216.2 + 32.7 + 32.7 + 65.9 = 1256.6。和 1256.5 差 0.1M，屬於 caption 說的四捨五入誤差。
- Adapter 1024 × 256 = 262,144，約 0.26M（推算）。
- 複製了哪些元件（張量數層級）：distill2/README.md L91–98。depth head 和 point head 不複製，都是隨機初始化；depth head 由蒸餾訓練，point head 沒有任何 loss 監督（論文 Appendix B）。

---

## 7. Table 7：模擬記憶體預算下的幀數（Appendix D）

論文原表（論文 Table 7，p.15；單位 GiB，含 CUDA context）：

| Model | Default 4 GiB | Default 6 GiB | Default 8 GiB | Default 12 GiB | Frozen 4 GiB | Frozen 6 GiB | Frozen 8 GiB | Frozen 12 GiB |
|---|---|---|---|---|---|---|---|---|
| VGGT-1B teacher (1256M) | 0 | 0 | 1 | 8 | 0 | 1 | 4 | 12 |
| Teacher-initialized (842M) | 0 | 4 | 12 | 32 | 1 | 4 | 20 | 48 |
| Sliced (464M), ours | 2 | 8 | 24 | 48 | 4 | 16 | 32 | 48 |
| Sliced + INT8, ours | 8 | 20 | 32 | 56 | 8 | 20 | 32 | 56 |

數字來源（32 格都和 JSON 相同）：

| 欄 | 檔案 | 行 |
|---|---|---|
| Default | distill2/reports/budget_incl_context_paper.json | teacher L43–49、pathb L36–42、sliced L29–35、sliced_int8 L22–28 |
| Frozen | distill2/reports/budget_incl_context_frozen.json | teacher L31–38、pathb L23–30、sliced L15–22、sliced_int8 L45–51 |

量測方法：

- 每個預算 B（GiB），把 caching allocator 限制在 B 減去 context，用 `torch.cuda.set_per_process_memory_fraction` 實作（measure_budget_incl_context.py L111–114）。
- context 是暖機 cuBLAS / cuDNN 之後量到的，為 303 MiB（paper JSON L20；frozen JSON L13；腳本 L86–97）。
- S 的格點是 {1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 96}（腳本 L20），從小往上跑，遇到第一個 OOM 就停。這個格點在 80 之後直接跳到 96。
- Default = 參數 requires_grad=True，和 Table 5 / Fig. 2 一樣；Frozen = requires_grad=False（腳本 L47–49；paper JSON L2 的 note）。
- 量測時其他行程已占用 1271 MiB（Default）或 1316 MiB（Frozen）（paper JSON L19；frozen JSON L12），卡的總量是 16303 MiB。

紀錄有、論文刻意省略的：16 GiB 欄。Default 是 teacher 20、842M 64、sliced 72、INT8 80；Frozen 是 28、64、72、80。論文的理由是 16 GiB 超過這張卡實際可用的記憶體（論文 Appendix D）。

論文文字和數據的關係（JSON 推算）：

- 摘要和 §4.7 說 teacher「at most one」frame，只在 Default 欄成立。Frozen 欄 teacher 在 8 GiB 能跑 4 幀（frozen JSON L35）。
- Appendix D 說「counts are lower than those read off the budget lines of Fig. 2」，只有 Default 的 sliced 模型成立（24 vs Fig. 2 的 28）。842M 兩邊都是 12。Frozen 欄的數字反而比 Fig. 2 多。另外，Fig. 2 的門檻是 8 GB，Table 7 是 8 GiB（約 8.59 GB，推算），單位不同。
- 這些全部是在 RTX 5080 上模擬的，沒有測過實體邊緣裝置（論文 §4.7、Limitations）。

---

## 8. Figure 2：峰值推論記憶體與輸入幀數（Appendix D）

論文的 Fig. 2 只有**三條曲線**：teacher、842M、464M，沒有 INT8 曲線。圖中有 8 GB 和 16 GB 兩條預算線。條件是 RTX 5080、518×518、bf16、no_grad，數值不含 CUDA context（論文 Fig. 2 caption）。

數據（GB = 1e9 bytes）：

| S | Teacher (1256M) | Teacher-init. (842M) | Sliced (464M) | Sliced + INT8（**不在 Fig. 2**） |
|---|---|---|---|---|
| 2 | 8.18 | 5.26 | 3.32 | 1.23 |
| 4 | 9.21 | 5.85 | 3.89 | 1.81 |
| 8 | 11.54 | 7.09 | 5.13 | 3.06 |
| 12 | 12.63 | 7.64 | 5.68 | 3.61 |
| 16 | 13.78 | 8.24 | 6.28 | 4.20 |
| 20 | 14.86 | 8.79 | 6.83 | 4.75 |
| 24 | 16.00 | 9.38 | 7.42 | 5.35 |
| 28 | 17.08 | 9.93 | 7.97 | 5.90 |
| 32 | 18.22（圖中超出座標軸） | 10.53 | 8.57 | 6.50 |
| 40 | 20.45 | 11.88 | 9.72 | 7.64 |
| 48 | 22.68 | 13.35 | 11.10 | 9.03 |
| 64 | 27.19 | 16.27 | 13.92 | 11.85 |
| 80 | — | 19.18 | 16.72 | 14.65 |
| 96 | — | 22.10 | 19.54 | 17.47 |

來源：

- teacher / 842M / 464M：distill2/reports/vram_curve.json，`sliced` 在 L2–19，`pathb` 在 L21–38，`teacher` 在 L40–55。
- INT8：distill2/reports/vram_curve_int8.json L2–17。
- 量測腳本：distill2/measure_vram_curve.py（L18 是 S 格點，L35–44 是量測方式）、distill2/measure_vram_int8.py。

注意事項：

- 每條曲線在圖上畫到第一個超過 16 GB 的格點為止。超過 16 GB 的值靠 Windows 共享記憶體溢出才跑完（作者開發紀錄），實務上不可用。
- 論文 §4.7 引用的值都可以從這張表讀到：16 GB 內 teacher 24 幀、sliced 64 幀；8 GB 內 teacher 連 2 幀都放不下（S=2 就要 8.18 GB），sliced 可到 28 幀（7.97 GB）。
- S=4 那一列就是 Table 5 的 VRAM 欄。
- 論文沒寫、只在 JSON 裡的值：842M 在 8 GB 內最多 12 幀（7.64 GB），INT8 是 40 幀（7.64 GB）。
- 這組曲線裡，論文只用到 INT8 的兩個數字：S=4 時 1.81 GB，以及 16 GB 內最多 80 幀（Table 5）。Table 7 的 INT8 幀數是另一組含 context 的量測（§7）。

---

## 9. 其他重要結果（論文未收錄或只簡述）

### 9.1 早期 scratch student 的 camera-token 塌縮（論文 §3.2 只簡述）

- **模型**：conv patch embed、寬度 512、12 blocks、8 heads（論文 §3.2），146.4M 參數（distill2/reports/efficiency_report.md L11）。
- **結果**（vs teacher，10 個場景，舊版評估腳本）：mean_rot_err 82.35°、RRA@15 0.098、abs-rel 0.238、δ1 0.620、mean_trans_err 0.315、focal_rel_err 0.195（distill2/reports/eval_results_before.json L8–19；`per_scene` 共 10 筆）。
- **壓縮比的寫法不同**：這個 JSON 的 `compression_ratio` 是 6.83，因為它把 teacher 近似成 1.0B（L3–6）。論文的 8.6× 是 1256.5 / 146.4（推算）。
- **診斷現象**：預測的四元數和平移跨幀幾乎相同（跨幀 std 約 0.01，teacher 約 0.5）；camera token 跨幀 cosine 0.998；`camera_token` 參數幾乎沒有離開初始值。論文 §3.2 有寫這三點；紀錄裡寫在 distill2/CHANGES.md L26–28（cosine 0.998 也見 distill2/README.md 一、核心發現），reports/ 裡沒有對應的數值檔。
- **兩組「起點」數字**：distill2/CHANGES.md L66–71 的起點是 82.7° / RRA 0.135 / abs-rel 1.04 / δ1 0.40，與上面的 JSON 不同，紀錄沒寫它出自哪一次評估；同一張表的終點 4.89° 是 20-epoch 的 `runs/distill_b1_full`，不是論文的 842M checkpoint（§0.5、§9.5）。論文 §3.2 用的是 JSON 的值（82.3°、0.098、0.238），並說明這是「preliminary diagnostic」，不能和 Tables 1–4 直接比。

### 9.2 DINOv2-S（寬度 384）從零訓練

- 110.7M 參數，11.3×；VRAM 1.52 GB；latency 109.9 ms（distill2/reports/efficiency_report.md L12），另一 session 是 103.5 ms（efficiency_prune.md L11）。這兩次都是用隨機初始化的同一架構量的（measure_efficiency.py L89–92）。
- rotation 約 82°，塌縮（作者開發紀錄）。reports/ 裡沒有這個 run 的精確誤差檔，runs/ 裡也沒有對應的 checkpoint。

### 9.3 二次蒸餾：拿 842M 當 teacher 訓練小 student（負面結果）

- **做法**：student 用 DINOv2-S reg 的 patch embed、寬度 384、12 blocks、6 heads（checkpoint meta）。loss 有三部分：對原始 1B soft labels 的 response KD；student camera token 對 Path B camera token 的逐幀 cosine feature KD；跨幀結構 KD（distill2/train2.py L1–17、L75–76；distill2/loss2.py 說明見 distill2/README.md L263）。指令見 distill2/README.md L146–150。
- **結果**：
  - 作者開發紀錄：加 feature KD 約 75°、加跨幀結構 KD 約 77°。
  - checkpoint（vs teacher，8 場景）：`runs/distill2_s/best.pt` 75.28°、RRA 0.208（best epoch 9 / 0–39）；`runs/distill2_s2/best.pt` 76.75°、RRA 0.198（best epoch 1 / 0–39）。
  - 紀錄沒寫哪個 checkpoint 對應哪種 KD 設定。
- 兩個結果都在塌縮區。

### 9.4 投影初始化：把 842M 的權重 SVD 投影到寬度 512（負面結果）

- **做法**：用 Path B aggregator「寫入 residual 的權重」做 SVD，取前 512 個左奇異向量當共用投影 P，每個 block 做 W' = P W Pᵀ。LayerNorm 和 LayerScale 只做近似處理（distill2/proj_init.py L1–17）。架構是 conv patch embed、512、12 blocks、8 heads，meta 標記 `init: proj_from_pathb`。
- **結果**：checkpoint `runs/proj512/best.pt` 是 75.52°、RRA 0.208（8 場景；best epoch 1；`last.pt` epoch = 7，共 8 epochs）。作者開發紀錄寫約 77°，和 checkpoint 值不同。不論哪個值都是塌縮。
- 這個實驗和事後切片的差別：事後切片在 slicegpt_slice.py 裡有精確的 LN / LayerScale 折疊和 head 介面的 de-slice（§4）。

### 9.5 842M 的其他評估

| 評估 | 結果 | 子集 | 來源 |
|---|---|---|---|
| vs teacher，65 場景 | 4.89°、RRA 0.968、abs-rel 0.176、δ1 0.698 | 65 場景 × 12 幀 | distill2/CHANGES.md L66–71；distill2/README.md L80、L283 說這一行是 20 epochs 的 `runs/distill_b1_full` |
| `runs/distill_b1/best.pt` | 3.42°、RRA 1.000、abs-rel 0.177、δ1 0.752 | 8 場景 | checkpoint eval；distill2/reports/quantize_report.md L9 |
| 同一 checkpoint | 2.87°、RRA 1.000、abs-rel 0.1622、δ1 0.756 | 12 場景 | distill2/reports/slicegpt_probe.md L9 |
| 相對位姿 vs teacher | RRA@15 0.984、RRA@30 0.988、RTA@15 0.977、RTA@30 0.986、AUC@30 0.888 | 20 場景 × 8 幀，參考值是 teacher | distill2/reports/eval_gt_report.md L3、L7（沒寫 checkpoint 路徑） |
| `runs/distill_b1_full/best.pt`（20 epochs） | 4.44°、RRA 0.958 | 8 場景 | checkpoint eval |

### 9.6 早期 GT 評估：20 場景 × 8 幀（已被 65 場景版取代）

| 模型 | RRA@15 | RRA@30 | RTA@15 | RTA@30 | AUC@30 | 來源 |
|---|---|---|---|---|---|---|
| teacher | 1.000 | 1.000 | 0.975 | 0.989 | 0.981 | eval_gt_co3d_pathb.md L8 |
| 842M | 0.979 | 0.988 | 0.954 | 0.979 | 0.877 | eval_gt_co3d_pathb.md L7 |
| 654M | 0.911 | 0.945 | 0.887 | 0.954 | 0.780 | eval_gt_co3d_k384.md L7 |
| 464M | 0.907 | 0.943 | 0.864 | 0.954 | 0.765 | eval_gt_co3d_ch512.md L7 |

以上檔案在 distill2/reports/ 下，每個檔的 L3 都寫「場景 20｜每場景 8 幀」。作者早期文件引用的 842M RRA@15 0.979 就是這一版（作者開發紀錄）。論文 Table 2 用的是 65 場景 × 10 幀版（論文 Table 2 caption、§4.1）。

### 9.7 取幀方式的穩健性：隨機 10 幀（論文 §4.1 只寫最大差值）

| 模型 | RRA@15 等距 → 隨機 | Δ（推算） | AUC@30 等距 → 隨機 | Δ（推算） |
|---|---|---|---|---|
| teacher | 1.000 → 1.000 | 0 | 0.976 → 0.979 | +0.003 |
| 842M | 0.964 → 0.965 | +0.001 | 0.876 → 0.882 | +0.006 |
| 654M | 0.905 → 0.923 | +0.018 | 0.792 → 0.813 | +0.021 |
| 464M | 0.883 → 0.875 | −0.008 | 0.767 → 0.762 | −0.005 |

- 來源：distill2/reports/eval_gt_co3d10rand_{pathb,k384,ch512}.md L7–8；等距版見 §2。
- 隨機版只用了一個 seed（0）（作者開發紀錄）。`--random_frames` 的值就是 seed，所有場景共用同一個亂數產生器（eval_gt.py L151–152、L169；取幀在 L120–125）。
- 最大差值 0.018（RRA@15）和 0.021（AUC@30）與論文 §4.1 相同。
- 隨機版其他欄位：teacher RTA@15 是 0.925（等距版 0.930）；464M 的 RTA@15 / RTA@30 是 0.790 / 0.890。

### 9.8 d8 延長訓練

d8 從 12 epochs 延長到 24 epochs，結果是 9.94°、RRA 0.8125（8 場景；`runs/prune/d8_long/best.pt`，best epoch 23）。沒有比 12 epochs 的 9.45° 更好，作者開發紀錄稱之為「平台期約 9.5–10°」。

### 9.9 層剪枝加寬度切片：d8 + agg256 + ch512（論文 Limitations 列為未來工作，但紀錄裡已經跑過一組）

| 模型 | rot_err | RRA@15 | storage | latency | 子集 | 來源 |
|---|---|---|---|---|---|---|
| d8 原模型 | 9.06° | 0.819 | 741M | 190 ms | 12 場景（推算：0.819 ≈ 118/144） | slicegpt_d8_combo.md L10 |
| d8 + 切片，零重訓 | 52.79° | 0.306 | 439M | 178 ms | 12 場景 | slicegpt_d8_combo.md L11 |
| d8 + 切片 + 恢復 6 epochs | 9.08° | 0.740 | 439M | 178 ms | 8 場景（checkpoint eval 9.082 / 0.7396 = 71/96） | slicegpt_d8_combo.md L17 |

以上檔案在 distill2/reports/ 下。

- 基底是 `runs/prune/d8/best.pt`，三條流的基底在 d8 上重新校準（slicegpt_d8_combo.md L3–4）。
- **對 GT（65 場景 × 10 幀）**：RRA@15 0.645、RRA@30 0.872、RTA@15 0.615、RTA@30 0.843、AUC@30 0.564（distill2/reports/eval_gt_co3d10_d8combo.md L7）。全深度的 464M 是 0.883 / 0.767。
- 報告說「9.08° = 9.06°，切片幾乎零代價」（slicegpt_d8_combo.md L31），但那是拿 8 場景和 12 場景的值比。用同樣 8 場景比，純 d8 是 9.45°（layer_pruning_report.md L10），恢復後的組合模型比純 d8 多訓練了 6 epochs（slicegpt_d8_combo.md L4）。
- 紀錄裡沒有純 d8 的 GT 評估，所以切片在 d8 上的 GT 代價無法單獨分離出來。
- 178 ms 是 slicegpt_slice.py 的 bench 量的（10 次平均），和 Table 5 不是同一個 session。報告的「2.16×」是拿另一個 session 的 teacher 384.6 ms 算的（slicegpt_d8_combo.md L32）。

### 9.10 INT8 weight-only

用 torchao 做 INT8 weight-only。紀錄裡兩個模型都有做（論文只報 464M）：

| 模型 | 大小 | rot_err（8 場景） | abs-rel | VRAM / latency（quantize.py，15 次平均） | 來源 |
|---|---|---|---|---|---|
| 842M fp32 | 3369 MB | 3.42° | 0.177 | 5.85 GB / 214.8 ms | distill2/reports/quantize_report.md L9 |
| 842M INT8 | 915 MB（3.68×） | 3.42° | 0.177 | 5.57 GB / 237.8 ms | quantize_report.md L10、L12 |
| 464M fp32 | 1858 MB | 5.20° | 0.193 | 3.89 GB / 223.9 ms | distill2/reports/quantize_sliced_report.md L9 |
| 464M INT8 | 537 MB（3.46×） | 5.21° | 0.193 | 3.68 GB / 248.8 ms | quantize_sliced_report.md L10、L12 |

- 3369 MB 和 `runs/distill_b1/best.pt` 的檔案大小 3,368,981,430 bytes 吻合（本 repo 的 weights/vggt_teacher_init_842M.pt.part0 + .part1 合併後就是這個檔）。
- 537 MB 和 `runs/sliced/quantized_sliced_int8.pt` 的 537,215,008 bytes 吻合。這個檔收在本 repo 的 `weights/vggt_sliced_464M_int8.pt`，可以照 weights/README.md L57–66 的方式載入：先用 `load_sliced` 建出 fp32 切片結構，再用 torchao 的 `quantize_(..., Int8WeightOnlyConfig())` 轉成 INT8 結構，最後 `load_state_dict`。842M 的 INT8 檔（915 MB）不在本 repo。
- 464M 的 checkpoint 檔（本 repo 的 weights/vggt_sliced_464M_agg256_ch512.pt）有 1,908,372,429 bytes，比 1858 MB 多約 50 MB（推算）。多出來的是存檔裡另外存放的完整基底 `Q_agg`（1024²）、`Q_pe`（1024²）、`Q_ch`（2048²）（distill2/README.md L195）。這 6,291,456 個元素在 checkpoint 裡是 float64（8 bytes），正好約 50.3 MB（推算）。
- 表中 INT8 的 VRAM 和 latency 不可當正式值，原因見 §5。正式值是 1.81 GB / 229.9 ms（unified_bench.json）。
- **論文立場**：只量過 teacher-relative rotation error 的變化（0.01°）。INT8 模型的 GT 精度和 depth 都沒有重測，所以論文不主張無損（論文 §4.7）。舊紀錄寫的「零損失」（distill2/README.md L289、L293；作者開發紀錄）比論文說得更強。
- abs-rel 0.193 是 464M 的 depth 數字，只在紀錄裡（quantize_sliced_report.md L9–10；checkpoint eval 0.1932），論文沒有報告切片模型或 INT8 模型的 depth。

### 9.11 Latency 的跨 session 漂移

| 模型 | 各次量測（ms） | 最大 / 最小（推算） |
|---|---|---|
| teacher | 384.6（efficiency_report.md L10）、388.1（unified_bench.json L3） | 1.009 |
| 842M | 233.9（efficiency_report.md L13）、222.8（efficiency_prune.md L12）、219.1（unified_bench.json L7）、214.8（quantize_report.md L9）、223（slicegpt_slice.md L8）、224（slicegpt_slice_combo.md L11） | 1.089 |
| d8 | 186.2（efficiency_prune.md L13）、180.1（unified_bench.json L11）、190（slicegpt_d8_combo.md L10） | 1.055 |
| 464M | 201.3（unified_bench.json L23）、201.1（distill2/README.md L286）、223.9（quantize_sliced_report.md L9） | 1.113 |
| 464M INT8 | 229.9（unified_bench.json L27）、248.8（quantize_sliced_report.md L10） | 1.082 |
| scratch conv-512 | 123.8（efficiency_report.md L11）、105.2（efficiency_prune.md L10） | 1.177（第二次比第一次快 15.0%） |
| DINOv2-S | 109.9（efficiency_report.md L12）、103.5（efficiency_prune.md L11） | 1.062 |

- 不同腳本的平均次數不同（§0.4），所以混在一起比會高估漂移。
- 只看同一支腳本、同一協定（measure_efficiency.py，20 次平均）在 07-01 和 07-12 兩次的量測，scratch student 還是差了約 15%。
- 論文 §4.7 寫的是「within about 7%」。Table 5 的加速倍數都取自同一 session，所以不受影響。

### 9.12 GT 直接微調（gtft）

- **設定**：pose 目標改成 CO3D GT 外參，由 `make_gt_labels.py` 產生 `gt_extrinsics.npy`，涵蓋全部 492 個場景，用 flip_xy；first-frame 正規化和 depth 監督不變。4 epochs、lr 1e-5，從各自的最佳權重繼續訓練（distill2/reports/gtft_report.md L5–7）。

| 模型 | 微調前 RRA@15 / RTA@15 / AUC@30 | GT 微調後 | 來源 |
|---|---|---|---|
| 842M | 0.964 / 0.893 / 0.876 | 0.960 / 0.898 / 0.856 | gtft_report.md L11；eval_gt_co3d10_pathb_gtft.md L7 |
| 464M | 0.883 / 0.814 / 0.767 | 0.883 / 0.821 / 0.770 | gtft_report.md L12；eval_gt_co3d10_ch512_gtft.md L7 |

- RRA@15 最大變化 0.004，AUC@30 最大變化 0.020，方向不一致。與論文 §4.3 相同（推算驗證）。
- **解讀不同**：gtft_report.md L16–18 下的結論是「差距來自容量／壓縮」；論文只說差距「unlikely to come from the teacher-label training signal」，沒有歸因於容量。以論文為準。
- **報告沒寫的副作用**：微調後 checkpoint 存的 vs-teacher depth 指標明顯崩壞。`runs/gtft/pathb/best.pt` 是 3.80°、RRA 1.000、abs-rel 0.914、δ1 ≈ 1.6e-7；`runs/gtft/sliced/best.pt` 是 5.32°、RRA 0.9375、abs-rel 0.907、δ1 ≈ 1.6e-7（8 場景）。紀錄沒有分析原因。這兩個模型沒有用在任何論文表格裡（gtft_report.md L19：「不更新主比較表」）。

### 9.13 三條流的零重訓組合（論文未收錄）

| 組合 | rot_err | RRA@15 | storage | 子集 | 來源 |
|---|---|---|---|---|---|
| agg0 + pe512 + ch1024 | 72.82° | 0.236 | 591M | 12 場景 | slicegpt_slice_combo.md L12 |
| agg256 + pe512 + ch1024 | 74.75° | 0.167 | 365M | 12 場景 | slicegpt_slice_combo.md L13 |

aggregator 不切（agg0），只把 front end 切到 512、camera trunk 切到 1024（一半寬），不重訓就已經塌縮（72.82°）。camera trunk 單獨切到 1024 的零重訓結果沒有量過，所以這一列不能把塌縮完全歸給 front end；front end 不可切的證據是 §4 的恢復實驗。

### 9.14 CPU-only latency（論文 Appendix D 有，這裡補設定細節）

- teacher 16233.7 ms，464M 7403.1 ms。比例 2.19×（推算 2.193）。
- 條件：fp32、8 threads、S=2，CPU 字串是「AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD」（distill2/reports/budget_incl_context_frozen.json L39–44）。
- 先暖機 1 次，再取 3 次平均（measure_budget_incl_context.py L140–146）。

### 9.15 視覺化圖（論文沒有使用）

distill2/viz/ 下有五張圖：camera_trajectory.png、rotation_histogram.png、depth_comparison.png、pointcloud.png、success_failure.png。由 distill2/visualize.py 產生，用的是 `runs/distill_b1/best.pt`，對照的塌縮模型是 `runs/abl/random/best.pt`（distill2/README.md L141–143）。各圖內容寫在 visualize.py L4–9：每幀旋轉誤差直方圖、teacher vs student 相機軌跡俯視圖、深度圖對照、由深度反投影的點雲（不是 point head 的輸出）、最好 / 最差場景的軌跡。論文沒有任何質性圖（點雲、深度圖、相機軌跡）。

---

## 10. 舊紀錄中與論文立場不同的說法

| 舊紀錄說法 | 出處 | 論文立場 |
|---|---|---|
| 466M | slicegpt_slice_combo.md L22、L28；slicegpt_d8_combo.md L34；gtft_report.md L12；measure_vram_curve.py L5；作者開發紀錄 | 464M / 464.4M |
| 加速 1.64× / 1.91×（384.6 → 233.9 / 201.1 ms） | distill2/README.md L286；作者開發紀錄 | 同一 session 的 1.77× / 1.93× |
| 「三者缺一不可」 | distill2/README.md L29、L290 | 不主張每個元件個別必要（§4.2） |
| INT8「零損失」 | distill2/README.md L289、L293；作者開發紀錄 | 不主張無損；GT 精度未知（§4.7） |
| GT 差距「來自容量／壓縮」 | gtft_report.md L18；作者開發紀錄 | 只說差距不太可能來自 teacher-label 訓練訊號（§4.3） |
| 「首個參數軸壓縮」 | 作者開發紀錄 | 論文沒有主張首創：§1、§2 指出參數層級壓縮已有 Distill3R 等工作；§1 的「to our knowledge」只用在「front end 不可壓縮」這個不對稱現象 |
| 壓縮率約 2.5× | distill2/CHANGES.md L62、L72 | 842M 是 1.49×（Table 3、§3.3） |
| 切片模型 GT RRA@15「較 Path B 低約 7 個百分點」 | distill2/README.md L295 | 0.964 → 0.883，差 8.1 個百分點（推算；論文 Table 2） |
| 早期進度文件「寬度不能動」 | 作者開發紀錄 | 已被後續的事後切片結果推翻（§4.5） |

---

## 11. 最佳結果一覽

除 CPU 列（fp32、S=2）與幀數列（掃 S）外，效率數字都是 RTX 5080、S=4、518×518、bf16。

| 項目 | VGGT-1B teacher | Teacher-initialized 842M | Sliced 464M（論文主模型） | Sliced 464M + INT8 |
|---|---|---|---|---|
| 參數 | 1256.5M（含 65.9M track head） | 842.2M | 464.4M（state_dict；參數 464.16M） | 464M（weight-only） |
| 序列化大小 | 5025 MB fp32（論文；推算下限約 5026 MB） | 3369 MB（只在紀錄） | 1858 MB | **537 MB**（相對 teacher 9.4×） |
| GFLOPs | 16814.3 | 9919.2 | 7496.8 | 7496.8 |
| Latency | 388.1 ms | 219.1 ms（1.77×） | **201.3 ms（1.93×）** | 229.9 ms（沒有 INT8 kernel，不加速） |
| VRAM（Default，不含 context） | 9.21 GB | 5.85 GB | 3.89 GB | **1.81 GB** |
| 16 GB 內最多幀數（Table 5；vram_curve*.json 格點，INT8 不在 Fig. 2） | 24 | 48 | 64 | 80 |
| 8 GiB 預算幀數（Default / Frozen） | 1 / 4 | 12 / 20 | 24 / 32 | 32 / 32 |
| 4 GiB 預算幀數（Default / Frozen） | 0 / 0 | 0 / 1 | 2 / 4 | 8 / 8 |
| CPU fp32、S=2 | 16.2 s | — | 7.4 s | — |
| vs teacher rotation error（RRA@15） | — | 4.32°（1.000）：消融配方 checkpoint，8 場景；主 checkpoint 2.87°（12 場景）/ 3.42°（8 場景） | 5.20°（0.969），8 場景 | 5.21°，8 場景 |
| vs teacher depth abs-rel | — | 0.189（消融 checkpoint，8 場景） | 0.193（8 場景，論文未列） | 0.193（8 場景，論文未列） |
| vs GT RRA@15 / RTA@15 / AUC@30（65 場景 × 10 幀） | 1.000 / 0.930 / 0.976 | 0.964 / 0.893 / 0.876 | 0.883 / 0.814 / 0.767 | 未量測 |
| vs GT 保留率 RRA@15 / AUC@30 | 100% | 96.4% / 89.8% | 88.3% / 78.6% | 未量測 |
| 本 repo 的權重檔 | —（用 `facebook/VGGT-1B`） | weights/vggt_teacher_init_842M.pt.part0 + .part1 | weights/vggt_sliced_464M_agg256_ch512.pt | weights/vggt_sliced_464M_int8.pt |

各列來源：

- 參數、GFLOPs、latency、VRAM、16 GB 幀數：§5
- 大小：§9.10，以及論文 §4.7
- 預算幀數：§7
- CPU：§9.14
- vs teacher：§1、§4、§9.5、§9.10
- vs GT：§2
- 權重檔：§0.5、[`weights/README.md`](../weights/README.md)

其他可參考的點：

| 模型 | 參數 | vs teacher | vs GT RRA@15 / AUC@30 | Latency | 來源 |
|---|---|---|---|---|---|
| agg k=384 + 恢復 | 654M | 4.18°（8 場景） | 0.905 / 0.792 | — | §2、§4 |
| d8 剪枝 | 741M | 9.45°（8 場景） | 未量測 | 180.1 ms | §3 |
| d8 + agg256 + ch512 + 恢復 | 439M | 9.08°（8 場景） | 0.645 / 0.564 | 178 ms（另一 session） | §9.9 |
| 842M + INT8 | 842M | 3.42°（8 場景） | 未量測 | 237.8 ms（quantize.py） | §9.10 |
