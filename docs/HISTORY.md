# 研究歷程（HISTORY）

這份文件按時間順序記錄本專案的經過：第一個失敗的 student、找出 camera-token 塌縮、修掉的 bug、試過但沒成功的方法、teacher-init（Path B）、層剪枝、光譜探針、SliceGPT 式切片與恢復蒸餾、INT8，一直到 camera-ready 前的數字更正。結果主表見 [RESULTS.md](RESULTS.md)，重現指令見 [REPRODUCE.md](REPRODUCE.md) 與 [`distill2/README.md`](../distill2/README.md)。這裡記的是「為什麼最後會走到這條管線」。

---

## 閱讀說明

- **日期從哪裡來**：以作者開發時 `distill2/` 的 git commit 紀錄為準（例如 `48f0412`；這些 commit 編號屬於作者的開發 repo，不在本公開 repo 的歷史裡），其次是報告檔與存檔的修改時間。記錄沒寫日期的事件，只標出前後順序。
- **論文內容的出處**：以 camera-ready 版論文的文字為準（論文 PDF 待 Springer 出版，沒有收錄在本 repo）。文中寫成（論文 §x.x／Table n／Appendix X）。
- **（作者開發紀錄）**：只存在於作者私人筆記或開發 repo、沒有收錄在本 repo 的資料。文中只保留已經對照原紀錄核實過的內容。
- **內部用語**：「Path B」＝論文的 teacher-initialized student（842M）；「P0」＝內部的實驗編號（`distill2/README.md` 開頭）。
- **兩種 RRA@15，名字一樣但不是同一個指標**：
  - **vs teacher**：`distill2/eval.py` 的 `eval_scene`。每個場景取等距 12 幀，算每一幀的旋轉誤差（位姿都以第 0 幀為基準座標，所以是每幀相對第 0 幀的旋轉），統計小於 15° 的比例，比對對象是 teacher soft label。論文 Table 1、3、4 用的是這個。
  - **vs GT**：`distill2/eval_gt.py`。用相對位姿配對，比對 CO3D 官方標註。論文 Table 2 用的是這個。
- **評估子集**：`quick_eval` 取 test split 依名稱排序後的**前 N 個場景**（`distill2/eval.py`）。前 8 個場景是 4 個類別（apple、backpack、ball、banana），前 12 個是 7 個類別。訓練腳本 `train.py`、`train2.py`、`train_sliced.py`、`run_ablation.py`、`run_layer_pruning.py` 預設 `--eval_scenes 8`；`slicegpt_probe.py` 與 `slicegpt_slice.py` 預設 12。
- **`推算`**：由記錄裡的數字自行算出、記錄本身沒寫的值。**`推測`**：依腳本預設值或時間順序推斷、記錄沒有直接寫明的對應關係。
- **checkpoint 與公開權重**：論文用到的三個模型放在 `weights/`（Git LFS，見 [weights/README.md](../weights/README.md)），都是訓練時存下的原檔：
  - `vggt_teacher_init_842M.pt.part0`／`.part1`：`runs/distill_b1/best.pt`（3,368,981,430 bytes；GitHub LFS 單檔上限 2 GB，所以切成兩段，用 `cat` 或 `copy /b` 合併）。
  - `vggt_sliced_464M_agg256_ch512.pt`：`runs/sliced/recover_agg256_ch512/best.pt`（論文主模型）。
  - `vggt_sliced_464M_int8.pt`：`runs/sliced/quantized_sliced_int8.pt`（INT8 weight-only）。

  `runs/`、`abl/` 底下其他的 checkpoint（消融、剪枝、其他切片設定、二次蒸餾等）不在 repo。文中凡引用存檔內 `epoch`、`eval` 欄位或元素數的地方，都是 2026-10-17 只讀取 pickle metadata（不載入權重）得到的，寫成「checkpoint metadata」。
- **路徑**：資料位置一律寫成 `<DATA_ROOT>`（teacher soft labels，train 427／test 65）與 `<CO3D_ROOT>`（CO3D 官方標註）。在 `distill2/` 底下執行時，權重路徑是 `../weights/...`。

---

## 時間軸總覽

| 日期 | 階段 | 關鍵結果 | 主要來源 |
|---|---|---|---|
| 2026-06-04 以前 | 舊管線：conv-512 student 從零蒸餾 | depth 學得起來，rotation 塌縮在 82.35° | `distill2/reports/eval_results_before.json` |
| 2026-06 上旬 | 修 bug：四元數慣例、EMA、import 路徑等 | bug 都是真的，但 loss 修正後 rotation 仍約 82° | `distill2/CHANGES.md` §3(A) |
| 2026-06-07～06-09 | 重建 `distill2/`，Path B（teacher-init） | rot_err 4.89°（65 場景，vs teacher；20 epoch 的 run） | `distill2/CHANGES.md` §4；`distill2/README.md` §四、§四之二；commit `48f0412` |
| 2026-07-01～07-07 | P0 實驗：效率、INT8、消融、視覺化、二次蒸餾 | 只有全複製加凍結前端成功；二次蒸餾塌縮 | `distill2/reports/ablation_report.md` 等；commit `a2e2ead` |
| 2026-07-11 | SVD 投影初始化（conv-512） | 塌縮 | `distill2/proj_init.py`；checkpoint metadata；作者開發紀錄 |
| 2026-07-12 | 層剪枝、光譜探針、SliceGPT 切片加恢復蒸餾 | 剪到 8 層仍可用；k=256 恢復後 4.95° | `distill2/reports/{layer_pruning_report,slicegpt_probe,slicegpt_slice}.md`；commit `30d55a5` |
| 2026-07-13～07-14 | 三條流擴展、前端負面結果、trunk 切到 1/4、INT8 疊加 | 464.4M @ 5.20°；537 MB @ 5.21° | `distill2/reports/{slicegpt_slice_combo,quantize_sliced_report}.md`；commits `c314bcd`、`faf9828` |
| 2026-07-21～08-09 | CO3D GT 評估（20×8 再到 65×10）、隨機取幀、直接用 GT 微調、剪枝×切片 | GT RRA@15：Path B 0.964、最終模型 0.883 | `distill2/reports/eval_gt_co3d10_*.md`、`gtft_report.md`、`slicegpt_d8_combo.md` |
| 2026-08-17～08-28 | VRAM 曲線、同一 session 的統一量測 | 論文 Table 5 的延遲與 VRAM | `distill2/reports/{vram_curve,vram_curve_int8,unified_bench}.json`；commits `dc2435e`、`691c186` |
| 2026-10-02～10-03 | 論文修訂時補做：含 CUDA context 的記憶體預算、GFLOPs、CPU 計時 | 論文 Table 7、Appendix D | `distill2/reports/budget_incl_context_*.json`、`flops_extra.json`；commit `a408314` |
| 2026-10-17 | camera-ready 前核對 | 464.4M、評估場景 8 vs 12、40 epochs、延遲漂移 | checkpoint metadata；`distill2/README.md` §四、§四之二、§五之三、§八 |

---

## 階段 0：舊管線與 camera-token 塌縮（2026-06-04 以前）

### 設定

- **資料**：用 VGGT-1B 對 CO3D 離線推理一次，把輸出存成 soft label，每個場景有 `images/`、`depth.npy`、`depth_conf.npy`、`extrinsics.npy`、`intrinsics.npy`、`pose_enc.npy`、`point_mask.npy`、`world_points.npy`、`meta.json`（來源：`distill2/README.md` §三）。由 `scripts/export_co3d_labels.py` 產生，單一場景版本是 `scripts/export_teacher_labels.py`。
- **切分**：train 427／test 65 場景（來源：`distill2/CHANGES.md` §1），共 33 個類別，訓練時完全不用 GT（來源：論文 §4.1）。
- **舊訓練管線**：Hydra／trainer 架構，`CHANGES.md` 稱為 `training/`，後來整套被 `distill2/` 取代（來源：`distill2/CHANGES.md` §3(D)、§5）。`training/` 沒有收錄在本 repo（`distill2/CHANGES.md` 開頭的說明）。

### 第一個 student 與觀察

第一個 student 用 conv patch embed、`embed_dim` 512、12 層、8 個 heads，共 146,432,920 個參數，從零開始蒸餾（來源：`distill2/reports/eval_results_before.json`；論文 §3.2）。在 10 個場景（test split 依名稱排序的前 10 個）上的平均：

| 指標 | 值 | 來源 |
|---|---|---|
| mean_rot_err | 82.35° | `distill2/reports/eval_results_before.json` |
| RRA@15（vs teacher，逐幀） | 0.098 | 同上 |
| abs_rel／δ1（depth vs teacher） | 0.238／0.620 | 同上 |
| 壓縮比 | json 寫 6.83×，因為以 `teacher_params_approx_B = 1.0` 估算；用實測 teacher 1256.5M 計算是 8.6× | `eval_results_before.json`；`distill2/reports/efficiency_report.md` |

論文把這次 run 寫成 82.3°／RRA@15 0.098／abs-rel 0.238，並明說它是「preliminary diagnostic」：用較早的 script 在 10 個場景上評估，不能和 Table 1–4 直接比較。同一套 pipeline 內可比較的塌縮參考值，是 Table 1 的 *random* 列（來源：論文 §3.2）。

`distill2/CHANGES.md` §4 的「起點（塌縮）」欄寫的是另一組數字：82.7°、RRA@15 0.135、abs_rel 1.04、δ1 0.40。記錄沒說明這組數字來自哪一個 run，因此不和上表混用。同一張表常被引用成「82.7° → 4.9°」，其中 4.89° 一側來自 20 epoch 的 `runs/distill_b1_full`（best 在 epoch 索引 19），不是論文用的 `runs/distill_b1/best.pt`（見[階段 2](#主-run-到底是幾個-epoch記錄與論文不一致)）。

### 診斷結果

直接載入訓練好的 student 檢查，發現（來源：`distill2/CHANGES.md` §2；論文 §3.2）：

- 每一幀輸出的四元數幾乎相同，都接近 identity。跨幀標準差約 0.01，teacher 約 0.5。平移也一起塌縮。
- aggregator 送給 camera head 的 camera token，**跨幀 cosine similarity 是 0.998**。
- 存檔裡的 `aggregator.camera_token` 參數幾乎沒離開初始值。

這種現象後來命名為 **camera-token collapse**：pose 路徑收斂到一個「不看輸入、一律輸出平均 pose」的退化解，旋轉誤差約 82°。depth 不受影響。當時的解釋是：depth/point head 經由大型 DPT decoder 讀取全部 (518/14)² 個 patch token，而一幀的相機預測只靠**單一個 token**（來源：`distill2/CHANGES.md` §2；論文 §3.1）。論文也指出，逐幀的 pose loss 本身不懲罰「跨視角不變」的表徵，所以退化解不會被 objective 排除（來源：論文 §3.2）。

當時的另一個判斷是「8.6× 小的模型容量不夠表示視角」（來源：`distill2/CHANGES.md` §2；論文 §3.2）。這個判斷後來被光譜探針推翻，見[階段 6](#階段-6光譜探針2026-07-12)。

---

## 階段 1：找到並修掉的 bug，以及同期的救援嘗試（2026-06 上旬）

### Bug 清單

| # | 問題 | 影響 | 修正 | 現在的位置 |
|---|---|---|---|---|
| 1 | 相對旋轉項把 VGGT 的 **scalar-last `(x,y,z,w)`** 四元數當成 scalar-first | 相對旋轉的梯度變成噪音 | `quat_mul`／`quat_conj` 改成 scalar-last，實部在 index 3 | `distill2/loss.py` 的 `_quat_mul`、`_quat_conj`；慣例見 `vggt/utils/rotation.py`（"XYZW … scalar-last"） |
| 2 | 旋轉 loss 用 `arccos` | 接近正解時梯度爆衝 | 改用 `1 − \|dot\|`。絕對值處理 q ≡ −q 的雙重覆蓋，梯度有界 | `distill2/loss.py` `camera_losses`；論文 Appendix B |
| 3 | rotation 和 translation、FoV 共用一個正規化 | rotation 被收斂很快的 T/FoV 稀釋 | rotation 與 (T+FoV) 各用一個 EMA 正規化（momentum 0.99，warmup 100 步）。權重 rot 4.0、T+FoV 2.0、depth 1.0 | `distill2/loss.py` `DistillLoss`；論文 Appendix B |
| 4 | loss 模組的 EMA buffer 放錯 device | 舊 trainer 沒有對 loss 呼叫 `.to(device)` | 修正 device。`distill2` 版改成在 `forward` 裡把 buffer 搬到資料所在的 device | `distill2/CHANGES.md` §3(A)4；`distill2/loss.py` `forward` |
| 5 | early stopping 監看 `loss_camera_kd` | T/FoV 先出現假性 plateau，訓練太早停 | 改成監看總 objective，並放寬 patience | `distill2/CHANGES.md` §3(B) |
| 6 | **eval 載到錯版的 `vggt` 套件** | `sys.path` 順序讓 eval 載入舊版 vggt（camera head `trunk_depth=2`、只支援 conv），訓練用的 4 層 camera head 被截成 2 層，旋轉評估失真 | 移除 `sys.path.insert`。`load_student` 依 checkpoint 自動偵測架構（conv／DINOv2、`embed_dim`、camera head 層數） | `scripts/eval_student_vs_teacher.py` 檔頭註解與 `load_student`；`distill2/CHANGES.md` §3(C) |

第 6 項的背景：作者本機同時有兩份 `vggt` 套件。一份是 editable install，支援 DINOv2 patch embed、camera head `trunk_depth=4`，訓練實際用的是這份；另一份是專案目錄裡的舊版（`trunk_depth=2`、只有 conv）（來源：`scripts/eval_student_vs_teacher.py` 檔頭註解）。當時的規則是：改模型、head、aggregator 要改 editable install 那份；改訓練或 loss 要改 `distill2/`（作者開發紀錄）。公開 repo 的 `vggt/` 是前者（`vggt/models/vggt_student.py` 的 camera head 寫死 `trunk_depth=4`，註解為「對齊 teacher」），用 `pip install -e .` 安裝即可，不需要另一份（`distill2/README.md` §二）。

**結果**：這些都是真的 bug，但 loss 的修正做完以後 rotation 仍然約 82°（來源：`distill2/CHANGES.md` §3(A) 結尾）；記錄沒有留下 early stopping 與 eval 修正後的 rotation 數字；最後是整套改寫成 `distill2/` 才解決（來源：`distill2/CHANGES.md` §3(D)）。論文的說法是：四元數慣例與 EMA 權重這兩個細節，是「讓 rotation 訊號能訓練所必需的」（來源：論文 §3.3、Appendix B）。合起來的意思是：**必要，但不足以解決塌縮**。

### 同期嘗試：DINOv2-S（384）從零蒸餾

把 patch embed 換成預訓練的 DINOv2 ViT-S/14（`dinov2_vits14_reg`，`embed_dim` 384，6 heads），同樣從零蒸餾，結果仍約 82°，塌縮（來源：`distill2/CHANGES.md` §3(B) 提到改用 DINOv2 patch embed 的實驗；結果數字為作者開發紀錄）。模型 110.7M、相對 teacher 11.3×（來源：`distill2/reports/efficiency_report.md`）。架構定義見 `vggt/models/vggt_student.py` 檔頭的規格表（檔頭註明那張表是早期的估計，表中參數量 ~96M 與實測不同，以 `efficiency_report.md` 為準）。

---

## 階段 2：重建 `distill2/` 與 Path B（2026-06-07～06-09）

### 做法

整套舊 Hydra 管線不再修補，改寫成乾淨、自包含的 `distill2/`（來源：`distill2/CHANGES.md` §3(D)）。核心想法：pose 表徵「學起來很貴，複製很便宜」，所以 student 直接**繼承** teacher 的權重，不再從零學起，精神同 DistilBERT（來源：`distill2/CHANGES.md` §3(D)；論文 §3.3）。記錄裡只出現「Path B」這個名稱，沒有說明對應的 Path A 是什麼。

`distill2/build_student.py --copy all` 的權重對應如下（來源：`distill2/README.md` §四之二）：

| student | 對應的 teacher 權重 | 張量數 | 備註 |
|---|---|---|---|
| `aggregator.patch_embed`（DINOv2 ViT-L/14-reg） | 同名，整個複製 | 344 | 訓練時凍結 |
| `aggregator.camera_token`／`register_token` | 同名 | 1 + 1 | |
| `aggregator.frame_blocks[i]`（i = 0…11） | `frame_blocks[2i]` | 216 | 每隔一層取一層 |
| `aggregator.global_blocks[i]` | `global_blocks[2i]` | 216 | 同上 |
| `camera_head` | 同名，整個複製 | 69 | 維度與 teacher 相同，pose 能力直接繼承 |
| `depth_head`／`point_head` | 不複製，隨機初始化 | 62 + 62 | 由蒸餾學 |

結果是 842.2M 參數，相對 teacher 1256.5M 為 1.49×（來源：`distill2/reports/efficiency_report.md`；論文 §3.3）。`distill2/CHANGES.md` §3(D) 與 §4 寫的「壓縮率約 2.5×」和實測參數不符（1256.5／842.2 = 1.49，`推算`），記錄沒說明 2.5× 怎麼來的，本 repo 一律用 1.49×（1.5×）。track head 在所有 student 中都被移除；point head 保留但沒有任何 loss 監督，也沒評估過（來源：論文 §3.1、Appendix A、Appendix B）。

**訓練配方**（`distill2/train.py`）：AdamW lr 5e-5、weight decay 0.05、5% warmup 後 cosine decay、bf16 autocast、梯度裁剪 1.0、凍結 patch embed（來源：`distill2/train.py`；論文 §4.1）。論文寫每個序列 2 幀（論文 §4.1）；`train.py` 的 `--img_per_seq` 預設是 4，而 checkpoint 沒有存訓練參數，所以 `runs/distill_b1` 實際用了幾幀無法核對（記錄只確定 20 epoch 的 `runs/distill_b1_full` 用 `--img_per_seq 2`，見 `distill2/README.md` §四）。`student_init.pt` 建立於 2026-06-07、最後寫入 06-08；Path B 在 2026-06-09 commit（來源：`distill2/README.md` §四之二；commit `48f0412`）。

### 主 run 到底是幾個 epoch（記錄與論文不一致）

- **論文**：「the main run is 20 epochs of 500 steps」（來源：論文 §4.1）。
- **記錄**：後續所有腳本預設使用的 `runs/distill_b1/best.pt`（即 `weights/vggt_teacher_init_842M.pt`，合併兩段後），存檔內 `epoch = 37`；同目錄 `last.pt` 是 `epoch = 39`。`epoch` 是從 0 起算的迴圈索引（`distill2/train.py`），所以這是一次 **40 epoch** 的訓練，best 出現在第 38 個 epoch（來源：checkpoint metadata，`推算`）。`distill2/train.py` 的 `--epochs` 預設值正是 40，`distill2/README.md` §四也把這個 run 記為 train.py 預設（40 epochs）。作者早期的操作筆記則把它的指令寫成 `--epochs 20 --eval_every 2 --img_per_seq 2`（作者開發紀錄），這組設定和下面 `runs/distill_b1_full` 的設定幾乎相同。
- 另有一個 `runs/distill_b1_full/`，是 20 epoch 的 run（`--epochs 20 --eval_every 2 --img_per_seq 2 --throttle 0.1`；best 與 last 的 `epoch` 都是 19；存檔內 quick_eval 為 4.44°、RRA@15 0.958，8 個場景）。`distill2/CHANGES.md` §4 的 65 場景 4.89° 就是這個 run（來源：`distill2/README.md` §四、§七；checkpoint metadata）。切片、GT 評估、量化、效率等腳本都沒有用它當預設 checkpoint（`slicegpt_probe.py`、`slicegpt_slice.py` 的 `--ckpt`、`train_sliced.py` 的 `--base_ckpt` 預設，以及 `measure_vram_curve.py`、`measure_unified_bench.py`、`measure_budget_incl_context.py` 寫死的路徑，都是 `runs/distill_b1/best.pt`），它也不在 `weights/`。
- 這個不一致也列在[階段 13](#階段-13camera-ready-前的核對與更正2026-10-17) 第 3 項。

### 結果

| checkpoint | 評估 | 值 | 來源 |
|---|---|---|---|
| `runs/distill_b1_full`（20 epochs） | vs teacher，全部 65 個 test 場景 | rot_err 4.89°、RRA@15 0.968、abs_rel 0.176、δ1 0.698 | `distill2/CHANGES.md` §4；`distill2/README.md` §四、§七 |
| 同上 | vs teacher，前 8 個 test 場景 | rot_err 4.44°、RRA@15 0.958 | 存檔內 `eval`（checkpoint metadata） |
| `runs/distill_b1`（40 epochs，論文的 842M） | vs teacher，前 8 個 test 場景 | rot_err 3.42°、RRA@15 1.000、abs_rel 0.177 | `distill2/reports/quantize_report.md`；存檔內 `eval` |
| 同上 | vs teacher，前 12 個 test 場景 | rot_err 2.87°、RRA@15 1.000 | `distill2/reports/slicegpt_probe.md` |
| 同上 | 相對位姿 vs teacher（20 場景 × 8 幀） | RRA@15 0.984、RTA@15 0.977、AUC@30 0.888 | `distill2/reports/eval_gt_report.md`（2026-07-02；指令見 `distill2/README.md` §五） |

**同一個 `runs/distill_b1` checkpoint，換評估子集就從 3.42°（前 8 個）變成 2.87°（前 12 個）；20 epoch 的 run 也有 4.44°（前 8 個）與 4.89°（全部 65 個）兩個值。**後面所有跨表比較都要注意這點。

### 權重複製的事後驗證（2026-10-17）

`distill2/verify_teacher_copy.py` 只讀 checkpoint 與 teacher 的 safetensors，不需要 GPU（來源：`distill2/README.md` §四之二）：

- `student_init.pt`：應複製的 847 個張量和 teacher **逐位元相同**；depth/point head 的 124 個張量不同，符合「不複製」的設計。
- `runs/distill_b1/best.pt`：`patch_embed` 的 344 個張量仍然相同，證實凍結有生效；其餘 503 個已被訓練更新。

`student_init.pt` 不在 repo。`runs/distill_b1/best.pt` 就是 `weights/` 裡的 842M 模型，合併兩段後可以在 `distill2/` 下用 `python verify_teacher_copy.py --ckpt ../weights/vggt_teacher_init_842M.pt` 自行重驗。

---

## 階段 3：P0 實驗與二次蒸餾（2026-07-01～07-07）

這一段是指導教授要求的 P0 實驗（作者開發紀錄），2026-07-07 整理 commit（`a2e2ead`）。

### 3.1 效率量測（第一版，2026-07-01）

RTX 5080、S=4、518×518、bf16、20 次平均（來源：`distill2/reports/efficiency_report.md`）：

| 模型 | Params | GMACs（fvcore） | VRAM | Latency | 壓縮比 |
|---|---|---|---|---|---|
| Teacher VGGT-1B | 1256.5M | N/A | 9.22 GB | 384.6 ms | 1.0× |
| Student-conv (512) | 146.4M | 738.5 | 1.76 GB | 123.8 ms | 8.6× |
| Student-DINOv2-S (384) | 110.7M | 668.8 | 1.52 GB | 109.9 ms | 11.3× |
| Path B (1024) | 842.2M | 3666.1 | 5.86 GB | 233.9 ms | 1.5× |

FLOPs 由 fvcore 以 MAC 估算，報告自己註明「注意力等自訂算子可能低估，僅供相對比較」。論文最後改用 PyTorch FLOP counter 的 GFLOPs（見[階段 12](#階段-12論文修訂時補做的量測2026-10-022026-10-03)），兩者不可混用。

### 3.2 Path B 的 INT8（2026-07-01）

torchao int8 weight-only，8 個場景（來源：`distill2/reports/quantize_report.md`）：

- 檔案 3369 MB → 915 MB（3.68×），rot_err 3.42° → 3.42°，abs_rel 0.177 → 0.177。
- latency 反而變慢：214.8 ms → 237.8 ms。
- 當時的定位：「不改架構、不碰塌縮風險，是壓縮的保底方案」（來源：`distill2/quantize.py` 檔頭）。
- 這個 842M 的 INT8 檔不在 repo（`weights/` 的 INT8 檔是 464M 切片模型的，見[階段 9](#階段-9在切片模型上疊加-int82026-07-14)）。

### 3.3 消融：要繼承哪些元件（2026-07-01～07-02）

六種組合寬度相同（1024，約 842M），只差在初始化方式與前端是否凍結，由 `distill2/run_ablation.py` 執行，每 epoch 400 步，每組訓練 12 個 epoch（`distill2/README.md` §五 的指令是 `--epochs 12`，`run_ablation.py` 預設是 15；六組 `last.pt` 的 epoch 索引都是 11，checkpoint metadata），eval 用前 8 個 test 場景（來源：`distill2/reports/ablation_report.md`；`run_ablation.py` 預設值）：

| 組合 | 複製元件 | 凍結前端 | rot_err | RRA@15 | abs_rel | δ1 |
|---|---|---|---|---|---|---|
| random | 無 | ✓ | 76.18° | 0.188 | 0.235 | 0.642 |
| patch_only | patch_embed | ✓ | 75.59° | 0.198 | 0.260 | 0.616 |
| agg_only | aggregator | ✓ | 75.58° | 0.208 | 0.265 | 0.574 |
| head_only | camera_head | ✓ | 75.48° | 0.208 | 0.247 | 0.578 |
| **full** | 三者全部 | ✓ | **4.32°** | **1.000** | 0.189 | 0.746 |
| full_ft_pe | 三者全部 | ✗ | 75.68° | 0.219 | 0.248 | 0.632 |

注意事項：

- random、agg_only、head_only 三列的前端，是 hub 上官方預訓練的 DINOv2，並且凍結，不是隨機初始化。patch_only 列複製的是 teacher 那份經 VGGT 訓練調整過的前端，兩者權重不同（來源：`distill2/run_ablation.py` 檔頭；論文 Table 1 caption）。
- `full` 的 best 在 epoch 索引 11，也就是最後一個（第 12 個）epoch（來源：`distill2/reports/ablation_report.csv` 的 `epoch` 欄）。論文寫消融與剪枝用「12–15 epochs of 400 steps」（來源：論文 §4.1）。
- **4.32° 和 Path B 主 run 不是同一個 checkpoint**：4.32° 來自 `runs/abl/full`（短配方、8 場景）。論文 Table 4 的 2.87° 是完整訓練的 `runs/distill_b1`，在 12 個場景上評估（來源：論文 Table 4 caption）。
- **內部記錄和論文的說法強度不同**：`distill2/README.md` §一、§七 寫「三者缺一不可」。論文只主張「單一元件不夠、三者同時繼承可以成功」，並明說沒跑三種兩兩組合，**不宣稱每個元件各自必要**（來源：論文 §4.2、Limitations）。
- 把前端解凍（full_ft_pe）就塌回 75.68°。因此「前端必須凍結」成為後續所有恢復訓練的固定規則（來源：`distill2/train_sliced.py` 檔頭）。

### 3.4 視覺化（2026-07-02）

`distill2/visualize.py` 產生 `distill2/viz/` 的五張圖：深度對照、點雲、相機軌跡、旋轉誤差直方圖、最好與最差場景。塌縮對照模型用的是 `runs/abl/random/best.pt`（來源：`distill2/README.md` §五）。論文本身沒有收錄任何定性圖（論文只有 Fig. 1 流程總覽與 Fig. 2 記憶體曲線兩張圖）。

### 3.5 二次蒸餾：把 Path B 當 teacher 訓練小 student（2026-07-01，失敗）

目的：拿回 110–150M 等級的壓縮率，同時避免塌縮（來源：`distill2/train2.py` 檔頭）。student 是 DINOv2-S（384）；Path B 當線上 teacher，提供 camera token。

- **camera-token feature KD**：約 75°，塌縮（作者開發紀錄）。
- **加上跨幀結構 KD**（兩兩 cosine 矩陣對齊 teacher）：約 77°，塌縮（作者開發紀錄）。
- 存檔佐證：`runs/distill2_s` 與 `runs/distill2_s2` 都訓練了 40 個 epoch（`last.pt` 的 epoch 索引都是 39），best 分別是 75.28°（epoch 索引 9）與 76.75°（epoch 索引 1），eval 為 8 個場景（來源：checkpoint metadata）。兩者與上面約 75°／約 77° 的對應關係是`推測`。
- **設計上的教訓**（來源：`distill2/loss2.py` 檔頭）：第一版把 projector 放在 student 端，把 student token 投影後去對齊 teacher。結果可學的 projector 會把塌縮的 token「硬投影成相似」，loss 下降了，但 camera head 實際讀的原始 token 仍然塌縮。之後改成 projector 放在 teacher 端，約束直接作用在原始 student token 上，並加入跨幀結構項，結果依然塌縮。

### 3.6 當時的結論（後來被推翻）

當時的進度紀錄寫：「『輕量』與『pose 學得起來』目前是衝突的：輕的塌、能 work 的不夠輕。」並把架構級壓縮列為受限（1.5×），**寬度不能動**（作者開發紀錄）。寬度的部分在[階段 6](#階段-6光譜探針2026-07-12)、[階段 7](#階段-7slicegpt-式切片與恢復蒸餾2026-07-12)被推翻。

---

## 階段 4：SVD 投影初始化（2026-07-11，失敗）

**假設**：conv-512 從零訓練會塌縮。如果改用已經成功的 Path B 權重，以 SVD 投影到 512 維來初始化，或許能在 8.6× 壓縮下不塌縮（來源：`distill2/proj_init.py` 檔頭）。

**做法**：取所有 block「寫入 residual 的權重」（`attn.proj` 與 `mlp.fc2`）的 SVD 左奇異向量前 512 個，組成共用投影 P。每個 block 以 W' = P W Pᵀ 投影；MLP hidden 維度另外逐 block 用 SVD 降維；camera head 用同樣方法從 2048 降到 1024。LayerNorm 與 LayerScale 的投影是近似，檔頭明寫「這是探路實驗，不是精確等價」（來源：`distill2/proj_init.py`）。

**結果**：約 77°，塌縮（作者開發紀錄）。存檔 `runs/proj512/best.pt` 記錄的 best 是 75.52°，出現在 8 個 epoch 中的 epoch 索引 1（`last.pt` 的 epoch 索引是 7）（來源：checkpoint metadata）。

到這裡，作者的紀錄總結出「**5 種變窄/變小訓練策略全塌**」：random 窄初始化、DINOv2-S 從零、feature KD、結構 KD、SVD 投影初始化（作者開發紀錄）。

---

## 階段 5：層剪枝，固定寬度只砍深度（2026-07-12）

寬度看起來不能動，於是改試深度。寬度保持 1024；teacher 24 層，以 stride `j = i·(24 // depth)` 取層；其餘配方和消融完全相同（`--copy all`、凍結 patch embed、12 epochs × 400 steps），以隔離「層數」這個單一變因（來源：`distill2/run_layer_pruning.py` 檔頭；`distill2/reports/layer_pruning_report.md`）。

| 層數 | Params | 壓縮比 | rot_err | RRA@15 | abs_rel | Latency（2026-07-12） | Latency（2026-08-28 統一量測） |
|---|---|---|---|---|---|---|---|
| 12（基準） | 842M | 1.49× | 4.32° | 1.000 | 0.189 | 222.8 ms | 219.1 ms |
| **8** | 741M | 1.69× | **9.45°** | 0.781 | 0.201 | 186.2 ms | 180.1 ms |
| 6 | 691M | 1.82× | 56.78° | 0.208 | 0.214 | 165.3 ms | 161.4 ms |
| 4 | 641M | 1.96× | 75.74° | 0.198 | 0.227 | 145.9 ms | 144.0 ms |

（來源：`distill2/reports/layer_pruning_report.md`、`efficiency_prune.md`、`unified_bench.json`。12 層那一列的精度是消融的 `runs/abl/full`；08-28 的 219.1 ms 是 `measure_unified_bench.py` 用 `runs/distill_b1` 量的，兩者架構相同。）

- **深度門檻是 8 層**：8 層可用，6 層接近塌縮，4 層回到塌縮。depth 指標只是緩慢變差（0.189 → 0.227），可見門檻完全由 pose 路徑決定（來源：論文 §4.4）。
- 8 層版延長到 24 epoch，rot_err 是 9.94°，沒有突破約 9.5–10° 的平台（作者開發紀錄；`runs/prune/d8_long/best.pt` metadata：epoch 索引 23、9.94°、RRA@15 0.8125）。
- 論文沒有對剪枝模型做 GT 評估（論文 Table 2 只有 teacher、842M 與兩個切片模型）。

---

## 階段 6：光譜探針（2026-07-12）

在投入費工的權重手術前，先用 forward hook 便宜地模擬 SliceGPT 切片後模型能保留的資訊上限：把每個 aggregator block 的輸出投影到該 block 自己 activation PCA 的前 k 維，y ↦ μ + (y−μ) Q_k Q_kᵀ，完全不動權重。腳本檔頭寫明判準：k≈512 仍可用，就值得做完整的權重手術；k≈896 就崩，表示 pose 路徑真的需要全寬度，可以「省下數天工程」（來源：`distill2/slicegpt_probe.py` 檔頭）。

校準用 8 個 train 場景（每場景 6 幀），eval 用前 12 個 test 場景，模型是 `runs/distill_b1/best.pt`（來源：`distill2/reports/slicegpt_probe.md`；`slicegpt_probe.py` 預設值）：

| k | 平均 PCA 能量 | rot_err | RRA@15 |
|---|---|---|---|
| 不截斷／1024 | 100.00% | 2.87° | 1.000 |
| 896 | 99.93% | 2.99° | 1.000 |
| 768 | 99.82% | 3.06° | 1.000 |
| 640 | 99.65% | 3.22° | 0.993 |
| 512 | 99.38% | 3.45° | 0.993 |
| 384 | 98.91% | 3.86° | 0.986 |
| **256** | 97.98% | **4.91°** | 0.958 |

**解讀**：殘差流的 1/4 就保留了 97.98% 的 activation 能量，代價約 2°（2.87° → 4.91°），表示 pose 資訊並沒有分散在全部 1024 維。由此推論：先前「變窄就塌」的主因是**優化**（從窄模型開始學學不到），而不是**容量**不足（來源：論文 §3.4、§4.5）。論文對這個結論用的是「Together these indicate」，屬於指向性證據，不是證明（來源：論文 §4.5）。

探針的基底是**逐 block、有中心化**的，只用來診斷，和後面切片實際用的基底不同；Table 4 的能量欄描述的是探針基底，不是切片基底保留的能量（來源：論文 §3.4、§3.5）。

---

## 階段 7：SliceGPT 式切片與恢復蒸餾（2026-07-12）

### 權重手術

`distill2/slicegpt_slice.py`（來源：檔頭；論文 §3.5、Appendix C）：

- 用 activation **未中心化二階矩**的特徵基底當旋轉 Q。一條殘差流的所有 block **共用一個 Q**，這樣旋轉可以直接摺進權重，residual 路徑上不需要額外的矩陣（論文的說法是「摺疊需要一個共同座標系」，見下方與原版 SliceGPT 的比較）。
- LayerNorm 改成 `SlicedNorm`（只計算 σ），中心化、γ、β 摺進下一個 Linear。
- LayerScale 摺進 `attn.proj`／`mlp.fc2`。
- head 介面用儲存的 Q_kᵀ 把每個 tap de-slice 回 1024 維，head 本體不動。
- **等價性測試**：k 取滿維、Q 用隨機正交矩陣時，切片後的模型必須重現原模型的輸出。aggregator 單流的 max|Δ| ≈ 3e-7（作者開發紀錄）。這個測試之後當作每條摺疊規則的 regression test（`python slicegpt_slice.py --check`，CPU 即可）。

與原版 SliceGPT（Ashkboos et al., ICLR 2024；論文參考文獻 [17]）的差異：原版是每個 block 各用 PCA 算一個正交矩陣，並在 residual 連接上加一個 Q_{l−1}ᵀQ_l；本專案是每條流共用一個 Q，residual 路徑上不需要額外的矩陣。論文 §2 把 SliceGPT 寫成「applying one globally shared orthogonal rotation to the residual stream」，和原論文的做法不符；`distill2/slicegpt_probe.py` 檔頭對 SliceGPT 的描述（「逐層 Q」）才是原版的做法。論文 §1、§3.5 說「把旋轉摺進權重需要一個共同座標系」也因此說得太滿：那是本專案「不在 residual 上加矩陣」這個選擇的前提，不是摺疊本身的必要條件。

校準用 train split 依名稱排序後的前 8 個場景，全部屬於 *apple* 類別，與所有評估場景都不重疊（來源：論文 §3.5）。

### 不重訓的結果

eval 用前 12 個 test 場景（來源：`distill2/reports/slicegpt_slice.md`）：

| k | 共享基底能量 | rot_err | RRA@15 | storage | Latency |
|---|---|---|---|---|---|
| 原模型 | 100.00% | 2.87° | 1.000 | 842M | 223 ms |
| 512 | 99.52% | 5.62° | 0.958 | 692M | 208 ms |
| 384 | 99.12% | 34.60° | 0.382 | 654M | 203 ms |
| 256 | 98.30% | 49.74° | 0.264 | 616M | 199 ms |

k=512 不重訓還能用；k 小於 512 就急劇崩壞。原因是摺疊後的權重已經和網路訓練時看到的不一致（來源：論文 §4.5）。

### 恢復蒸餾

`distill2/train_sliced.py`：把切片後的模型放回原本的蒸餾迴圈，6 epochs × 400 steps、lr 2e-5，**凍結 DINOv2 本體**；旋轉 adapter（`patch_embed.lin`）、blocks、tokens、heads 都可訓練（來源：`distill2/train_sliced.py` 檔頭與預設值）。單張 RTX 5080 約 25 分鐘（來源：論文 §3.5）。

| k | 不重訓 | 恢復後 rot_err | RRA@15 | abs_rel | 權重 |
|---|---|---|---|---|---|
| 384 | 34.60° | **4.18°** | 1.000 | 0.181 | `runs/sliced/recover_k384/best.pt` |
| 256 | 49.74° | **4.95°** | 0.990 | 0.188 | `runs/sliced/recover_k256/best.pt` |

（來源：`distill2/reports/slicegpt_slice.md`。這兩個 checkpoint 不在 repo；`weights/` 只有 842M、464M 與 INT8 三個模型。）

**評估子集的問題**：恢復後的數字是在**前 8 個** test 場景上評估的（報告原文「eval 8 test scenes」；`train_sliced.py` 預設 `--eval_scenes 8`），而探針與不重訓的數字是在前 12 個場景上。因此「恢復後 4.95° 幾乎等於探針預測的 4.91°」（論文 §4.5）其實跨了兩個子集；同一個 842M checkpoint，在 12 個場景上是 2.87°（`slicegpt_probe.md`），在 8 個場景上是 3.42°（`quantize_report.md`）。結論「事後切片加恢復蒸餾能打通寬度壓縮」不受影響，但兩個數字不能逐位比較。

另外，論文沒有做「identity（直接截斷通道）」或「隨機正交基底」在相同恢復預算下的對照，所以只主張共享基底**足以**在恢復後達到 k=256，不主張它比簡單基底好（來源：論文 §4.5、Limitations）。

---

## 階段 8：三條流的擴展、前端負面結果、trunk 切到 1/4（2026-07-13～07-14）

切片推廣到三條互相獨立的殘差流（來源：`distill2/slicegpt_slice.py` 檔頭）：

1. aggregator（1024）
2. DINOv2 patch embed，即前端（ViT-L，1024）
3. camera head trunk（2048）。adaLN 調變是依資料而定的對角運算，不能旋轉，保留在原本的 2048 基底。

切片順序固定為 PE → camera trunk → aggregator（`apply_slices_` 依這個順序呼叫）；PE 必須先於 aggregator 切，`slice_patch_embed_` 裡的 assert 會檢查，順序錯了就直接報錯。三條流一起切的等價性測試：pose max|Δ| = 3.0e-7、depth 2.6e-6（來源：`distill2/reports/slicegpt_slice_combo.md`；論文 Appendix C）。

### 不重訓（前 12 個場景）

| 組合 | rot_err | RRA@15 | storage |
|---|---|---|---|
| agg0 + pe512 + ch1024 | 72.82° | 0.236 | 591M |
| agg256 + pe512 + ch1024 | 74.75° | 0.167 | 365M |

### 恢復蒸餾（前 8 個場景）

| 組合 | 恢復條件 | rot_err | RRA@15 | Params | 結論 |
|---|---|---|---|---|---|
| agg256 + pe512 + ch1024 | 凍結 DINOv2，6 ep | 69.4° | 0.177 | 365M | 塌 |
| agg256 + pe512 + ch1024 | `--train_pe`，8 ep | 77.2° | 0.188 | 365M | 塌 |
| agg256 + ch1024 | 凍結 DINOv2，6 ep | 5.36° | 0.948 | 516M | 可用 |
| **agg256 + ch512** | 凍結 DINOv2，6 ep | **5.20°** | **0.969** | **464.4M** | 可用，最終模型 |
| agg256 + ch512 + pe768 | 凍結 DINOv2，6 ep | 48.1° | 0.208 | 約 390M | 塌 |

（來源：`distill2/reports/slicegpt_slice_combo.md`。報告把最終模型寫成 466M，這是舊的錯誤數字，正確值見[階段 13](#階段-13camera-ready-前的核對與更正2026-10-17)。其他幾列從存檔重算的 state dict 元素數是 516.35M、364.72M、389.10M，和報告一致（checkpoint metadata，`推算`）。）

`--train_pe` 那列的 best.pt 存檔內 eval 是 75.26°、RRA@15 0.198（epoch 索引 3，checkpoint metadata）；報告與論文的 77.2° 推測是較後面 epoch 的 eval。兩者都屬塌縮，結論不變。

- **camera trunk 最好切**：1/4 寬（5.20°）和 1/2 寬（5.36°）的差距落在 run-to-run 的 1–2° 變動內，應視為相當，所以採用較小的版本（來源：論文 §4.6）。
- **前端（PE）不能切**：切到一半，不論 DINOv2 凍結或一起訓練都會塌縮；切到 3/4（k_pe=768）也只有 48.1°。和同預算下 aggregator／trunk 不到 6° 對照，這與「Path B 必須精確複製前端」的經驗一致。這是重要的負面結果（來源：`distill2/reports/slicegpt_slice_combo.md`）。
- 論文的措辭：結論寫的是「在測過的切片設定下」都失敗（none of the slicing configurations we tested），§1 寫的是「在測過的每一種恢復設定下」都塌縮（under every recovery setting we tested）；摘要只說 the same procedure fails on the DINOv2 front end，§1 的貢獻列表與 §4.6 則寫得比較寬（non-compressible under the same procedure；incompressible under this family of methods），§4.6 同一段又說沒試過的 per-site 基底仍可能是出路；Limitations 也承認無法提出機制性解釋（來源：論文摘要、§1、§4.6、§5）。本 repo 採用「在測過的設定下」的說法。
- 最終模型：`runs/sliced/recover_agg256_ch512/best.pt`（2026-07-13 23:34），best 為 epoch 索引 5，即第 6 個 epoch。公開權重是 `weights/vggt_sliced_464M_agg256_ch512.pt`，與原檔逐位元相同：1,908,372,429 bytes，SHA256 `FE6E2CF735DCCB993560C98D21B2E3DC1357F2EE0186986CB6BFE7C7E8F1B352`（來源：`distill2/README.md` §五之三；`weights/SHA256SUMS.txt`）。在 `distill2/` 下用 `load_sliced("../weights/vggt_sliced_464M_agg256_ch512.pt", "cuda")` 載入。

---

## 階段 9：在切片模型上疊加 INT8（2026-07-14）

torchao int8 weight-only，8 個場景（來源：`distill2/reports/quantize_sliced_report.md`；commit `faf9828`）：

| | 大小 | rot_err | abs_rel | VRAM | Latency |
|---|---|---|---|---|---|
| fp 權重 | 1858 MB | 5.20° | 0.193 | 3.89 GB | 223.9 ms |
| INT8 weight-only | **537 MB** | 5.21° | 0.193 | 3.68 GB | 248.8 ms |

- 1858 MB 是 `torch.save(model.state_dict())` 寫出的位元組數（`distill2/quantize.py` 的 `size_mb`，除以 1e6）。公開權重檔 1,908,372,429 bytes 另外還含旋轉基底 `Q_agg`、`Q_pe`、`Q_ch` 與 meta，所以兩個數字不同（三個 Q 以 float64 存，共 6,291,456 個元素、約 50.3 MB，`推算`）。
- 這份報告的 VRAM 與延遲不能直接和論文 Table 5 比：`quantize.py` 量 INT8 模型時，未量化的基準模型還留在 GPU 上（程式沒有釋放 `base`），所以 3.68 GB 裡含有基準模型的 fp32 權重（約 1.86 GB；3.68 − 1.81 = 1.87，`推算`）；延遲是 15 次平均（`latency_vram` 的 `runs=15`）。Table 5 的 1.81 GB／229.9 ms 來自 `measure_unified_bench.py` 的單獨量測（見[階段 11](#階段-11效率與記憶體量測2026-08-1708-28)）。
- 這個 INT8 模型就是 `weights/vggt_sliced_464M_int8.pt`（`runs/sliced/quantized_sliced_int8.pt` 原檔，537,215,008 bytes），以 `{"model": state_dict, "quant": ...}` 存檔。載入方式：先用 `load_sliced("../weights/vggt_sliced_464M_agg256_ch512.pt", "cuda")` 建出切片結構，`quantize_(model, Int8WeightOnlyConfig())`，再 `load_state_dict(ck["model"])`（見 [weights/README.md](../weights/README.md)）。
- 相對 teacher fp32 5025 MB 是 9.4×（來源：論文摘要、§4.7）。5025 MB 這個數字見[階段 13](#階段-13camera-ready-前的核對與更正2026-10-17) 第 6 項。
- **沒有加速**：torch 2.10.0+cu128 沒有 fused int8 kernel，權重在執行時才反量化（來源：論文 §4.7）。
- **措辭的更正**：內部記錄稱 INT8「零損失」（`distill2/README.md` §七）。論文只量了 vs teacher 的 rot_err（5.20° → 5.21°），沒有重量 GT 的 RRA、RTA、AUC 與 depth，所以**不宣稱 lossless**，INT8 模型的 GT 精度「未知」（來源：論文 §4.7）。

---

## 階段 10：CO3D GT 評估與穩健性檢查（2026-07-21～08-09）

### GT 座標慣例

CO3D 的 PyTorch3D 相機要換成 OpenCV 慣例，採用 `flip_xy`。正確性由「teacher 對 GT 的 RRA@15 = 1.000」驗證；`distill2/eval_gt.py` 在 teacher 偏低時會提示改試 `flip_yz` 或 `none`（來源：`distill2/make_gt_labels.py` 檔頭；`distill2/eval_gt.py`）。標註的實際位置是 `<CO3D_ROOT>/<category>/frame_annotations.jgz`。作者本機的 CO3D 資料夾名稱和腳本預設路徑不同，曾經因此找錯地方（作者開發紀錄）。

### 初步版：20 場景 × 8 幀（2026-07-21）

| 模型 | RRA@15 | RTA@15 | AUC@30 |
|---|---|---|---|
| Teacher | 1.000 | 0.975 | 0.981 |
| Path B | 0.979 | 0.954 | 0.877 |
| k384 恢復版 | 0.911 | 0.887 | 0.780 |
| agg256 + ch512 | 0.907 | 0.864 | 0.765 |

（來源：`distill2/reports/eval_gt_co3d_{pathb,k384,ch512}.md`；commit `efba476`）

### 正式版：65 場景 × 等距 10 幀，即論文 Table 2（2026-08-09）

| 模型 | Params | RRA@15 | RRA@30 | RTA@15 | RTA@30 | AUC@30 |
|---|---|---|---|---|---|---|
| Teacher | 1256M | 1.000 | 1.000 | 0.930 | 0.957 | 0.976 |
| Path B | 842M | 0.964 | 0.981 | 0.893 | 0.941 | 0.876 |
| k384 恢復版 | 654M | 0.905 | 0.939 | 0.826 | 0.910 | 0.792 |
| agg256 + ch512 | 464M | 0.883 | 0.940 | 0.814 | 0.908 | 0.767 |

（來源：`distill2/reports/eval_gt_co3d10_{pathb,k384,ch512}.md`；論文 Table 2）

- teacher 在這個子集上已經飽和，所以結果以「相對 teacher 的保留率」解讀：RRA@15 保留 96.4% 與 88.3%，AUC@30 保留 89.8% 與 78.6%（來源：論文 §4.3）。
- 本專案的 split 比文獻協定容易（teacher AUC@30 在這裡是 0.976，文獻值 0.882），絕對值不能和文獻直接比較。這裡的 AUC@30 只算旋轉（來源：論文 §4.1）。
- **隨機取幀**（seed 0，作者開發紀錄）：Path B 0.965／0.882、k384 0.923／0.813、agg256+ch512 0.875／0.762（RRA@15／AUC@30）。和等距取幀相比，RRA@15 最多差 0.018，AUC@30 最多差 0.021（來源：`distill2/reports/eval_gt_co3d10rand_*.md`；論文 §4.1）。作者當時的紀錄寫「差異 <±0.02」（作者開發紀錄），對 AUC@30 而言不精確，以論文的數字為準。

### 直接用 GT 微調（2026-08-09，結果中性）

假設：把監督目標從 teacher soft label 換成 CO3D GT 外參，也許能提升 GT 指標。`make_gt_labels.py` 為全部 492 個場景產生 `gt_extrinsics.npy`，以 `--gt_pose 1` 從各自最佳權重續訓 4 epochs，lr 1e-5（來源：`distill2/reports/gtft_report.md`）。

| 模型 | 微調前 RRA@15／RTA@15／AUC@30 | 微調後 |
|---|---|---|
| Path B | 0.964／0.893／0.876 | 0.960／0.898／0.856 |
| agg256 + ch512 | 0.883／0.814／0.767 | 0.883／0.821／0.770 |

沒有實質增益。內部報告把差距歸因於「容量／壓縮」（`gtft_report.md`）；論文只說差距「不太可能來自 teacher label 這個訓練訊號」，沒有歸因（來源：論文 §4.3）。

### 剪枝 × 切片疊加（2026-08-09，論文未收錄）

在 8 層剪枝模型上重新校準，切 agg256 + ch512，再恢復 6 epochs（來源：`distill2/reports/slicegpt_d8_combo.md`）：

| 模型 | rot_err | RRA@15 | storage | Latency |
|---|---|---|---|---|
| 8 層原模型 | 9.06° | 0.819 | 741M | 190 ms |
| 加切片，不重訓 | 52.79° | 0.306 | 439M | 178 ms |
| 加切片並恢復 | 9.08° | 0.740 | 439M | 178 ms |

- GT（65 × 10）：RRA@15 0.645、RRA@30 0.872、AUC@30 0.564，精度代價很大（來源：`distill2/reports/eval_gt_co3d10_d8combo.md`）。
- 報告的結論「切片幾乎零代價」是拿 9.08° 和 9.06° 比。但依腳本預設值推斷，9.06° 是 `slicegpt_slice.py` 在前 12 個場景上的結果，9.08° 是 `train_sliced.py` 在前 8 個場景上的結果；同一個 8 層模型在前 8 個場景上是 9.45°（`layer_pruning_report.md`）。這個子集判讀是`推測`。
- 439M 已從存檔重算：`runs/sliced_d8/recover/best.pt` 的 state dict 共 439.20M 個元素，和報告一致（checkpoint metadata，`推算`）。報告結論裡拿來對照的全深度版「466M」則是舊的錯誤數字（正確是 464.4M）。
- 論文把「剪枝與切片疊加」列為未來工作，沒有收錄這個實驗（來源：論文 Limitations）。

---

## 階段 11：效率與記憶體量測（2026-08-17～08-28）

### VRAM 對幀數曲線（2026-08-17）

條件：518×518、bf16、`no_grad`；記錄的是峰值 allocated 記憶體，不含 CUDA context（來源：`distill2/measure_vram_curve.py`；`distill2/reports/vram_curve.json`；論文 Fig. 2）。

| S | Teacher | Path B（842M） | 切片版（464M） | 切片版 + INT8 |
|---|---|---|---|---|
| 2 | 8.18 GB | 5.26 GB | 3.32 GB | 1.23 GB |
| 4 | 9.21 GB | 5.85 GB | 3.89 GB | 1.81 GB |
| 24 | 16.00 GB | 9.38 GB | 7.42 GB | 5.35 GB |
| 28 | 17.08 GB | 9.93 GB | 7.97 GB | 5.90 GB |
| 64 | 27.19 GB | 16.27 GB | 13.92 GB | 11.85 GB |
| 80 | — | 19.18 GB | 16.72 GB | 14.65 GB |

（INT8 欄來源：`distill2/reports/vram_curve_int8.json`，2026-08-28；這條曲線不在論文 Fig. 2 中。）

- 16 GB 內可處理的最大幀數：teacher 24、Path B 48、切片版 64、INT8 80，對應論文 Table 5 的「Max frames」。
- 超過 16 GB 的數值是靠 Windows 共享記憶體溢出才跑得出來，速度會大幅下降，實務上不可用（作者開發紀錄）。

### 統一批次量測（2026-08-28）

早期的延遲來自不同 session：teacher 384.6 ms、Path B 233.9 ms、切片版 201.1 ms，算出 1.64× 與 1.91×（來源：`distill2/reports/efficiency_report.md`；`distill2/README.md` §七）。校對論文時，把所有可直接比較的模型放在同一個 session 重量（S=4、warmup 5、20 次平均，來源：`distill2/measure_unified_bench.py`；commit `691c186`）：

| | Teacher | Path B | 8 層 | 6 層 | 4 層 | 切片版 | 切片版 + INT8 |
|---|---|---|---|---|---|---|---|
| Latency | 388.1 ms | 219.1 ms | 180.1 ms | 161.4 ms | 144.0 ms | 201.3 ms | 229.9 ms |
| VRAM | 9.21 GB | 5.85 GB | 5.07 GB | 4.67 GB | 4.28 GB | 3.89 GB | 1.81 GB |

（來源：`distill2/reports/unified_bench.json`）

論文 Table 5 與 §4.7 用的是這一組：加速 1.77×（388.1／219.1）與 1.93×（388.1／201.3）。conv-512 scratch student 沒有納入這批量測，在 Table 5 以 † 標成較早批次的參考值（來源：論文 Table 5 註腳）。

---

## 階段 12：論文修訂時補做的量測（2026-10-02～2026-10-03）

論文修訂時補了三項量測（來源：`distill2/measure_budget_incl_context.py`、`measure_flops_extra.py`；commit `a408314`）。這些內容在論文中對應：摘要的記憶體預算句、Table 5 的 GFLOPs 欄、Table 7 與 Appendix D、§4.7 的「Edge deployment」段。

### 1. 含 CUDA context 的記憶體預算模擬（論文 Table 7）

做法：先量出 CUDA context（約 303 MiB），再以 `torch.cuda.set_per_process_memory_fraction` 把 caching allocator 限制在「預算 B 減 context」，然後逐步增加 S，直到第一次 OOM（來源：`distill2/measure_budget_incl_context.py`；論文 Appendix D）。

| 模型 | Default 4／6／8／12 GiB | Frozen 4／6／8／12 GiB |
|---|---|---|
| Teacher | 0／0／1／8 | 0／1／4／12 |
| Path B（842M） | 0／4／12／32 | 1／4／20／48 |
| 切片版（464M） | 2／8／24／48 | 4／16／32／48 |
| 切片版 + INT8 | 8／20／32／56 | 8／20／32／56 |

（來源：`distill2/reports/budget_incl_context_paper.json`（Default）、`budget_incl_context_frozen.json`（Frozen）；論文 Table 7）

- **Default** 指參數維持預設的 `requires_grad=True`，與 Table 5、Fig. 2 的量測狀態相同；**Frozen** 指 `requires_grad=False`。背景說明（論文沒寫）：Default 狀態下，autocast 會為可訓練的 fp32 權重保留一份 bf16 副本，所以峰值記憶體較高（來源：`distill2/measure_budget_incl_context.py` 的 `MODE` 註解）。
- 因此 Table 5 的 VRAM 與 Fig. 2，對三個非 INT8 模型而言是偏保守的數字。
- JSON 裡還有 16 GiB 一欄，論文刻意省略，因為超過這張卡實際可用的記憶體（來源：論文 Appendix D）。
- 「8 GB 內可處理幾幀」因此有三種定義、三組答案：teacher 在 Fig. 2 讀值不到 2 幀（S=2 已是 8.18 GB）、Table 7 Default 是 1、Frozen 是 4；切片版分別是 28、24、32（來源：論文 §4.7、Table 7；`distill2/reports/vram_curve.json`）。注意 Fig. 2 用 GB（10⁹ bytes），Table 7 的預算用 GiB。

### 2. GFLOPs（論文 Table 5 的 GFLOPs 欄）

改用 `torch.utils.flop_counter.FlopCounterMode`，S=4：teacher 16814.3、Path B 9919.2、切片版 7496.8、8 層 8194.1、conv-512 2399.8（來源：`distill2/reports/budget_incl_context_frozen.json`、`flops_extra.json`）。INT8 是 weight-only 量化，GFLOPs 與切片版相同（來源：論文 Table 5 caption）。和[階段 3.1](#31-效率量測第一版2026-07-01) 的 fvcore GMACs 是不同的計數器，不能放在一起比。

### 3. CPU-only 計時（論文 Appendix D）

fp32、8 threads、S=2：teacher 16233.7 ms、切片版 7403.1 ms（來源：`distill2/reports/budget_incl_context_frozen.json`）。論文寫成 16.2 s 對 7.4 s，2.19×，並註明兩者都不算互動速度（來源：論文 Appendix D）。

### 沒有做的事

沒有在實體邊緣裝置上測試，記憶體預算只是在 RTX 5080 上模擬；也沒有量邊緣硬體上的延遲、功耗、溫度，沒有匯出到具備 fused INT8 kernel 的推理 runtime（來源：論文 §4.7、Appendix D、Limitations）。

---

## 階段 13：camera-ready 前的核對與更正（2026-10-17）

論文已被 ICS 2026 接受，將收錄於 Springer CCIS（見根目錄 [README.md](../README.md)）。2026-10-17 把論文的文字逐項對照程式、報告與 checkpoint metadata，下表列出記錄與論文不一致之處。「論文」欄是論文的寫法；「記錄」欄是本 repo 認定的事實。

| # | 項目 | 論文 | 記錄 | 依據 |
|---|---|---|---|---|
| 1 | 最終模型參數量 | 464M（Table 6 為 464.4M） | **464.4M**。早期報告寫的 466M 是同一個模型的錯誤數字；壓縮比 1256.5／464.4 = 2.71×（舊的 2.70× 是用 466M 算的，`推算`） | `distill2/README.md` §五之三（464.16M 參數；state_dict 464.43M，多出的約 0.28M 是非參數 buffer：`deslice_Q` 0.262M 與 SlicedNorm 的 `v` 0.017M）、§八；`weights/README.md`；論文 Table 6 |
| 2 | Table 4 的評估子集 | 全部用前 12 個 test 場景（7 個類別） | 只有探針、不重訓的值與 2.87° 用 12 個場景。恢復後的值（4.18、4.95、5.36、5.20、69.4、77.2、48.1）與 INT8 檢查（5.20 → 5.21）用**前 8 個場景（4 個類別）**；Table 1、3 也是 8 個 | 論文 §4.1、Table 4 caption；`distill2/reports/slicegpt_slice.md`、`slicegpt_slice_combo.md`（「恢復蒸餾 eval 8 test scenes」）、`quantize_sliced_report.md`；`train_sliced.py`、`run_ablation.py`、`run_layer_pruning.py` 預設 `--eval_scenes 8` |
| 3 | 主 run 的 epoch 數 | 20 epochs × 500 steps | `runs/distill_b1/best.pt` 來自 **40 epoch** 的 run（best 在 epoch 索引 37） | 論文 §4.1；checkpoint metadata（`weights/vggt_teacher_init_842M.pt` 合併後也讀得到同樣的 `epoch`）；`distill2/train.py` 預設 `--epochs 40`；`distill2/README.md` §四 |
| 4 | 跨 session 延遲漂移 | 約 7% | 最多約 15%。scratch student 123.8 ms（07-01）對 105.2 ms（07-12），以較慢的一次為基準差約 15%；切片版 223.9 ms（07-14）對 201.3 ms（08-28），約 10%（以較快的一次為基準則是約 18% 與 11%）（`推算`）（後者兩次量測的次數不同：15 次 vs 20 次平均，所以不完全是 session 差異；最大值 15% 來自同一支 measure_efficiency.py 的兩次量測） | 論文 §4.7；`efficiency_report.md`、`efficiency_prune.md`、`quantize_sliced_report.md`、`unified_bench.json` |
| 5 | scratch student 是否重測過 | Table 5 註腳寫「unrepeated batch」 | 有重測（105.2 ms），只是沒有在 Table 5 的 session 中重測 | 論文 Table 5 註腳；`efficiency_prune.md` |
| 6 | teacher fp32 大小 | 5025 MB | 用 537 MB、1858 MB 的同一算法（位元組數 ÷ 1e6）估計約 **5026 MB**：teacher 共 1,256,537,516 個 fp32 元素，即 5,026,150,064 bytes，HF 上的 `model.safetensors` 是 5,026,367,224 bytes；記錄裡找不到 5025 MB 的算法。9.4× 不受影響（5026／537 ≈ 9.36）（`推算`） | 論文 §1、Fig. 1、§4.7；`facebook/VGGT-1B` 的 `model.safetensors` 檔頭 |
| 7 | INT8 的精度 | 不宣稱 lossless | 內部早期記錄寫「零損失」，以論文的保守寫法為準 | 論文 §4.7；`distill2/README.md` §七 |
| 8 | 三元件的必要性 | 不宣稱各自必要 | 內部記錄寫「三者缺一不可」，以論文為準 | 論文 §4.2；`distill2/README.md` §一、§七 |
| 9 | SliceGPT 的描述 | 「applying one globally shared orthogonal rotation」 | SliceGPT 是逐 block 一個 Q，並在 residual 連接上加 Q_{l−1}ᵀQ_l；「共用一個 Q」是本專案的做法 | 論文 §2；SliceGPT 原論文（論文參考文獻 [17]）；`distill2/slicegpt_probe.py` 檔頭 |
| 10 | 剪枝 × 切片 | 列為未來工作 | 已有一次實驗：439M、9.08°、GT RRA@15 0.645，論文未收錄 | `distill2/reports/slicegpt_d8_combo.md`、`eval_gt_co3d10_d8combo.md` |

上表的「論文」欄以本 repo 對照的論文文字為準；正式出版的版本若有修改，以出版版本為準。本 repo 的文件以「記錄」欄為準；第 6 項例外：根目錄 README 與其他文件沿用論文的 5025 MB，另註推算值約 5026 MB。

---

## 教訓

給想重做這個研究的人：

1. **pose 的訊號要另外檢查，不能看總 loss**。loss 在下降、depth 很好，不代表 pose 有學到。要直接看跨幀四元數的標準差、camera token 的跨幀 cosine、`camera_token` 參數有沒有離開初始值（`distill2/CHANGES.md` §2）。
2. **四元數慣例先確認**。VGGT 是 scalar-last `(x,y,z,w)`（`vggt/utils/rotation.py`）。相對旋轉項只要用錯一處，梯度就變成噪音，而且不會報錯。
3. **rotation 要和 translation、FoV 分開正規化**；early stopping 也不要監看會出現假性 plateau 的子項（`distill2/loss.py`；`distill2/CHANGES.md` §3(A)(B)）。
4. **確認實際 import 到的是哪一份 `vggt`**。不要用 `sys.path.insert` 把舊版放到前面；camera head 的 `trunk_depth`（2 或 4）一旦弄錯，評估會失真（`scripts/eval_student_vs_teacher.py` 檔頭；`distill2/CHANGES.md` §3(C)）。
5. **修 bug 是必要條件，不是解法**。loss 的四項修正做完，rotation 仍然約 82°（`distill2/CHANGES.md` §3(A)）；early stopping 與 eval 修正之後，記錄也沒有舊管線脫離塌縮的結果，最後是整套改寫成 `distill2/` 才解決（§3(D)）。
6. **pose 路徑要繼承，不要從頭學**。前端、aggregator（每隔一層）、camera head 同時複製並凍結前端才成功；單一元件都不夠；前端一解凍就塌回 75.68°（`distill2/reports/ablation_report.md`）。
7. **要縮寬度，先訓練再切**。五種「先變窄再訓練」的策略都塌縮了；「全寬訓練、事後切片、短程恢復」才成功（作者開發紀錄；`distill2/reports/slicegpt_slice.md`）。
8. **先用便宜的探針判斷可不可行，再做權重手術**。hook 投影不動任何權重，幾分鐘就能回答「值不值得做」（`distill2/slicegpt_probe.py` 檔頭）。
9. **權重手術一定要有等價性測試**：k 取滿維、Q 用隨機正交矩陣時，必須重現原模型輸出，誤差應在浮點雜訊等級（本專案 pose 3.0e-7、depth 2.6e-6；`python slicegpt_slice.py --check`）。每改一條摺疊規則就重跑一次。
10. **DINOv2 前端不要動**：不微調，也不切片，在 512 與 768 都會塌縮（`distill2/reports/slicegpt_slice_combo.md`）。
11. **feature KD 的 projector 可能把塌縮藏起來**。可學的 projector 能讓已塌縮的 token「看起來像 teacher」；約束要作用在 head 實際讀取的原始 token 上（`distill2/loss2.py` 檔頭）。
12. **固定並寫明評估子集，不要跨子集比較**。同一個 `runs/distill_b1` checkpoint 在前 12、前 8 個場景上分別是 2.87°、3.42°；20 epoch 的 run 在前 8 個與全部 65 個場景上是 4.44°、4.89°。`quick_eval` 取的是依名稱排序的前 N 個場景，各腳本預設的 N 不同（8 或 12）。
13. **最好另切 validation split**。`train.py` 與 `train_sliced.py` 是用 test split 前 8 個場景的 rot_err 挑 `best.pt`，而同一批場景又被拿來報告 Table 1、3、4 的恢復值（`distill2/train.py` 定期 eval 的區段）。
14. **vs teacher 與 vs GT 的 RRA@15 是不同指標**：前者看每幀相對第 0 幀的旋轉，後者看兩兩配對的相對位姿。最終模型同時有 0.969（vs teacher；存檔內 quick_eval 在前 8 個場景上的逐幀值）與 0.883（vs GT 配對，AUC@30 0.767）兩個值，引用時一定要寫明是哪一個。teacher 在本子集上已經飽和，請用保留率解讀。
15. **只比同一個 session 量到的延遲**。同一個模型跨 session 的延遲差距可達約 15%（`推算`，見階段 13 第 4 項）。
16. **報 VRAM 時要寫明量測狀態**：參數有沒有 `requires_grad`（autocast 會保留 bf16 權重副本），以及有沒有把 CUDA context 算進預算。兩者都會改變「可處理幾幀」的答案（論文 Table 7；`distill2/measure_budget_incl_context.py`）。同一支腳本裡有沒有其他模型還留在 GPU 上，也會改變峰值（見階段 9 的 3.68 GB）。
17. **參數量與檔案大小要從存檔重算，並交叉核對**。466M 對 464.4M 是早期報告算錯；1858 MB（state_dict）對 1,908,372,429 bytes（另含 float64 的 Q）是定義不同；teacher 的 5025 MB 對約 5026 MB 則找不到算法。
18. **訓練實際跑了幾個 epoch，看存檔的 `epoch` 欄**，不要從筆記裡的指令推斷（作者早期筆記裡主 run 的指令寫 `--epochs 20`，`runs/distill_b1` 實際跑了 40 個 epoch）。
19. **措辭不要超出做過的對照**。沒跑兩兩組合，就不說「缺一不可」；沒做 basis 對照，就不說「共享基底比較好」；沒重量 GT，就不說 INT8「零損失」；沒有機制性解釋，就不說「前端不可壓縮」是定律（論文 Limitations）。
20. **沒有固定全域 seed，也沒有做過同設定的重複實驗**（`distill2/README.md` §八）。論文用「1–2° 的 run-to-run 變動」來解讀小差距（論文 §4.6、Limitations）；幾個設定不同的「複製全部」run 在前 8 個場景上落在 3.42°–4.44°，1–2° 的差距不要過度解讀。
21. **在這套 torch 版本上，INT8 weight-only 不會加速**（torch 2.10.0+cu128 沒有 fused kernel）。它的好處是儲存空間與記憶體，不是延遲。
22. **已經證明無效的方向不必重跑**：窄模型從零訓練、DINOv2-S 從零、feature KD 與結構 KD 的二次蒸餾、SVD 投影初始化、直接用 GT 微調（作者開發紀錄；`distill2/reports/gtft_report.md`）。
23. **Windows 環境的實務問題**：子行程輸出被重導向時要設 `PYTHONUTF8=1`，否則 cp950 編碼錯誤會讓訓練中途中止（`distill2/run_layer_pruning.py`）；GPU 滿載導致螢幕黑屏時，可加 `--throttle 0.05`（`distill2/README.md` §四）。
