> **說明（2026-10 公開時加註）**：本檔是開發期間的改動紀錄，保留原文。文中的數字是當時的紀錄（例如壓縮倍率、早期評估值），不一定等於論文的最終數字；論文數字與出處請以 [`docs/RESULTS.md`](../docs/RESULTS.md) 為準。文中提到的 `training/` 是更早的舊管線，未收錄在本 repo。

# VGGT 蒸餾改動說明

本 repo 是基於官方 [VGGT](https://github.com/facebookresearch/vggt) 的蒸餾實驗：
把 VGGT-1B（teacher）壓縮成較小的 student。以下說明相對原始 VGGT 我做了哪些改動、
以及解決 rotation 學不起來的完整過程。

---

## 1. 目標

用 teacher（VGGT-1B）對 CO3D 場景產生 soft labels（depth / camera pose / point map），
再訓練一個較小的 student 去蒸餾這些輸出。資料集在另一個資料夾
`<DATA_ROOT>`（train 427 / test 65 場景）。

## 2. 遇到的核心問題：rotation 完全學不起來

最初的 student（conv patch embed、embed_dim 512、8x 壓縮、從零訓練）：

- **depth 學得起來**（abs_rel ~0.24）
- **但 camera rotation 完全塌縮**：`mean_rot_err ≈ 82°`（等於隨機）、`RRA@15 ≈ 0.13`

### 診斷結論
直接載入訓練好的 student 檢查，發現：
- student 對**每一幀都輸出幾乎相同的 ≈identity 四元數**（跨幀標準差 ~0.01 vs teacher ~0.5），連平移也一起塌縮。
- aggregator 餵給 camera head 的 **camera token 跨幀 cosine 相似度 = 0.998**（幾乎不分視角）。
- checkpoint 裡 `aggregator.camera_token` 參數仍接近初始化值。

→ **camera 這條路徑塌縮成「不管輸入、一律輸出平均 pose」的偷懶解**。
原因是「小模型 + conv 隨機初始化 + 小資料」學不出判斷視角所需的幾何特徵；
depth 因為有大 DPT decoder + 讀空間 patch tokens 而不受影響。

## 3. 改動一覽（依時間順序）

### (A) Loss 修正 — `training/distill_loss.py`
1. **四元數慣例修正**：VGGT 的 pose_enc 四元數是 **scalar-last `(x,y,z,w)`**（實部在 index 3），
   但原相對旋轉項誤用 scalar-first，導致相對旋轉梯度是噪音。已修正 `quat_mul`/`quat_conj`。
2. **旋轉 loss 改 `1 - |dot|`**：取代 `arccos`，避免接近正解時梯度爆衝。
3. **rotation 與 (translation+FoV) 拆開、各自獨立 EMA 正規化 + 獨立權重**（`rot_weight`），
   避免 rotation 被快收斂的 T/FoV 稀釋。
4. 修了 loss 模組的 EMA buffer device bug（trainer 沒對 loss `.to(device)`）。

> 這些都是真 bug 該修，但**單靠它們沒有解決塌縮**（rotation 仍 ~82°）。

### (B) Early stopping / 訓練設定 — `training/distill_trainer.py`, config
- early stopping 從監看 `loss_camera_kd`（會因 T/FoV 假性 plateau 而過早停）改成監看總 objective，patience 放寬。
- config 加 `rot_weight`、改用 DINOv2 patch embed 等實驗。

### (C) Eval 修正 — `scripts/eval_student_vs_teacher.py`
- 發現 eval 因 `sys.path` 順序載入了**錯誤版本的 vggt 套件**（trunk_depth=2），把訓練用的 4 層 camera head 截斷成 2 層 → 旋轉評估失真。
- 移除錯誤的 `sys.path.insert`，並讓 `load_student` 自動偵測架構（conv / DINOv2、embed_dim、camera head 層數）。

### (D) ✅ 真正的解法：`distill2/`（全新、自包含的重建版）
推翻舊的 Hydra/trainer 那套，新建乾淨管線，採 **Path B — teacher-init（structured distillation，精神同 DistilBERT）**：

**不從零學，而是直接從 teacher 複製權重再蒸餾微調：**
- embed_dim=1024（對齊 teacher）、agg_depth=12（teacher 24 層取每隔一層）、patch_embed = DINOv2 ViT-L/14（複製 teacher）。
- **複製內容**：patch embed、camera/register token、aggregator 每隔一層的 block、
  **整個 camera head（維度與 teacher 相同 → pose 能力直接繼承）**。
- depth/point head 較小、不複製，由蒸餾學。
- 壓縮率約 **2.5x**（換取 pose 學得起來）。

## 4. 最終結果

| 指標 | 起點（塌縮） | 最終（distill2, 全 65 test 場景） |
|---|---|---|
| mean_rot_err | 82.7° | **4.89°** |
| RRA@15 | 0.135 | **0.968** |
| abs_rel (depth) | 1.04 | **0.176** |
| delta_1 (depth) | 0.40 | **0.698** |
| 壓縮率 | — | ~2.5x |

**結論**：rotation 從隨機（82°）降到 teacher 等級（~5°）。
**核心教訓**：pose 這種高度依賴 teacher 內部表徵的任務，小模型從零蒸餾極難，
**直接繼承 teacher 權重（teacher-init）** 才是對的路；depth 反而適合從零蒸餾。

## 5. 目錄導覽

| 路徑 | 說明 |
|---|---|
| `distill2/` | **最終採用的乾淨重建版**（見 `distill2/README.md`）|
| `distill2/build_student.py` | 從 teacher 切出 student 並複製權重 |
| `distill2/{data,loss,train,eval}.py` | 資料 / loss / 訓練 / 評估 |
| `scripts/export_teacher_labels.py` | 用 teacher 產生 soft labels |
| `scripts/eval_student_vs_teacher.py` | 舊版 eval（已修架構偵測）|
| `training/` | 舊版 Hydra 訓練管線（已被 distill2 取代，保留供對照）|

> 注意：checkpoint（`*.pt`）、訓練 log（`training/logs`、`distill2/runs`）、
> teacher 資料集等大檔都在 `.gitignore`，不會上傳。
