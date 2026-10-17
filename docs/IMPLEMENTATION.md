# 實作說明（IMPLEMENTATION）：程式碼如何對應論文的方法

這份文件逐項說明論文 *Compressing a 3D Foundation Model: Teacher-Initialized Distillation and Global Projection Slicing for Deployable VGGT*（Tzu-Chi Huang、Kuan-You Chen、Chung-Wei Lin、Chow-Sing Lin，國立臺南大學；ICS 2026 已接受，Springer CCIS，尚未出版）的每個方法寫在哪個檔案、哪個函式，對應論文的哪條式子或規則，以及程式和論文描述不一致的地方。

## 閱讀約定

- 「論文」指 camera-ready 版本（16 頁）。論文 PDF 不在本 repo；引用論文時寫成 `（來源：論文 §3.5 / Table 4 / Appendix C）`。
- 每個數字後面都附來源，程式與報告寫成 `（來源：檔案 L行號 / 章節）`，檔案都是 repo 裡的路徑。
- **推算**：用引用的數字做算術得到，沒有任何檔案直接寫出這個值。
- **由程式碼推論**：從程式邏輯推得，沒有量測紀錄支持。
- **（作者開發紀錄）**：只存在作者私人紀錄、或沒有放進 repo 的 checkpoint（例如 `runs/` 底下的其他檔案）裡的事實。本文件只保留逐項核對過的部分。
- 路徑都是 repo 相對路徑。`runs/...` 是作者訓練時的目錄名稱，用來說明某個檔案是怎麼來的。資料位置用兩個佔位符：`<DATA_ROOT>` 是 teacher soft labels 的根目錄（底下有 `train/`、`test/`），`<CO3D_ROOT>` 是 CO3D 標註根目錄（每個類別一個 `frame_annotations.jgz`）。
- **權重**：repo 根目錄的 `weights/` 放了論文的三個模型（Git LFS，說明見 [`weights/README.md`](../weights/README.md)）：842M teacher-initialized student `vggt_teacher_init_842M.pt`（＝`runs/distill_b1/best.pt`，因 GitHub LFS 單檔 2 GB 上限切成 `.part0`、`.part1`，用 `cat` 或 `copy /b` 合併）、464M 切片主模型 `vggt_sliced_464M_agg256_ch512.pt`（＝`runs/sliced/recover_agg256_ch512/best.pt`）、INT8 版 `vggt_sliced_464M_int8.pt`（＝`runs/sliced/quantized_sliced_int8.pt`）。其他 checkpoint（`runs/`、`abl/` 底下的其餘檔案，例如消融、剪枝、`student_init.pt`）不在 repo 內。在 `distill2/` 底下執行的指令，權重路徑寫成 `../weights/...`。
- 程式片段逐字引用，行號以本 repo 的版本為準。

---

## 0. 論文與程式的快速對照

| 論文位置 | 內容 | 程式 |
|---|---|---|
| §3.1、Appendix A（Table 6） | student 架構與參數預算 | `vggt/models/vggt_student.py`（`VGGTStudent`） |
| §3.3 | teacher-init 權重複製 | `distill2/build_student.py`；驗證：`distill2/verify_teacher_copy.py` |
| §3.3、§4.1 | 資料 canonicalization | `distill2/data.py`；soft label 產生：`scripts/export_co3d_labels.py` |
| §3.3、Appendix B | 蒸餾 loss | `distill2/loss.py` |
| §4.1 | 訓練配方 | `distill2/train.py` |
| §4.2（Table 1） | 初始化消融 | `distill2/run_ablation.py` |
| §4.4（Table 3） | 層剪枝 | `distill2/run_layer_pruning.py` |
| §3.4（Table 4 "Probe"） | 光譜探針 | `distill2/slicegpt_probe.py` |
| §3.5、Appendix C | 全域投影切片 | `distill2/slicegpt_slice.py` |
| §3.5（Table 4 recovered） | 恢復蒸餾 | `distill2/train_sliced.py` |
| §4.7 | INT8 weight-only | `distill2/quantize.py` |
| Table 5、Fig. 2 | latency / VRAM / max frames | `distill2/measure_unified_bench.py`、`measure_vram_curve.py`、`measure_vram_int8.py` |
| Table 5 GFLOPs、Table 7、Appendix D CPU | 記憶體預算模擬、FLOPs | `distill2/measure_budget_incl_context.py`、`measure_flops_extra.py` |
| Tables 1/3/4（對 teacher） | 評估 | `distill2/eval.py`（`quick_eval`） |
| Table 2（對 CO3D GT） | 評估 | `distill2/eval_gt.py` |

**執行前提**

- `distill2/` 的腳本假設工作目錄就是 `distill2/`。它們彼此用 `from eval import ...`、`from loss import ...` 匯入。另外需要能 `import vggt`，也就是本 repo 的 `vggt/` 套件，在 repo 根目錄 `pip install -e .` 安裝（來源：[`distill2/README.md`](../distill2/README.md) §二）。
- 很多參數的預設值是 Windows 的反斜線相對路徑，例如 `r"runs\distill_b1\best.pt"`（來源：`distill2/slicegpt_slice.py` L462、`distill2/train_sliced.py` L34），在 Linux/macOS 上請用參數覆寫。`measure_unified_bench.py`、`measure_vram_curve.py`、`measure_vram_int8.py`、`measure_flops_extra.py`、`measure_budget_incl_context.py` 則把 checkpoint 與輸出 JSON 路徑直接寫死在程式裡（例如 `measure_budget_incl_context.py` L38、L41、L66），沒有參數可覆寫：要用公開權重，得把路徑改成 `../weights/vggt_teacher_init_842M.pt`（先合併）與 `../weights/vggt_sliced_464M_agg256_ch512.pt`，或把檔案複製到程式預期的 `runs/...` 位置（見 [`CODE_MAP.md`](CODE_MAP.md)、[`REPRODUCE.md`](REPRODUCE.md)）；剪枝模型 `runs/prune/d{8,6,4}` 不在 repo 內。
- `--data_root` 的預設值是作者本機的絕對路徑，一律要改成 `<DATA_ROOT>`。
- 程式註解裡的論文表號（例如「Table 6」「Table 3/6」）沿用早期草稿的編號（來源：`distill2/measure_budget_incl_context.py` L48、`distill2/measure_unified_bench.py` L2、`distill2/measure_vram_int8.py` L2、`distill2/measure_flops_extra.py` L2）。對應到最終版：效率是 Table 5，記憶體預算是 Table 7；`measure_unified_bench.py` 的「Table 3」是層剪枝表，最終版也是 Table 3（它的 latency 欄 219.1 / 180.1 / 161.4 / 144.0 ms 就是 `distill2/reports/unified_bench.json` 的值）。

---

## 1. Student 架構（`vggt/models/vggt_student.py`）

### 1.1 `VGGTStudent` 可設定的參數

`VGGTStudent.__init__` 的參數有：`img_size`、`patch_size`、`embed_dim`、`agg_depth`、`num_heads`、`patch_embed`（`"conv"` 或 `"dinov2_vit{s,b,l}14_reg"`）、`pretrained_patch_embed`，以及 `enable_camera/depth/point/track`（來源：`vggt/models/vggt_student.py` L78–91）。選 DINOv2 前端時，`embed_dim` 必須等於該 ViT 的寬度，否則直接拋錯（來源：同檔 L94–107）。forward 的輸入輸出和 `VGGT.forward` 相同（來源：同檔 L224–304；`vggt/models/vggt.py` L29–96）。

### 1.2 842M student 的實際設定

842M student 由 `build_student.py` 建立：

`distill2/build_student.py` L52–57
```python
    student = VGGTStudent(
        img_size=518, patch_size=14, embed_dim=1024, agg_depth=student_depth,
        num_heads=16, patch_embed="dinov2_vitl14_reg",
        pretrained_patch_embed=(not copy_pe),   # 不複製時用 hub 預訓練；複製時待會蓋掉
        enable_camera=True, enable_depth=True, enable_point=True, enable_track=False,
    )
```

`student_depth` 預設是 12（來源：`distill2/build_student.py` L48、L98）。

| 元件 | Teacher（VGGT-1B） | 842M Student | 處置 |
|---|---|---|---|
| 前端 | DINOv2 ViT-L/14-reg（`Aggregator` 預設 `patch_embed="dinov2_vitl14_reg"`，來源：`vggt/models/aggregator.py` L75） | 同一個 ViT-L/14-reg | 從 teacher 複製並凍結 |
| Aggregator | 24 層、寬 1024、16 heads（來源：`vggt/models/aggregator.py` L66–68；`vggt/models/vggt.py` L22 使用預設值） | 12 層、寬 1024、16 heads；`mlp_ratio=4.0`、`aa_order=["frame","global"]`、`aa_block_size=1`、`qk_norm=True`、`rope_freq=100`、`init_values=0.01`（來源：`vggt/models/vggt_student.py` L115–129） | 每隔一層取一層 |
| Camera head | `CameraHead(dim_in=2048)`，預設 `trunk_depth=4`、`num_heads=16`（來源：`vggt/heads/camera_head.py` L28–33；`vggt/models/vggt.py` L24） | `CameraHead(dim_in=2048, trunk_depth=4, num_heads=16)`（來源：`vggt/models/vggt_student.py` L143–150） | 形狀完全相同，所以可以整個複製 |
| Depth head | `DPTHead`，`features=256`、`out_channels=[256,512,1024,1024]`、`intermediate_layer_idx=[4,11,17,23]`（來源：`vggt/heads/dpt_head.py` L50–52） | `features=128`、`out_channels=[128,256,512,512]`、`intermediate_layer_idx=[2,5,8,11]`（來源：`vggt/models/vggt_student.py` L154–166） | 縮窄，從頭訓練 |
| Point head | 同上，`output_dim=4`、`activation="inv_log"`（來源：`vggt/models/vggt.py` L25） | 同 depth head 的縮窄設定（來源：`vggt/models/vggt_student.py` L170–182） | 縮窄；有實例化但沒有任何 loss 讀它 |
| Track head | `TrackHead`（來源：`vggt/models/vggt.py` L27） | `enable_track=False`，所以是 `None`（來源：`vggt/models/vggt_student.py` L185–188） | 移除 |

**參數預算（論文 Table 6，從 state dict 計數）**

| 元件 | Teacher | Teacher-init（842M） |
|---|---|---|
| Front end | 304.4M | 304.4M |
| Aggregator | 604.7M | 302.4M |
| Camera head | 216.2M | 216.2M |
| Depth head | 32.7M | 9.6M |
| Point head | 32.7M | 9.6M |
| Track head | 65.9M | — |
| 合計 | 1256.5M | 842.2M |

（來源：論文 Appendix A, Table 6）

**DPT 中間層索引**是依 teacher 的比例縮放出來的：

`vggt/models/vggt_student.py` L208–212
```python
        ratios = [4/23, 11/23, 17/23, 1.0]
        idx = []
        for r in ratios:
            i = min(round(r * (depth - 1)), depth - 1)
            idx.append(i)
```

depth=12 時結果是 `[2, 5, 8, 11]`（來源：同檔 L25、L206）。student 第 i 層來自 teacher 第 2i 層（見 §2），所以 student 的 DPT 取點 2、5、8、11 對應 teacher 的第 4、10、16、22 層，而 teacher 自己取的是 4、11、17、23（推算）。

**Token 排列與 head 讀取方式**

- 每幀有 1 個 camera token 和 4 個 register token。參數形狀是 `(1, 2, X, C)`：index 0 給第一幀，index 1 給其餘幀（來源：`vggt/models/aggregator.py` L139–142、L353–376）。
- patch token 從 `patch_start_idx = 1 + 4 = 5` 開始（來源：同檔 L145）。每幀 patch 數是 (518/14)² = 1369（推算）。
- Aggregator 每一對 frame/global block 輸出一個 frame 與 global 串接後的張量 `[B,S,P,2C]`（來源：同檔 L295–298）。student 有 12 層，所以 list 長度是 12。
- Camera head 只讀最後一個張量的第 0 個 token（來源：`vggt/heads/camera_head.py` L86–89）。
- 所有 head 都在 `torch.cuda.amp.autocast(enabled=False)` 下執行（來源：`vggt/models/vggt_student.py` L266），和 teacher 一樣（來源：`vggt/models/vggt.py` L65）。

**Pose encoding 的格式**：9 維，依序是 translation T(3)、四元數(4)、fov_h、fov_w（來源：`vggt/utils/pose_enc.py` L46–55）。四元數是 scalar-last，也就是 (x, y, z, w)（來源：`vggt/utils/rotation.py` L16）。

**相對 teacher 移除或停用的部分**

- track head；
- 24 對 aggregator block 中的 12 對；
- feature projector（已註解掉，來源：`vggt/models/vggt_student.py` L190–191）。

**過時的註解**：`vggt_student.py` 的 docstring 有一張規格表（L27–36），列的是早期變體（Student-conv 約 121M、Student-dinov2s 約 96M）。實測參數量是 146.4M 和 110.7M（來源：`distill2/reports/efficiency_report.md`）。那張表沒有更新，論文也沒有用；檔頭 L1–4 的註記也說明這張表來自早期階段，論文的 student 由 `distill2/build_student.py` 建立。

---

## 2. Teacher-initialized 權重複製（`distill2/build_student.py`）

### 2.1 複製流程

`--copy` 的值可以是 `all`、`none`，或用逗號分隔的 `patch_embed,aggregator,camera_head`（來源：`distill2/build_student.py` L23、L33–45）。核心迴圈：

`distill2/build_student.py` L62–77
```python
        t_agg, s_agg = teacher.aggregator, student.aggregator
        t_depth = t_agg.depth
        stride = t_depth // student_depth

        if copy_pe:
            _copy(s_agg.patch_embed, t_agg.patch_embed, "patch_embed")
        if "aggregator" in copy:
            with torch.no_grad():
                s_agg.camera_token.copy_(t_agg.camera_token)
                s_agg.register_token.copy_(t_agg.register_token)
            for i in range(student_depth):
                j = i * stride
                _copy(s_agg.frame_blocks[i], t_agg.frame_blocks[j], f"frame[{i}]<-t[{j}]")
                _copy(s_agg.global_blocks[i], t_agg.global_blocks[j], f"global[{i}]<-t[{j}]")
        if "camera_head" in copy and student.camera_head is not None and getattr(teacher, "camera_head", None):
            _copy(student.camera_head, teacher.camera_head, "camera_head")
```

- `stride = 24 // 12 = 2`，對應論文的 j = 2i（來源：論文 §3.3）。
- 層剪枝時用 `--student_depth d`，stride 變成 `24 // d`（來源：`distill2/run_layer_pruning.py` L6–8、L67–68）。d = 8、6、4 時 stride 分別是 3、4、6（推算）。
- `_copy` 呼叫的是 `load_state_dict(..., strict=False)`，只印出 missing/unexpected 的數量（來源：同檔 L26–30）。也就是說，形狀以外的不一致不會讓程式中止，這是 `verify_teacher_copy.py` 存在的原因（§2.4）。
- 存檔格式是 `{"model": state_dict, "meta": {...}}`（來源：同檔 L79–81）。`meta` 會被 `eval.build_model` 用來重建架構（來源：`distill2/eval.py` L20–31）。

### 2.2 參數對應表（`--copy all`）

| student 張量群 | ← teacher | 張量數 | 訓練時 |
|---|---|---|---|
| `aggregator.patch_embed.*`（DINOv2 ViT-L/14-reg） | 同名 | 344 | 凍結（`train.py --freeze_patch_embed 1`） |
| `aggregator.camera_token` / `register_token` | 同名 | 1 + 1 | 可訓練 |
| `aggregator.frame_blocks.{i}.*`（i = 0…11） | `frame_blocks.{2i}.*` | 216 | 可訓練 |
| `aggregator.global_blocks.{i}.*` | `global_blocks.{2i}.*` | 216 | 可訓練 |
| `camera_head.*` | 同名，整個複製 | 69 | 可訓練 |
| `depth_head.*` / `point_head.*` | 不複製，隨機初始化 | 62 + 62 | depth 由蒸餾學；point 無 loss |

（來源：`distill2/README.md` §四之二；`distill2/verify_teacher_copy.py` L7–13）

- 216 = 12 層 × 每個 Block 18 個張量（推算；Block 的組成見 `vggt/layers/block.py` L50–73）。
- 合計：847 個複製張量加 124 個未複製張量，等於 971（推算）。`weights/vggt_teacher_init_842M.pt`（＝`runs/distill_b1/best.pt`）的 `model` 也正好是 971 個張量，共 842,162,584 個元素（來源：該檔的 pickle 中繼資料）。

**沒有複製的部分**

- **depth/point head**：隨機初始化，沒有固定 seed，兩版程式都不保證逐位元重現（來源：`distill2/README.md` §四之二）。
- **track head**：student 裡不存在。
- **前端不在 `--copy` 裡時**（消融的 random、agg only、head only 列）：`pretrained_patch_embed=True`，會下載官方 DINOv2 ViT-L/14-reg 權重（來源：`vggt/models/aggregator.py` L27–32、L201–202），並在訓練時凍結。這組權重和 teacher 經過 VGGT 訓練的前端不同（來源：論文 Table 1 caption）。

### 2.3 消融與層剪枝如何用這支程式

| 論文列（Table 1） | `run_ablation.py` 設定 | `--copy` | 凍結前端 |
|---|---|---|---|
| random | `random` | `none` | 1 |
| FE only | `patch_only` | `patch_embed` | 1 |
| agg. only | `agg_only` | `aggregator` | 1 |
| head only | `head_only` | `camera_head` | 1 |
| full | `full` | 三者全部 | 1 |
| full, FE tuned | `full_ft_pe` | 三者全部 | 0 |

（來源：`distill2/run_ablation.py` L32–39）

消融與剪枝的訓練參數是 `--eval_every 3`、`--eval_scenes 8`、每 epoch 400 steps、`--img_per_seq 2`（來源：`distill2/run_ablation.py` L51–54、L76–79；`distill2/run_layer_pruning.py` L51–54、L75–78）。README 記錄的消融指令是 `--epochs 12`（來源：`distill2/README.md` §五），`run_ablation.py` 的預設是 15（來源：L51），`run_layer_pruning.py` 的預設是 12（來源：L51）。論文寫「12–15 epochs of 400 steps」（來源：論文 §4.1）。實際上 Table 1 的六個 run 和 Table 3 的 d8/d6/d4 三個 run，`last.pt` 的 `epoch` 都是 11，也就是都訓練了 12 個 epoch（作者開發紀錄；層剪枝這點也寫在 `distill2/reports/layer_pruning_report.md`「訓練 12 epochs × 400 steps」）。表中的數字是各 run `best.pt` 存的 `eval`（來源：`distill2/run_ablation.py` L84–91、`distill2/run_layer_pruning.py` L88–95）。

### 2.4 驗證：`distill2/verify_teacher_copy.py`

驗證腳本依同樣的對應規則，把 student checkpoint 的每個張量和 teacher `model.safetensors` 的原始張量逐位元比對：

`distill2/verify_teacher_copy.py` L34–44
```python
def map_key(k, stride):
    """student key -> 對應的 teacher key；None = build_student.py 不複製。"""
    if k.startswith("aggregator.patch_embed.") or k in ("aggregator.camera_token",
                                                        "aggregator.register_token"):
        return k
    m = re.match(r"aggregator\.(frame_blocks|global_blocks)\.(\d+)\.(.*)", k)
    if m:
        return f"aggregator.{m[1]}.{int(m[2]) * stride}.{m[3]}"
    if k.startswith("camera_head."):
        return k
    return None
```

- student 用 `torch.load(mmap=True)` 讀，teacher 用 `safe_open` 逐張量讀，比對用 `torch.equal`，不需要 GPU（來源：同檔 L15、L54–77）。
- 作者的用法：`python verify_teacher_copy.py --ckpt student_init.pt runs/distill_b1/best.pt`。teacher 權重預設從 HF 快取取 `facebook/VGGT-1B`（來源：同檔 L89–97；`distill2/README.md` §四之二）。
- `student_init.pt` 不在 repo 內。用公開權重驗證訓練後的 842M 模型：先依 [`weights/README.md`](../weights/README.md) 合併兩段，再執行 `python verify_teacher_copy.py --ckpt ../weights/vggt_teacher_init_842M.pt`。要驗證初始化本身，可以用 `build_student.py --copy all` 重新產生 `student_init.pt`（被複製的張量是 teacher 原值，不受隨機初始化影響）。

**2026-10-17 的實跑結果**（來源：`distill2/README.md` §四之二）

| 檔案 | 應複製的 847 個張量 | depth/point head（124 個） |
|---|---|---|
| `student_init.pt` | 全部與 teacher 逐位元相同，0 個不同 | 與 teacher 不同（符合預期） |
| `runs/distill_b1/best.pt`（訓練後，＝`weights/vggt_teacher_init_842M.pt`） | `patch_embed` 344 個仍相同，證實凍結有效；其餘 503 個已被訓練更新 | 不同 |

`student_init.pt` 是用較早版本的 `build_student.py` 產生的。和現在的版本相比，`--copy all` 時複製的元件、層對應、`strict=False` 的做法都一樣；其餘差異（teacher/student 建構順序對調、移除層數整除 assert、`embed_dim`/`num_heads` 寫死成 1024/16、存檔 `meta` 多一個 `copy` 欄位）不影響被複製的張量（來源：`distill2/README.md` §四之二）。`weights/vggt_teacher_init_842M.pt` 的 `meta` 也確實沒有 `copy` 欄位（來源：該檔的 pickle 中繼資料）。

### 2.5 主線訓練配方（`distill2/train.py`）與論文 §4.1 對照

| 項目 | 程式 | 論文 §4.1 |
|---|---|---|
| Optimizer | AdamW，lr `5e-5`，weight decay `0.05`（L45–46、L80） | 相同 |
| LR schedule | 前 5% 線性 warm-up，接著 cosine 降到 `min_lr=1e-7`（L29–33、L47、L87–88） | 「5% linear warm-up then cosine decay」，沒寫 min_lr |
| 精度 | bf16 autocast（L59、L103） | 相同 |
| Gradient clipping | 1.0（L110） | 相同 |
| 每序列幀數 | 預設 4（L43）。`runs/distill_b1` 當時的完整指令沒有存下來（來源：`distill2/README.md` §四）；作者開發紀錄裡的訓練指令用 `--img_per_seq 2`，20 epochs 的 `runs/distill_b1_full` 也是 `--img_per_seq 2`（來源：`distill2/README.md` §四） | 2 |
| Steps / epoch | 500（L42），batch 1 個場景（L44） | 500 |
| Epochs | 預設 40（L41） | 20（不一致，見 §2.6） |
| 前端 | 預設凍結（L51、L69–74） | 凍結 |
| 評估與選模 | 每 2 個 epoch（以及最後一個 epoch）在 test split 前 8 個場景跑 `quick_eval`，`best.pt` 取 `mean_rot_err` 最低的那次（L49–50、L129–140） | 論文沒有描述選模方式 |
| Seed | 不固定全域 seed（來源：`distill2/README.md` §八） | Limitations 寫「no global seed fixed」 |

主線指令（`<DATA_ROOT>` 為佔位符）：

```bash
cd distill2
python build_student.py --out student_init.pt --copy all
python train.py --student_init student_init.pt --data_root <DATA_ROOT> \
    --out_dir runs/distill_b1 --epochs 40 --eval_every 2 --img_per_seq 2
```

`--epochs 40`、`--eval_every 2` 是 `train.py` 的預設值，對應 `runs/distill_b1` 實際的 epoch 數；`--img_per_seq 2` 依論文 §4.1 與作者開發紀錄。訓練好的結果就是 `weights/vggt_teacher_init_842M.pt`。

注意：`best.pt` 是用 test split 前 8 個場景的 rotation error 挑出來的，而 Tables 1、3 和 Table 4 恢復值報告的也是這 8 個場景。也就是選模子集和報告子集相同（由程式碼推論，依據 `distill2/train.py` L132–139、`distill2/train_sliced.py` L148–159）。

### 2.6 不一致：主線 epoch 數

- **論文**：「the main run is 20 epochs of 500 steps」（來源：論文 §4.1）。作者開發紀錄裡的訓練指令也寫 `--epochs 20`。
- **紀錄**：論文用的 842M checkpoint `runs/distill_b1/best.pt`（＝`weights/vggt_teacher_init_842M.pt`）來自 40 個 epoch 的 run。依據有三點：
  1. `train.py` 預設 `--epochs 40`（L41）；
  2. 該 checkpoint 的 `epoch` 欄位是 37（0 起算）（來源：`weights/vggt_teacher_init_842M.pt` 的 pickle 中繼資料），20 個 epoch 的 run 不可能產生這個值；
  3. 同目錄的 `last.pt` 的 `epoch` 欄位是 39，也就是之後還有一次評估（作者開發紀錄；`distill2/README.md` §四也記為「epoch 37（`last.pt` = epoch 39）」）。
- `distill2/README.md` §四已經依 checkpoint 改寫：`runs/distill_b1` 是 train.py 預設的 40 epochs，`--epochs 20` 的是另一個 run `runs/distill_b1_full`（best 為 epoch 19），它只用在 README「七」的 4.9° 那一行，論文的切片、GT、量化、效率表都不是用它。

同一個 checkpoint 在不同評估子集上的數字也不同：前 12 個 test 場景是 2.87°（來源：`distill2/reports/slicegpt_probe.md`），前 8 個是 3.42°（來源：checkpoint 內 `eval.mean_rot_err = 3.4232`；`distill2/reports/quantize_report.md`）。Tables 1、3 的 4.32° 是另一個 checkpoint，也就是消融配方的 `full` run（來源：`distill2/reports/ablation_report.md`；論文 Table 4 caption）。

---

## 3. 資料管線與 canonicalization（`distill2/data.py`）

### 3.1 Soft labels 的產生（`scripts/export_co3d_labels.py`）

這支腳本用 VGGT-1B 對每個 CO3D 場景推理一次，並存下：

- `images/`：PNG，經 `load_and_preprocess_images_square` 前處理到 518（L137、L237–238）；
- `depth.npy`、`depth_conf.npy`；
- `point_mask.npy`，定義為 `depth_pred > 1e-8`（L164）；
- `extrinsics.npy`、`intrinsics.npy`：由 teacher 的 `pose_enc` 經 `pose_encoding_to_extri_intri` 轉出（L157）；
- `pose_enc.npy`、`original_coords.npy`；
- `world_points.npy`（以及 `world_points_conf.npy`）：只有加 `--save_points` 才存（L186）；
- `meta.json`：含 `scene_name`、`num_images`、`image_paths`、`source_images` 等欄位（L195–205）。

每個場景的幀數由 `--max_images_per_scene` 均勻抽樣決定，預設 10（L253）。作者的資料集裡，427 個 train 場景有 421 個是 50 幀（其餘 6 個是 28–49 幀）、65 個 test 場景有 64 個是 50 幀（另 1 個是 29 幀）；各場景 `meta.json` 的 `save_points` 都是 true（作者開發紀錄，依各場景 `meta.json`）。所以建立資料集時用的是 `--max_images_per_scene 50` 加 `--save_points`（推算；`distill2/eval.py` L73 的註解也寫「整段 50 幀」）。資料集規模是 427 個 train 場景、65 個 test 場景、33 個類別（來源：論文 §4.1；`distill2/README.md` §三）。

### 3.2 `TeacherDataset` 的取樣方式

- **train**：每次隨機挑一個場景，再隨機取 `k = min(img_per_seq, n)` 幀並排序。
- **test**：場景依序輪流，幀用 `np.linspace` 等距取。

（來源：`distill2/data.py` L94–102）

資料集長度在 train 是 `length`（等於每 epoch 的 step 數），test 是 `length // 5`（來源：同檔 L78）。

### 3.3 `_normalize`：論文說的「canonicalize + 重算 pose 目標」

第一步，把**第一個取樣幀**設成單位外參：

`distill2/data.py` L49–52
```python
    homog = torch.cat([extr, torch.zeros(S, 1, 4)], dim=1)  # [S,4,4]
    homog[:, 3, 3] = 1.0
    inv0 = closed_form_inverse_se3(homog[:1])  # [1,4,4] cam0->world
    new_extr = torch.matmul(homog, inv0)[:, :3, :].clone()  # [S,3,4] frame0->identity
```

第二步，做尺度正規化：

`distill2/data.py` L54–64
```python
    # scale：把 teacher world_points 轉到 frame0 相機座標後取平均範數（同舊 pipeline）
    scale = torch.tensor(1.0)
    if world_points is not None:
        R0 = extr[0, :3, :3]; t0 = extr[0, :3, 3]
        wnew = world_points @ R0.T + t0          # [S,H,W,3]
        m = point_mask.bool()
        if m.any():
            scale = wnew[m].norm(dim=-1).mean().clamp(min=1e-6, max=1e6)

    new_extr[:, :3, 3] = new_extr[:, :3, 3] / scale
    depth = depth / scale
```

第三步，從正規化後的外參重算 pose 目標：

`distill2/data.py` L66–69
```python
    pose_enc = extri_intri_to_pose_encoding(
        new_extr.unsqueeze(0), intr.unsqueeze(0),
        image_size_hw=image_hw, pose_encoding_type="absT_quaR_FoV",
    )[0]  # [S,9]
```

用式子寫：Ê_i = [E_i; 0001]·[E_0; 0001]⁻¹，所以 Ê_0 = I。尺度 s 是 teacher world points 在 cam-0 座標系下距離的平均值，translation 和 depth 都除以 s。

幾個要注意的地方：

- **為什麼要重算 pose**：canonical 幀是「這次取樣的第一幀」，不是場景的第 0 幀（來源：`distill2/data.py` L12）。每個訓練樣本的參考幀都不同，所以不能直接拿存好的 `pose_enc.npy`（它是相對場景第 0 幀）。論文的理由是直接重用「would leave rotation and translation in different frames」（來源：論文 §3.3）。
- **尺度不影響旋轉**：rotation 目標與 scale 無關，scale 只影響 translation 和 depth（來源：`distill2/data.py` L46）。`depth_conf` 不做縮放。
- **world points 只用來算 scale**：它沒有出現在回傳的 batch 裡（來源：同檔 L138–141），所以 point head 沒有監督目標，和論文 Appendix B 一致。
- **缺 `world_points.npy` 時**：`scale = 1`，translation 和 depth 都不會被正規化（來源：同檔 L55–56、L113–114）。所以 soft labels 必須用 `--save_points` 匯出。
- **`gt_pose=True` 的副作用**：外參改讀 `gt_extrinsics.npy`，由 `distill2/make_gt_labels.py` 用 CO3D 標註和 flip_xy 慣例產生（來源：`distill2/data.py` L106–110）。但 scale 仍然是把 GT 的 `extr[0]` 套在 teacher 的 world points 上算出來的（L57–58），等於混用了兩個座標系。這只影響 translation/depth 的尺度，不影響 rotation 目標（由程式碼推論）。這個選項只用在 GT 微調的旁支實驗（來源：`distill2/reports/gtft_report.md`；論文 §4.3：RRA@15 最多變 0.004、AUC@30 最多變 0.020）。

---

## 4. 蒸餾 loss（`distill2/loss.py`）與 Appendix B

### 4.1 整體結構

總 loss 為：

`total = 4.0·N(L_rot) + 2.0·N(L_T + L_FoV) + 1.0·N(L_depth)`

其中 N(·) 是各項各自的 EMA 正規化（來源：`distill2/loss.py` L74–75、L100–110）。

### 4.2 逐條對照

| Appendix B 的描述 | 程式 |
|---|---|
| refinement stages 以 γ^(n−s−1) 加權，γ = 0.6 | `w = gamma ** (n - s - 1)`（L37）。`n = len(pose_enc_list)`；camera head 預設迭代 4 次（`vggt/heads/camera_head.py` L73），所以權重是 0.216、0.36、0.6、1.0（推算） |
| 單位四元數上的 1 − \|⟨q_pred, q_tgt⟩\| | L42–45 |
| 對連續幀的相對旋轉施加同樣的量，用四元數合成 | L47–53：q_rel = conj(q_s) ⊗ q_{s+1} |
| VGGT 四元數是 scalar-last | `_quat_mul` / `_quat_conj` 採 (x,y,z,w)，實部在 index 3（L8、L16–28） |
| translation 和 FoV 用 L1 | L39–40：`pose_enc[..., :3]` 與 `[..., 7:]` |
| rotation 用自己的 EMA（momentum 0.99、100 warm-up steps），與 translation/FoV 分開 | `_ema_rot` 和 `_ema_tf` 是不同的 buffer（L79–80），`_norm` 見 L84–89。translation 與 FoV 共用一個 EMA、一個權重 |
| 權重 4.0 對 2.0 | `rot_weight=4.0, cam_weight=2.0`（L74、L101–102） |
| depth：teacher-confidence 加權 MSE，只算 valid pixels，權重 1.0 | `depth_loss`（L62–70），valid 遮罩就是 `point_mask` |
| point head 不受監督 | `DistillLoss.forward` 裡沒有 point 項（L91–114） |

camera loss 的核心：

`distill2/loss.py` L36–53
```python
    for s, pred in enumerate(pose_enc_list):
        w = gamma ** (n - s - 1)
        pred = pred.float()
        loss_T = (pred[..., :3] - tgt[..., :3]).abs().mean()
        loss_FL = (pred[..., 7:] - tgt[..., 7:]).abs().mean()

        pq = F.normalize(pred[..., 3:7], dim=-1)
        tq = F.normalize(tgt[..., 3:7], dim=-1)
        dot_abs = (pq * tq).sum(-1).abs().clamp(0, 1)
        loss_R = (1.0 - dot_abs).mean()

        if pq.shape[1] > 1:
            p0, p1 = pq[:, :-1], pq[:, 1:]
            t0, t1 = tq[:, :-1], tq[:, 1:]
            pr = F.normalize(_quat_mul(_quat_conj(p0), p1), dim=-1)
            tr = F.normalize(_quat_mul(_quat_conj(t0), t1), dim=-1)
            dot_rel = (pr * tr).sum(-1).abs().clamp(0, 1)
            loss_R = loss_R + (1.0 - dot_rel).mean()
```

EMA 正規化：

`distill2/loss.py` L84–89
```python
    def _norm(self, loss, ema):
        v = loss.detach().abs().clamp(min=1e-8)
        ema.mul_(self.momentum).add_(v * (1 - self.momentum))
        if self._step < self.warmup:
            return loss
        return loss / (ema + 1e-8)
```

depth loss：

`distill2/loss.py` L64–70
```python
    if mask.sum() < 100:
        return (pred_depth * 0).mean()
    pd = pred_depth.float().squeeze(-1)[mask]
    td = tgt_depth.float()[mask]
    w = tgt_conf.float()[mask].clamp(min=0.0)
    mse = ((pd - td) ** 2) * w
    return mse.mean()
```

### 4.3 程式裡有、論文沒寫的細節

- **stage 加總後除以 stage 數 n**（`rot = tot_rot / n`，L57–58），不是除以權重和。這是常數倍，warm-up 後又會被 EMA 正規化掉。
- **absolute 與 relative 旋轉項是相加，不是平均**（L53）。
- **EMA 在 warm-up 期間也會更新**：buffer 初值是 1.0（L79–81）。到第 100 步時，初值 1.0 在 EMA 裡的殘留權重是 0.99^100 ≈ 0.366（推算）。`_step` 每呼叫一次 loss 加 1（L112）。
- **depth 項也有自己的 EMA**（`_ema_depth`，L109）。論文 Appendix B 只寫了 depth 權重 1.0，沒提到 depth 也經過 EMA 正規化。
- **depth MSE 的平均方式**：除以有效像素數，不是除以 Σw；conf 有 `clamp(min=0)`；有效像素少於 100 時回傳 0（L64–70）。
- **student 的 `depth_conf` 輸出沒有被監督**：`depth_loss` 雖然有 `pred_conf` 參數，但沒有使用（L62–70），所以 student 的 depth confidence 通道不受任何 loss 約束（由程式碼推論）。
- loss 一律轉成 fp32 計算（`.float()`），即使外層開著 autocast。
- **device 修正**：loss 第一次被呼叫時，會把 EMA buffer 搬到 batch 所在的 device（L92–95）。舊版 trainer 曾因沒有對 loss 呼叫 `.to(device)` 出錯（來源：`distill2/CHANGES.md` §3 (A)4）。
- **歷史**：舊版相對旋轉項誤用 scalar-first，使該項梯度變成雜訊（來源：`distill2/CHANGES.md` §3 (A)1；論文 Appendix B）。

`distill2/loss2.py`、`train2.py`、`proj_init.py` 是二次蒸餾（camera-token feature KD、跨幀結構 KD）和 SVD 投影初始化的嘗試，結果都塌縮（約 75°–77°），論文沒有使用（來源：`distill2/README.md` §五；`distill2/proj_init.py` 檔頭；[`HISTORY.md`](HISTORY.md) 階段 3.5、階段 4）。

---

## 5. 光譜探針（`distill2/slicegpt_probe.py`）

### 5.1 做法

- **hook 位置**：aggregator 的 24 個 block 輸出，也就是 `frame[i]`、`global[i]`，i = 0…11（來源：`distill2/slicegpt_probe.py` L59–64）。
- **統計量**：每個位置各自累積一階和 xᵀx（float64），再算**中心化**的共變異數並做特徵分解：

`distill2/slicegpt_probe.py` L51–56
```python
    def pca(self):
        mu = self.s / self.n
        cov = self.c / self.n - torch.outer(mu, mu)
        evals, evecs = torch.linalg.eigh(cov)          # 升冪
        order = torch.argsort(evals, descending=True)
        return mu.float(), evecs[:, order].float(), evals[order].float()
```

- **校準資料**：train split 排序後的前 8 個場景，每個場景等距取 6 幀（預設值，來源：L81–88、L124–125）。
- **截斷**：在每個位置用 forward hook 套用 y ↦ μ + (y − μ)Q_kQ_kᵀ，對應論文 §3.4 的式子。權重完全不動。

`distill2/slicegpt_probe.py` L107–114
```python
        mu, Q, _ = bases[name]
        P = (Q[:, :k] @ Q[:, :k].T).contiguous()       # [C, C] fp32

        def make_hook(mu=mu, P=P):
            def hook(module, inputs, output):
                y = output.float()
                y = (y - mu) @ P + mu
                return y.to(output.dtype)
```

- **評估**：`quick_eval` 跑 test split 前 12 個場景（L126、L147–149）。
- **能量欄**：每個位置前 k 個特徵值佔總和的比例，再對 24 個位置取平均（L141–145、L159）。這就是論文 Table 4 的「Energy」欄。

### 5.2 結果（842M checkpoint `runs/distill_b1/best.pt`，前 12 個 test 場景）

| k | 平均 PCA 能量 | rot_err | RRA@15 |
|---|---|---|---|
| 無截斷 / 1024 | 100.00% | 2.87° | 1.000 |
| 896 | 99.93% | 2.99° | 1.000 |
| 768 | 99.82% | 3.06° | 1.000 |
| 640 | 99.65% | 3.22° | 0.993 |
| 512 | 99.38% | 3.45° | 0.993 |
| 384 | 98.91% | 3.86° | 0.986 |
| 256 | 97.98% | 4.91° | 0.958 |

（來源：`distill2/reports/slicegpt_probe.md`；論文 Table 4 收錄 1024/512/384/256 四列）

用公開權重重跑時，把 `--ckpt` 指到 `../weights/vggt_teacher_init_842M.pt`（先合併兩段）。

### 5.3 探針基底和切片基底的差別

論文明確要求兩者不要混為一談（來源：論文 §3.5、Table 4 caption）。

| | 探針（`slicegpt_probe.py`） | 切片（`slicegpt_slice.py`） |
|---|---|---|
| 二階統計 | 中心化共變異數（L53） | 未中心化二階矩 xᵀx/n（L62–65） |
| 基底數量 | 每個 block 位置一個（24 個） | 每條殘差流一個，由該流所有 block 共用 |
| 收集位置 | 每個 block 的輸出 | 第一個 block 的輸入，加上所有 block 的輸出，合併成一個累加器（§6.1） |
| 權重 | 不動，只用 hook 投影 | 摺疊後刪除行與列 |
| 能量 | 99.38% / 98.91% / 97.98%（k = 512/384/256，就是 Table 4 的 Energy 欄） | 共享基底保留的能量是 99.52% / 99.12% / 98.30%（來源：`distill2/reports/slicegpt_slice.md`），論文沒有刊出這組數字 |

---

## 6. 全域投影切片（`distill2/slicegpt_slice.py`）與 Appendix C

### 6.1 校準

一次 forward 就同時收集三條流的統計：aggregator（1024）、DINOv2 前端（1024）、camera head trunk（2048）（來源：`distill2/slicegpt_slice.py` L68–101）。

`distill2/slicegpt_slice.py` L78–88
```python
    handles = [
        agg.frame_blocks[0].register_forward_pre_hook(lambda m, inp: accs["agg"].bump(inp[0])),
        vit.blocks[0].register_forward_pre_hook(lambda m, inp: accs["pe"].bump(inp[0])),
        ch.trunk.register_forward_pre_hook(lambda m, inp: accs["ch"].bump(inp[0])),
    ]
    for blk in list(agg.frame_blocks) + list(agg.global_blocks):
        handles.append(blk.register_forward_hook(lambda m, i, o: accs["agg"].bump(o)))
    for blk in vit.blocks:
        handles.append(blk.register_forward_hook(lambda m, i, o: accs["pe"].bump(o)))
    for blk in ch.trunk:
        handles.append(blk.register_forward_hook(lambda m, i, o: accs["ch"].bump(o)))
```

基底的計算（float64，特徵值降冪排列）：

`distill2/slicegpt_slice.py` L62–65
```python
    def pca(self):
        evals, evecs = torch.linalg.eigh(self.c / self.n)
        order = torch.argsort(evals, descending=True)
        return evecs[:, order].cpu(), evals[order].cpu()   # float64
```

- **校準資料**：train split 排序後的前 8 個場景，每個場景等距取 6 幀（預設 `--calib_scenes 8 --calib_frames 6`，L469–470）。論文說這 8 個場景是字典序最前面的 train 場景，全部屬於 apple 類別，不和任何評估場景重疊（來源：論文 §3.5）。產生 464M 模型的校準指令沒有覆寫這兩個參數（來源：`distill2/README.md` §五之三），所以實際用的是每場景 6 幀（推算）。論文沒有寫每場景幾幀。
- 三組基底都在**未切片的** 842M 模型上一起算出，存在同一個檔案（L395–397、L431–438）。
- camera head 每次 forward 會呼叫 trunk 4 次（迭代 refinement，來源：`vggt/heads/camera_head.py` L110–126），所以 trunk 的統計包含 4 次迭代的輸入與輸出（由程式碼推論）。

### 6.2 符號對應

- 論文用列向量：y = xW + b，W ∈ R^{d_in×d_out}。PyTorch 的 `nn.Linear` 存的 `weight` ∈ R^{d_out×d_in}，計算是 y = x·weightᵀ + b。所以論文的 W 等於程式的 `weight.T`。
- 論文的 Q_k ∈ R^{d×k} 就是程式的 `Q[:, :k]`。v = Q_kᵀ1 對應 `Qk.sum(0)`。
- 論文的 M = Q_kᵀ − v1ᵀ/d ∈ R^{k×d}，是程式裡 `M = Qk - 1vᵀ/d`（d×k）的轉置：

`distill2/slicegpt_slice.py` L186–191
```python
def _qvm(Q, k, d):
    """共用的旋轉幾何量：Q_k、v = Q_kᵀ1、M = Q_k − 1vᵀ/d。"""
    Qk = Q[:, :k].contiguous().double()
    v = Qk.sum(0)
    M = Qk - torch.ones(d, dtype=torch.float64).outer(v) / d
    return Qk, v, M
```

### 6.3 Appendix C 七條規則在程式裡的位置

| 規則（論文 Appendix C） | 程式 | 備註 |
|---|---|---|
| **Rule 1** 寫入殘差流的層：W ← WQ_k，b ← bQ_k | `_fold_out`（L175–179），用在 `attn.proj` 和 `mlp.fc2`（L202、L205）；前端 conv stem 的輸出通道（L229–238）；aggregator 入口 adapter `pe_lin = _lin(Qk.T)`（L276–277）；camera trunk 入口 `entry = _lin(Qk.T)`（L259–260） | VGGT 原本在前端與 aggregator 之間、以及 trunk 入口都**沒有**線性層。這兩個 adapter 是新加的無 bias `nn.Linear`，初值就是 Q_k，等於把 Rule 1 套在單位矩陣上 |
| **Rule 2** 讀取殘差流的層：W ← Q_kᵀW | 程式裡沒有單獨套用 Rule 2 的地方 | 每個讀取者（`qkv`、`fc1`、`pose_branch.fc1`）前面都有 LayerNorm，所以一律走 Rule 4(ii) 的 `_fold_in`。head 讀殘差流由 Rule 6 處理 |
| **Rule 3** 存在流裡的參數：p ← pQ_k | aggregator 的 `camera_token`、`register_token`（L272–274）；前端的 `cls_token`、`register_tokens`、`pos_embed`、`mask_token`（L220–227） | aggregator 沒有加性位置嵌入。它的 RoPE 在 attention 內作用於 q/k 的 head 空間，切片不會碰到（`SlicedAttention` L137–139） |
| **Rule 4** LayerNorm：(i) 換成只除以 σ 的正規化；(ii) 把中心化、γ 和旋轉摺進下一層，W ← M diag(γ) W；(iii) b ← b + βW | `SlicedNorm`（L108–119）＋ `_fold_in`（L168–172），在 `_surgery_block` 內套用（L196–207）；前端最後的 LN 摺成出口 adapter（L243–247）；`trunk_norm` 摺進 `pose_branch.fc1`（L262–264） | ε 沿用原 LN 的 eps（L196、L206–207）。`clamp_min(0)` 是額外的數值保護 |
| **Rule 5** LayerScale：W ← W diag(λ) Q_k，b ← (b⊙λ)Q_k，並刪除 LayerScale 模組 | `_fold_out` 的 `ls_gamma`；`ls1/ls2 → nn.Identity()`（L199、L208–209） | 沒有 LayerScale 的 block 視為 λ = 1（`_ls_gamma`，L182–183） |
| **Rule 6** head 介面：用存下的 Q_kᵀ 把 k 維快照 de-slice 回 d 維，head 不動 | buffer `aggregator.deslice_Q`（形狀 k×D）＋ aggregator 的 forward hook（L282–295） | 見 §6.4 |
| **Rule 7** camera head trunk：自己一條 2048 寬的流和自己的 Q′；內部套 Rules 1–5；最後的 pose MLP de-slice 一次 | `slice_camera_trunk_`（L250–264） | adaLN 調變（`poseLN_modulation`、`adaln_norm`）是隨資料變化的逐維運算，無法旋轉，所以留在原 2048 基底（L18–20 說明）。de-slice 是摺進 `pose_branch.fc1` 完成，沒有另外存矩陣 |

**LayerNorm 的摺疊**：

`distill2/slicegpt_slice.py` L116–119（`SlicedNorm.forward`）
```python
    def forward(self, x):
        s2 = (x.pow(2).sum(-1) - (x @ self.v).pow(2) / self.d) / self.d
        sigma = torch.sqrt(s2.clamp_min(0) + self.eps).unsqueeze(-1)
        return x / sigma
```

這和論文的 σ = √(‖x̃M‖²/d + ε) 相等。因為 Q_k 的各行正交，‖x̃M‖² = ‖x̃‖² − (x̃v)²/d（推算，代數推導）。

**輸入側與輸出側的摺疊**：

`distill2/slicegpt_slice.py` L168–179
```python
def _fold_in(linear, gamma, beta, M):
    """輸入側摺疊：W·LN → (W diag(γ) M, b + Wβ)。M = Q_k − 1vᵀ/d，float64。"""
    W = linear.weight.double()
    b = linear.bias.double() if linear.bias is not None else torch.zeros(W.shape[0], dtype=torch.float64)
    return _lin((W * gamma.unsqueeze(0)) @ M, b + W @ beta)


def _fold_out(linear, ls_gamma, Qk):
    """輸出側摺疊：LayerScale + 旋轉 → (Q_kᵀ diag(ls) W, Q_kᵀ(ls⊙b))。"""
    W = linear.weight.double()
    b = linear.bias.double() if linear.bias is not None else torch.zeros(W.shape[0], dtype=torch.float64)
    return _lin(Qk.T @ (W * ls_gamma.unsqueeze(1)), Qk.T @ (ls_gamma * b))
```

把程式的 `weight` 轉置回論文記號：`_fold_in` 等於 W ← M diag(γ) W、b ← b + βW；`_fold_out` 等於 W ← W diag(λ) Q_k、b ← (b⊙λ)Q_k（推算，記號轉換）。

**單一 pre-LN block 的手術**（aggregator、DINOv2、camera trunk 三處共用）：

`distill2/slicegpt_slice.py` L201–209
```python
    qkv = _fold_in(blk.attn.qkv, g1, b1, M)
    proj = _fold_out(blk.attn.proj, ls1, Qk)
    blk.attn = SlicedAttention(blk.attn, qkv, proj)
    blk.mlp.fc1 = _fold_in(blk.mlp.fc1, g2, b2, M)
    blk.mlp.fc2 = _fold_out(blk.mlp.fc2, ls2, Qk)
    blk.norm1 = SlicedNorm(v.float(), d, eps1)
    blk.norm2 = SlicedNorm(v.float(), d, eps2)
    blk.ls1 = nn.Identity()
    blk.ls2 = nn.Identity()
```

**為什麼要另寫 `SlicedAttention`**：原本的 `Attention` 用輸入寬度 C 來 reshape 輸出（`reshape(B, N, C)`，來源：`vggt/layers/attention.py` L69）。切片後輸入寬度變成 k，但 head 數和 head 寬度沒變。`SlicedAttention` 改用 `num_heads * head_dim` 來 reshape（來源：`distill2/slicegpt_slice.py` L141），並沿用原模組的 `q_norm`、`k_norm`、`rope`（L127–130）。它一律呼叫 `scaled_dot_product_attention`（L140）；`Block` 的預設 `fused_attn=True`（來源：`vggt/layers/block.py` L45），所以行為與原本相同。

**camera trunk 的切片**：

`distill2/slicegpt_slice.py` L256–264
```python
    blocks = list(ch.trunk)
    for blk in blocks:
        _surgery_block(blk, Qk, v, M, d)
    entry = _lin(Qk.T)                               # x @ Q_k（進切片座標）
    ch.trunk = nn.Sequential(entry, *blocks)

    tn = ch.trunk_norm
    ch.pose_branch.fc1 = _fold_in(ch.pose_branch.fc1, tn.weight.double(), tn.bias.double(), M)
    ch.trunk_norm = SlicedNorm(v.float(), d, tn.eps)
```

**前端的切片**（論文 §4.6 的負面結果，最終模型沒有使用）：最後一層 LN 摺成輸出回 1024 維的出口 adapter：

`distill2/slicegpt_slice.py` L243–247
```python
    # 最終 LN → SlicedNorm + affine/de-slice 摺成出口 adapter（輸出回 d=1024）
    g, b = vit.norm.weight.double(), vit.norm.bias.double()
    out_lin = _lin(g.unsqueeze(1) * M, b)            # [d, k]
    vit.norm = SlicedNorm(v.float(), d, vit.norm.eps)
    agg.patch_embed = AdaptedPatchEmbed(vit, out_lin)
```

所有摺疊都在 CPU 上以 float64 計算，完成後才搬上 GPU（來源：同檔 L413、L159–165；`train_sliced.py` L81–83、`load_sliced` L373–380 也是先在 CPU 上切）。

### 6.4 Head 介面的 de-slice（Rule 6）

`distill2/slicegpt_slice.py` L283、L287–293
```python
    agg.register_buffer("deslice_Q", Qk.T.float())   # [k, D]
```
```python
    def deslice_hook(mod, inputs, output):
        lst, psi = output
        QT = mod.deslice_Q
        # .to(t.dtype)：autocast 下 matmul 會降成 bf16，但 head 的 LN 需要與
        # 殘差流一致的 dtype（stock 模型此處是 fp32）
        return ([torch.cat([t[..., :k] @ QT, t[..., k:] @ QT], dim=-1).to(t.dtype)
                 for t in lst], psi)
```

- 論文寫 de-slice 發生在「before concatenation」。程式則是在 aggregator 已經把 frame 與 global 串接成 2k 維**之後**，從第 k 維切開，兩半各自 de-slice 再串回 2048 維。兩種做法數學上等價。
- hook 對 aggregator 輸出的所有 12 個張量都做 de-slice（L292–293），雖然 head 只讀其中 4 個：DPT 讀索引 2、5、8、11，camera head 讀的最後一個（索引 11）和 DPT 的最後一個是同一個（來源：§1.2；由程式碼推論）。
- aggregator 流實際上有兩個 k×d 矩陣：入口 adapter `aggregator.patch_embed.lin.weight`（256×1024）和 `aggregator.deslice_Q`（256×1024），兩者都算在 464.4M 裡（來源：checkpoint 形狀，見 §6.7）。論文 Table 6 的註腳把 adapter 的 0.26M 算在 Front end 那一列（來源：論文 Table 6 footnote a）；`deslice_Q` 則落在 Aggregator 的 76.0M 裡（推算，見 §6.7），和 Rule 6「included in the reported parameter totals」一致（來源：論文 Appendix C）。

### 6.5 切片順序與 assertion

前端切片函式一開始就檢查順序：

`distill2/slicegpt_slice.py` L216
```python
    assert not isinstance(vit, AdaptedPatchEmbed), "PE 切片須在 aggregator 切片之前"
```

- **論文**：三條流必須依「前端 → camera trunk → aggregator」的順序切，而且「our implementation asserts this ordering and fails loudly if the streams are sliced in any other order」（來源：論文 Appendix C, Order and verification）。
- **程式**：只檢查「前端在 aggregator 之前」這一個條件。原因是 aggregator 切片會把前端包進 `AdaptedPatchEmbed`。`slice_camera_trunk_` 只動 `model.camera_head`，和另外兩條流沒有共用模組，程式沒有檢查它的位置；它放在哪個順序，結果都一樣（由程式碼推論）。
- 實際呼叫的順序固定是前端 → trunk → aggregator：`apply_slices_`（L360–365）、`check`（L339–341）、`run`（L414–419）都是如此。

### 6.6 等價性測試（`--check`）

`distill2/slicegpt_slice.py` L334–341
```python
    ref = model(images)
    torch.manual_seed(0)
    Q_pe, _ = torch.linalg.qr(torch.randn(D, D, dtype=torch.float64))
    Q_ch, _ = torch.linalg.qr(torch.randn(2048, 2048, dtype=torch.float64))
    Q_agg, _ = torch.linalg.qr(torch.randn(D, D, dtype=torch.float64))
    slice_patch_embed_(model, Q_pe, D)
    slice_camera_trunk_(model, Q_ch, 2048)
    slice_model_(model, Q_agg, D)
```

- 測試在 CPU、fp32 上執行，用第一個 test 場景的第 0 與第 5 幀。三條流都用隨機正交 Q、k 取滿維。印出 `pose_enc` 和 `depth` 的最大絕對差，程式訊息以 1e-3 為通過門檻（L328–348）。
- 紀錄結果：pose max|Δ| = 3.0e-7，depth = 2.6e-6（來源：`distill2/reports/slicegpt_slice_combo.md` L5；論文 §3.5、Appendix C）。
- 測試範圍：k = d 時只驗證 Rules 1–7 的摺疊代數是否正確，不涵蓋 k < d 時的截斷誤差（由程式碼推論）。
- 指令（`--ckpt` 預設是 `runs\distill_b1\best.pt`，用公開權重時改成合併後的 842M 檔）：`python slicegpt_slice.py --check --ckpt ../weights/vggt_teacher_init_842M.pt --data_root <DATA_ROOT>`

### 6.7 存檔內容與 `load_sliced`

`run()` 存下的切片檔（`--save_k`）：

`distill2/slicegpt_slice.py` L433–437
```python
            torch.save({"model": m.state_dict(), "meta": base_meta,
                        "Q_agg": Q, "k_agg": k, "Q_pe": Q_pe, "k_pe": args.k_pe,
                        "Q_ch": Q_ch, "k_ch": args.k_ch,
                        "note": "sliced; 用 slicegpt_slice.load_sliced() 載入"},
                       out / f"sliced_{label}.pt")
```

`train_sliced.py` 的存檔還多了 `epoch`，`best.pt` 另外有 `eval`（來源：`distill2/train_sliced.py` L151–159）。

**論文主模型** `weights/vggt_sliced_464M_agg256_ch512.pt`（repo 根目錄，Git LFS）是 `runs/sliced/recover_agg256_ch512/best.pt` 的原檔複製，大小 1,908,372,429 bytes，SHA256 為 `FE6E2CF735DCCB993560C98D21B2E3DC1357F2EE0186986CB6BFE7C7E8F1B352`（來源：`distill2/README.md` §五之三；`weights/SHA256SUMS.txt`）。讀它的 pickle 中繼資料可得：

| 欄位 | 值 |
|---|---|
| `model` | 861 個張量，共 464,434,584 個元素 |
| `meta` | `{'embed_dim': 1024, 'agg_depth': 12, 'patch_embed': 'dinov2_vitl14_reg', 'num_heads': 16, 'init_from': 'facebook/VGGT-1B'}` |
| `epoch` | 5（第 6 個 epoch，也是最後一個） |
| `Q_agg` / `Q_pe` / `Q_ch` | 1024×1024 / 1024×1024 / 2048×2048，皆為 float64 |
| `k_agg` / `k_pe` / `k_ch` | 256 / 0 / 512 |
| `eval`（前 8 個 test 場景） | `mean_rot_err` 5.1988、`rra_15` 0.96875、`abs_rel` 0.1932、`delta_1` 0.7145 |

- 雖然 `k_pe = 0`，`Q_pe` 仍然存在，因為它是從 `--q_ckpt` 一路帶過來的（來源：`distill2/train_sliced.py` L78、L153）。
- 三個 float64 的 Q 合計 50,331,648 bytes。檔案大小扣掉它們剩 1,858,040,781 bytes，和 `quantize.py` 量到的 state_dict 大小 1858 MB 相符（推算；後者來源：`distill2/reports/quantize_sliced_report.md`）。

**切片後的張量形狀**（來源：同一檔案的中繼資料）

| 位置 | 張量 | 形狀 |
|---|---|---|
| aggregator block | `attn.qkv.weight` | (3072, 256) |
| | `attn.proj.weight` | (256, 1024) |
| | `mlp.fc1.weight` | (4096, 256) |
| | `mlp.fc2.weight` | (256, 4096) |
| | `norm1.v` / `norm2.v` | (256,) |
| | `attn.q_norm.weight` | (64,)，未變 |
| aggregator 介面 | `aggregator.patch_embed.lin.weight` | (256, 1024) |
| | `aggregator.deslice_Q`（buffer） | (256, 1024) |
| camera head | `trunk.0.weight`（新加的入口） | (512, 2048) |
| | trunk block `qkv.weight` | (6144, 512) |
| | trunk block `proj.weight` | (512, 2048) |
| | trunk block `fc1.weight` / `fc2.weight` | (8192, 512) / (512, 8192) |
| | `trunk_norm.v` | (512,) |
| | `pose_branch.fc1.weight` | (1024, 512) |
| | `poseLN_modulation.1.weight` | (6144, 2048)，未切 |
| | `token_norm.weight` | (2048,)，未切 |

- 每個 aggregator block 從 18 個張量變成 14 個：LN 的 weight/bias 與 LayerScale 消失，新增 `v` buffer（推算：`frame_blocks` 共 168 = 12 × 14 個張量，`global_blocks` 相同）。
- 依元件加總：前端 304.373M ＋ adapter 0.262M = 304.6M；aggregator 76.0M（含 `deslice_Q` 0.262M 與 `v` buffer 0.012M，只算參數是 75.69M）；camera head 64.594M（含 `v` buffer 0.005M）；depth/point head 各 9.620M（推算自該檔中繼資料）。這與論文 Table 6 的 304.6M / 76.0M / 64.6M / 9.6M / 9.6M 一致。
- `v` buffer 與 `deslice_Q` 合計 279,040 個元素，這些不是參數。README 記載參數量是 464.16M，state_dict 是 464.43M（來源：`distill2/README.md` §五之三）。

**載入方式**：

`distill2/slicegpt_slice.py` L372–376（`load_sliced`）
```python
    ck = torch.load(path, map_location="cpu")
    m = build_model(ck.get("meta", {}), "cpu")
    m.eval()
    apply_slices_(m, ck)
    missing, unexpected = m.load_state_dict(ck["model"], strict=False)
```

- 流程是：先依 `meta` 建出原始結構（隨機初始化，`pretrained_patch_embed=False`，不會下載 DINOv2），再依存檔裡的 Q 和 k 重做一次手術，得到切片後的模組形狀，最後用 state_dict 覆蓋全部權重。`v` 和 `deslice_Q` 也在 state_dict 裡，同樣會被覆蓋。
- 所以存檔裡的 Q 只用來重建結構，最終權重完全來自 state_dict（由程式碼推論）。
- 一定要用 `load_sliced()` 載入，例如在 `distill2/` 下 `load_sliced("../weights/vggt_sliced_464M_agg256_ch512.pt", "cuda")`。直接建 `VGGTStudent` 再 `load_state_dict` 會對不上（來源：`distill2/README.md` §五之三；[`weights/README.md`](../weights/README.md)）。
- `apply_slices_` 也接受只有 aggregator 的舊格式鍵 `Q` / `k`（L356–365）。`quantize.load_any` 會偵測 `Q` 或 `Q_agg` 鍵，自動改用 `load_sliced`（來源：`distill2/quantize.py` L28–36）。

### 6.8 不重訓的切片結果（前 12 個 test 場景）

| k（只切 aggregator） | 共享基底能量（論文未刊） | rot_err | RRA@15 | storage |
|---|---|---|---|---|
| 原模型 | 100.00% | 2.87° | 1.000 | 842M |
| 512 | 99.52% | 5.62° | 0.958 | 692M |
| 384 | 99.12% | 34.60° | 0.382 | 654M |
| 256 | 98.30% | 49.74° | 0.264 | 616M |

（來源：`distill2/reports/slicegpt_slice.md`）

`storage` 是 state_dict 的元素總數，包含 buffer（來源：`distill2/slicegpt_slice.py` L299–300），也就是論文 Table 4 的 Params 欄。

---

## 7. 恢復蒸餾（`distill2/train_sliced.py`）與 INT8（`distill2/quantize.py`）

### 7.1 恢復蒸餾

**流程**：從 `--q_ckpt` 取出完整的 Q，在 `--base_ckpt`（預設 `runs\distill_b1\best.pt`）上依命令列給的 k 切片：

`distill2/train_sliced.py` L76–82
```python
        ref = torch.load(args.q_ckpt, map_location="cpu")
        meta = ref.get("meta", {})
        qs = {kk: ref[kk] for kk in ("Q", "Q_agg", "Q_pe", "Q_ch") if kk in ref}
        del ref

        model = load_student(args.base_ckpt, "cpu")
        apply_slices_(model, qs, k_agg=args.k, k_pe=args.k_pe, k_ch=args.k_ch)
```

**凍結範圍**：只凍結 DINOv2 本體。程式會沿著 adapter 鏈往內找到 ViT：

`distill2/train_sliced.py` L87–94
```python
    if not args.train_pe:
        pe = model.aggregator.patch_embed
        while isinstance(pe, AdaptedPatchEmbed):
            pe = pe.pe
        n_frozen = 0
        for p in pe.parameters():
            p.requires_grad = False
            n_frozen += p.numel()
```

- 入口 adapter、所有 block、camera/register token、各 head 都可訓練（來源：同檔 L9）。
- `deslice_Q` 和 `SlicedNorm.v` 是 buffer，不是參數，所以恢復期間固定不變（由程式碼推論）。
- 論文 Table 6 把前端那一列（含 adapter 的 0.26M）標為「inherited, frozen」，但 adapter 在恢復時其實可訓練。

**配方對照**

| 項目 | 程式預設 | 論文 §3.5 |
|---|---|---|
| Epochs × steps | 6 × 400（L41–42），共 2400 steps（推算） | six epochs of 400 steps |
| 學習率 | `2e-5`，weight decay `0.05`，warm-up 5%＋cosine（L44–46、L119） | 2 × 10⁻⁵ |
| 每序列幀數 | 2（L43） | — |
| Loss | 同 `DistillLoss`（L102） | 「under the objective above」 |
| 凍結範圍 | DINOv2 本體（L86–97） | 「front end frozen and the rest trainable」 |
| 評估與選模 | 每 2 epoch（以及最後一個 epoch）在前 **8** 個 test 場景評估，`best.pt` 取 rot_err 最低（L48–49、L146–160） | 論文 §4.1 寫 Table 4 全用前 12 個場景（不一致，見下） |

**論文主模型的產生指令**（來源：`distill2/README.md` §五之三；路徑已改成佔位符與正斜線）：

```bash
# 1) 在 842M 模型上一次校準三條流的基底並存檔
#    --ckpt 原本是預設的 runs/distill_b1/best.pt，公開權重請用合併後的 ../weights/vggt_teacher_init_842M.pt
#    原始執行寫到 reports/slicegpt_slice_combo.md（之後手動整理過），重跑請換檔名以免蓋掉
python slicegpt_slice.py --run --ckpt ../weights/vggt_teacher_init_842M.pt --data_root <DATA_ROOT> \
    --ks 0,256 --k_pe 512 --k_ch 1024 --bench --save_k 256 --save_dir runs/sliced \
    --out reports/slicegpt_slice_combo_rerun.md
#    → runs/sliced/sliced_agg256+pe512+ch1024.pt（下一步只取裡面的 Q，k 重新指定）

# 2) 切 aggregator 256 + camera trunk 512（前端不切），恢復蒸餾 6 epochs
#    --base_ckpt 原本也是預設的 runs/distill_b1/best.pt
python train_sliced.py --k 256 --k_ch 512 --base_ckpt ../weights/vggt_teacher_init_842M.pt \
    --q_ckpt "runs/sliced/sliced_agg256+pe512+ch1024.pt" --data_root <DATA_ROOT> \
    --out_dir runs/sliced/recover_agg256_ch512 \
    --epochs 6 --img_per_seq 2 --eval_every 2 --throttle 0.05
#    → runs/sliced/recover_agg256_ch512/best.pt（＝ weights/vggt_sliced_464M_agg256_ch512.pt）
```

`--out_dir` 在 Windows 上就是預設值（L62–64）；在 Linux/macOS 上要明確給。不需要重新校準時，也可以直接用公開的 464M 檔：它本身就存了完整的 `Q_agg` / `Q_pe` / `Q_ch`（§6.7）。

其他選項：

- `--train_pe`：連 DINOv2 一起訓練，對應 Table 4 的「FE tuned (8 ep.)」列；
- `--gt_pose`：pose 目標改用 CO3D GT；
- `--resume_sliced`：從已切片的 checkpoint 續訓。

（來源：`distill2/train_sliced.py` L51–57）

**恢復後的結果**

| 設定 | 不重訓（12 場景） | 恢復後（8 場景） | RRA@15（對 teacher） | storage |
|---|---|---|---|---|
| agg k=384 | 34.60° | 4.18° | 1.000 | 654M |
| agg k=256 | 49.74° | 4.95° | 0.990 | 616M |
| agg256 + trunk1024 | — | 5.36° | 0.948 | 516M |
| **agg256 + trunk512** | — | **5.20°** | **0.969** | **464.4M** |
| agg256 + trunk1024 + fe512，FE 凍結 | 74.75° | 69.4° | 0.177 | 365M |
| agg256 + trunk1024 + fe512，`--train_pe` 8 ep | — | 77.2° | 0.188 | 365M |
| agg256 + trunk512 + fe768，FE 凍結 | — | 48.1° | 0.208 | ~390M |

（來源：`distill2/reports/slicegpt_slice.md`、`distill2/reports/slicegpt_slice_combo.md`；論文 Table 4）

**不一致一：評估子集**

- 論文說 Table 4 全部用前 12 個 test 場景（7 個類別）（來源：論文 §4.1、Table 4 caption）。
- 紀錄顯示只有探針值、不重訓值和 2.87° 是 12 個場景。所有恢復後的數字（4.18、4.95、5.36、5.20、69.4、77.2、48.1）以及 INT8 的 5.20° → 5.21° 都是前 8 個場景（4 個類別：apple、backpack、ball、banana 各 2 個；作者開發紀錄，依 test split 排序後的場景名稱）。依據是 `train_sliced.py` 預設 `--eval_scenes 8`（L49），以及報告寫的「恢復蒸餾 eval 8 test scenes」（來源：`distill2/reports/slicegpt_slice.md`、`slicegpt_slice_combo.md` L3–4）；INT8 檢查用 `quantize.py --num_scenes 8`（來源：`distill2/reports/quantize_sliced_report.md`「rot_err 取 8 場景平均」）。Tables 1、3 也是 8 個場景（`run_ablation.py` L54、`run_layer_pruning.py` L54）。
- 因此論文 §4.5 拿 4.95°（8 個場景）和探針的 4.91°（12 個場景）相比，等於跨了兩個子集。

**不一致二：參數量**

早期的報告與註解把 agg256+ch512 寫成 466M，例如 `distill2/reports/slicegpt_slice_combo.md`、`distill2/reports/gtft_report.md`、`distill2/measure_vram_curve.py` 檔頭。同一個 checkpoint 的正確數字是 464.4M（state_dict 元素數），論文用的也是 464M / 464.4M（來源：`distill2/README.md` §八；`weights/vggt_sliced_464M_agg256_ch512.pt` 的中繼資料）。

### 7.2 INT8 weight-only

`distill2/quantize.py` L69–78（`try_torchao_int8`）
```python
    try:
        from torchao.quantization import quantize_
        try:
            from torchao.quantization import int8_weight_only          # 舊版
            cfg = int8_weight_only()
        except ImportError:
            from torchao.quantization import Int8WeightOnlyConfig      # 新版
            cfg = Int8WeightOnlyConfig()
        quantize_(model, cfg)
        return model, "torchao int8 weight-only (GPU)"
```

- 程式依 torchao 版本走新或舊 API。論文記載的是 torchao 0.17 的 `Int8WeightOnlyConfig`：對稱量化、per-output-channel scale、用權重本身的絕對最大值校準、不需要 activation 資料（來源：論文 §4.7）。
- `quantize_` 沒有傳 `filter_fn`（L77）。

**量化範圍**

| | 論文 §4.7 | 在切片模型中對應的物件 |
|---|---|---|
| 會量化 | 每一個 `nn.Linear`，包括凍結前端裡的 | 全部 214 個 `nn.Linear` 的 weight：前端 ViT 的 `qkv`/`proj`/`fc1`/`fc2`（96 個）、摺疊後的 aggregator `qkv`/`proj`/`fc1`/`fc2`（96 個）、入口 adapter `aggregator.patch_embed.lin`，以及 camera head 的 21 個（trunk 入口 `camera_head.trunk.0`、4 個 trunk block 的 16 個、`embed_pose`、`poseLN_modulation.1`、摺進 trunk de-slice 的 `pose_branch.fc1`、`pose_branch.fc2`）（來源：`weights/vggt_sliced_464M_int8.pt` 的中繼資料：214 個張量是量化後的型別，其餘 647 個是一般張量） |
| 不量化 | 卷積（前端 stem、DPT decoder）、正規化參數、bias、camera/register token、de-slicing 矩陣；activation 保持 bf16 | `aggregator.deslice_Q`（buffer，在 hook 裡用 matmul，L292）、各 `SlicedNorm.v` buffer、前端 conv stem、tokens、所有 bias（來源：同一檔案的中繼資料） |

注意：camera trunk 的 de-slice 摺在 `nn.Linear` 裡，所以它會隨該層一起被量化。論文說「de-slicing matrices stay unquantized」，實際上只對以獨立張量存在的 `aggregator.deslice_Q` 成立（來源：同一檔案的中繼資料，`camera_head.pose_branch.fc1.weight` 是量化張量）。

**檔案大小的量法**：

`distill2/quantize.py` L39–42
```python
def size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6
```

這和論文「bytes written by torch.save of the state dict」的定義一致（來源：論文 §4.7）。

**結果**

| | 大小 | rot_err（前 8 個 test 場景） |
|---|---|---|
| 切片模型 fp32 | 1858 MB | 5.20° |
| 切片模型 INT8 | 537 MB | 5.21° |

壓縮倍數 3.46×（來源：`distill2/reports/quantize_sliced_report.md`）。論文 §4.7 說 INT8 之後沒有重新量測 GT 指標與 depth 指標；GT 指標確實沒有量，但 `distill2/reports/quantize_sliced_report.md` 其實也記了 INT8 模型對 teacher 的 abs_rel（0.193 → 0.193，前 8 個 test 場景）。公開的 `weights/vggt_sliced_464M_int8.pt` 是 537,215,008 bytes（來源：[`weights/README.md`](../weights/README.md)）。

**實作上要注意的事**

- **INT8 存檔要先建結構再載入**：存檔只有 `{"model": ..., "quant": ...}`（L145），沒有 `Q_*`、`k_*`、`meta`，所以不能直接交給 `load_sliced`。做法是先用 fp32 檔 `load_sliced` 建出切片結構，套 `quantize_(model, Int8WeightOnlyConfig())`，再 `load_state_dict` INT8 檔的 `model`（`torch.load` 要加 `weights_only=False`），完整程式見 [`weights/README.md`](../weights/README.md)：

  ```python
  # 在 distill2/ 下執行
  import torch
  from torchao.quantization import quantize_, Int8WeightOnlyConfig
  from slicegpt_slice import load_sliced

  mq = load_sliced("../weights/vggt_sliced_464M_agg256_ch512.pt", "cuda")   # 只用來建立結構
  quantize_(mq, Int8WeightOnlyConfig())
  ck = torch.load("../weights/vggt_sliced_464M_int8.pt", map_location="cuda", weights_only=False)
  mq.load_state_dict(ck["model"])
  ```

  也可以從 fp32 權重重新量化（需要 `torchao` 與 GPU；weight-only 量化是確定的，見 `weights/README.md`）。`--out`、`--report` 不給時會寫到目前目錄的 `quantized_int8.pt`、`quantize_report.md`（L103–104），所以重跑請另取檔名（來源：`distill2/README.md` §五之三）：

  ```bash
  python quantize.py --ckpt ../weights/vggt_sliced_464M_agg256_ch512.pt --data_root <DATA_ROOT>/test --num_scenes 8 \
      --out runs/sliced/quantized_sliced_int8_rerun.pt --report reports/quantize_sliced_report_rerun.md
  ```

- **`quantize_*_report.md` 的 VRAM 欄不能拿來比較**：量 INT8 時，基準模型 `base`（L115）沒有釋放，又另外載入一份模型去量化（L127），所以 INT8 那一列的峰值記憶體包含基準模型（由程式碼推論）。論文 Table 5 的 1.81 GB 來自 `measure_unified_bench.py`（§8.1）。
- **teacher 的 fp32 大小**：論文寫 5025 MB（來源：論文 §1、§4.7）。repo 裡沒有量 teacher 檔案大小的報告；用 Table 6 的 1256.5M 參數 × 4 bytes 估算約 5026 MB（推算）。無論哪個值，9.4× 都不變（推算：5025/537 與 5026/537 都約為 9.36）。
- 842M 模型的 INT8 結果（3369 MB → 915 MB，3.42° → 3.42°）只出現在作者紀錄（`distill2/reports/quantize_report.md`、`distill2/README.md` §七），論文沒有使用；這個 842M 的 INT8 檔也不在 repo 內。

---

## 8. 記憶體預算模擬與 FLOP 計數

### 8.1 Table 5 與 Fig. 2：不含 CUDA context 的量測

- **`measure_unified_bench.py`**：在同一個 session 裡依序量 teacher、842M、d8/d6/d4、切片、切片 INT8。條件是 S = 4、518×518、bf16 autocast、`no_grad`，warm-up 5 次後取 20 次平均。VRAM 是 `max_memory_allocated() / 1e9`，單位為 GB（十進位）（來源：L13、L45–61）。結果存在 `distill2/reports/unified_bench.json`，例如 teacher 388.1 ms / 9.21 GB、切片 201.3 ms / 3.89 GB、INT8 229.9 ms / 1.81 GB。
- **`measure_vram_curve.py`、`measure_vram_int8.py`**：S 掃 `[2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 80, 96]`，記錄每個 S 的 allocated 峰值，遇到 OOM 就停（來源：`measure_vram_curve.py` L18、L35–62；`measure_vram_int8.py` L11、L14–41）。Table 5 的「Max frames」是 16 GB 內最大的 S：24 / 48 / 64 / 80（來源：`distill2/reports/vram_curve.json`、`vram_curve_int8.json`；論文 Table 5）。JSON 裡的 `max_frames` / `max_frames_alloc` 欄位（切片 96、842M 96、teacher 64、INT8 96）是「沒有 OOM 的最大 S」，曲線在超過 16 GB 後仍沒有 OOM，所以 Table 5 的值要從 `curve` 讀 ≤ 16 GB 的那一點（推算，依 JSON 的 `curve`）。
- 這三支腳本都不改 `requires_grad`，模型保持載入時的可訓練狀態，也就是 Table 7 說的「Default」（來源：論文 Table 7 caption）。

### 8.2 Table 7：含 CUDA context 的預算模擬（`measure_budget_incl_context.py`）

**模式**：由 `argv[2]` 決定。`frozen` 是預設值；`paper` 就是論文 Table 7 的「Default」。

`distill2/measure_budget_incl_context.py` L47–49
```python
    # MODE "frozen": parameters frozen as in deployment (autocast then keeps no bf16 weight copies).
    # MODE "paper": parameters left as loaded (requires_grad=True), matching the paper's Fig. 2 / Table 6.
    m.requires_grad_(MODE == "paper")
```

- 註解裡的「Table 6」是早期草稿的編號，指的是最終版的 Table 5。
- frozen 模式之所以比較省記憶體，依程式註解是 autocast 不會保留 bf16 權重副本。INT8 模型在兩種模式下的數字完全相同（來源：兩份 JSON）。
- 指令：`python measure_budget_incl_context.py sliced_int8,sliced,pathb,teacher paper`（這是作者當時的指令；程式裡的 checkpoint 路徑寫死成 `runs\...`，用公開權重前要改成 `../weights/...`，見 §0。輸出會直接改寫 `reports/budget_incl_context_{frozen,paper}.json`，重跑前請先備份或改 L66 的檔名）。不給參數時是四個模型、frozen 模式（L64–66、L77）。

**CUDA context 的量法**：在本行程碰 CUDA 之前，先用 `nvidia-smi` 讀一次已用記憶體。如果其他行程已經用掉超過 4500 MiB，就直接中止（L82–85）。接著：

`distill2/measure_budget_incl_context.py` L86–94
```python
    torch.cuda.init()
    a = torch.randn(1024, 1024, device="cuda")
    (a @ a).sum().item()                              # cuBLAS handle/workspace
    conv = torch.nn.Conv2d(3, 8, 3).cuda()
    conv(torch.randn(1, 3, 64, 64, device="cuda")).sum().item()   # cuDNN
    torch.cuda.synchronize()
    used_after, _ = smi_used_mib()
    reserved = torch.cuda.memory_reserved() / 2 ** 20
    ctx_mib = (used_after - base_used) - reserved
```

兩次執行量到的 context 都是 303.0 MiB，卡片總量 16303 MiB（來源：`distill2/reports/budget_incl_context_frozen.json`、`budget_incl_context_paper.json`）。

**上限設定與 S 掃描**：

`distill2/measure_budget_incl_context.py` L111–123
```python
        for B in BUDGETS:
            cap = B * GB - ctx_mib * 2 ** 20
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(max(cap, 0) / total_bytes, 0)
            best = 0
            for s in S_LIST:
                try:
                    fwd(model, s, "cuda", torch.bfloat16)
                    best = s
                except torch.cuda.OutOfMemoryError:
                    break
                finally:
                    torch.cuda.empty_cache()
```

- `GB = 1024 ** 3`，所以預算單位是 GiB（L23）。Table 5 和 Fig. 2 用的是十進位 GB（`/ 1e9`），兩者單位不同。
- 上限限制的是 allocator 的 **reserved** 總量（包含模型權重），不只是 allocated（來源：論文 Appendix D）。
- 每個 S 的輸入是隨機影像，在 `no_grad` 加 bf16 autocast 下 forward（L55–61）。
- 結果 0 代表連 S = 1 都放不下。
- `BUDGETS = [4, 6, 8, 12, 16]`（L21）。16 GiB 只出現在 JSON 裡，論文省略了，理由是它超過卡片實際可用的記憶體（來源：論文 Appendix D）。

**結果**

| 模型 | Default 4 / 6 / 8 / 12 GiB | Frozen 4 / 6 / 8 / 12 GiB |
|---|---|---|
| VGGT-1B teacher | 0 / 0 / 1 / 8 | 0 / 1 / 4 / 12 |
| Teacher-init（842M） | 0 / 4 / 12 / 32 | 1 / 4 / 20 / 48 |
| 切片（464M） | 2 / 8 / 24 / 48 | 4 / 16 / 32 / 48 |
| 切片 + INT8 | 8 / 20 / 32 / 56 | 8 / 20 / 32 / 56 |

（來源：`distill2/reports/budget_incl_context_paper.json`、`budget_incl_context_frozen.json`；論文 Table 7）

**S 網格**：論文描述為「{1, 2, 4, 8, 12, 16, 20, …} in steps of 4 up to 32 and of 8 beyond」（來源：論文 Appendix D）。程式的 `S_LIST` 在 80 之後直接跳到 96，中間沒有 88（L20）。Table 7 刊出的最大值是 56，所以論文數字不受這個缺口影響（推算）。

**CPU 延遲**：只在 frozen 模式量，模型是切片與 teacher，fp32、S = 2，warm-up 1 次後取 3 次平均，執行緒數是 `torch.get_num_threads()`，實際為 8（L137–150；JSON）。結果是切片 7403.1 ms、teacher 16233.7 ms（來源：`distill2/reports/budget_incl_context_frozen.json`），論文寫成 7.4 s 對 16.2 s、2.19×（來源：論文 Appendix D）。

### 8.3 FLOP 計數

`distill2/measure_budget_incl_context.py` L104–108
```python
        if name != "sliced_int8" and MODE == "frozen":   # device-independent compute measure, S=4
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(display=False) as fc:
                fwd(model, 4, "cuda", torch.bfloat16)
            res[name]["gflops_S4"] = round(fc.get_total_flops() / 1e9, 1)
```

- 這支腳本量到 teacher 16814.3、842M 9919.2、切片 7496.8 GFLOPs（來源：`distill2/reports/budget_incl_context_frozen.json`）。
- 另外兩列由 `measure_flops_extra.py` 用同樣的方法量：d8 剪枝 8194.1，conv-512 scratch student 2399.8（來源：`distill2/reports/flops_extra.json`）。scratch student 用隨機初始化建立，因為 FLOPs 和權重值無關（來源：`distill2/measure_flops_extra.py` L2–4、L28–29）。
- **INT8 沒有量**（L104 排除）。Table 5 把它填成和切片相同，論文 caption 也明寫「INT8 equals sliced, as quantization is weight-only」。
- `distill2/reports/efficiency_report.md` 的 GMACs（例如 842M 是 3666.1）是用 fvcore 估的，報告自己註明注意力等自訂算子可能被低估。那是不同的工具，論文沒有使用，兩者不能直接比較。
- 論文沒有寫 GFLOPs 比例。例如 16814.3 / 7496.8 = 2.24×（推算）。

### 8.4 跨 session 的延遲漂移

- **論文**：「run-to-run drift across sessions is within about 7%」（來源：論文 §4.7）。
- **紀錄**：同一模型跨 session 差得更多（以下百分比都以較慢那次為分母，推算）：
  - scratch student 在 `distill2/reports/efficiency_report.md` 是 123.8 ms，在 `distill2/reports/efficiency_prune.md` 是 105.2 ms，差 18.6 ms，約 15.0%（以較快那次為分母是 17.7%）。
  - 842M 在 `efficiency_report.md` 是 233.9 ms，在 `distill2/reports/quantize_report.md` 是 214.8 ms，差 19.1 ms，約 8.2%。
  - 切片模型在 `unified_bench.json` 是 201.3 ms，在 `quantize_sliced_report.md` 是 223.9 ms，差 22.6 ms，約 10.1%。`quantize.py` 是 warm-up 4 次、15 次平均（L46），量法也和 `measure_unified_bench.py` 略有不同。
- 所以跨 session 的漂移最多約 15%，比論文寫的 7% 大。
- 論文的 1.77× 和 1.93× 都是同一 session 內的比值（388.1 / 219.1、388.1 / 201.3，推算；來源：`distill2/reports/unified_bench.json`；論文 Table 5）。

---

## 9. 評估程式與論文指標的對應

### 9.1 對 teacher 的指標：`distill2/eval.py` 的 `quick_eval`（Tables 1、3、4）

- **場景**：test split 排序後的前 N 個場景（L102–107）。
- **幀**：每個場景取 `min(12, n)` 個等距幀，而且一定包含第 0 幀：

`distill2/eval.py` L74
```python
    ids = sorted(set(np.linspace(0, n - 1, min(max_frames, n)).astype(int).tolist()))
```

  作者資料集的 test 場景最少有 29 幀（§3.1），所以實際上每個場景都是 12 幀（推算）。
- **旋轉誤差**：student 的 pose 和 teacher 存好的 `pose_enc.npy` 都轉成外參，逐幀計算旋轉角誤差（L81–88）。
- **`rra_15` 是逐幀算的**：它是「每一幀的旋轉誤差小於 15°」的比例，不是對所有幀對計算：

`distill2/eval.py` L98–99
```python
    return {"mean_rot_err": float(rot.mean()), "rra_15": float(np.mean(rot < 15)),
            "abs_rel": abs_rel, "delta_1": delta1}
```

  兩邊的第 0 幀都是參考幀，所以逐幀誤差等於「相對第 0 幀的旋轉」的誤差。第 0 幀本身也計入平均（由程式碼推論）。論文把這個指標也稱為 RRA@15（來源：論文 §4.1），但它和 Table 2 那種對所有幀對計算的 RRA@15 定義不同。
- **depth 指標**：`abs_rel` 和 `δ1` 是在遮罩內直接比較 student 與 teacher 的深度值，沒有做尺度或中位數對齊（L90–94）。訓練目標則是除以 scale 之後的深度（§3.3）。
- **精度**：`quick_eval` 沒有包 `torch.autocast`。影像以 `dtype` 傳入，在 CUDA 上是 bf16（L65、L77）。aggregator 一開始會拿影像和 fp32 的正規化常數相減，型別升回 fp32，所以 forward 實際以 fp32 計算（由程式碼推論，依據 `vggt/models/aggregator.py` L246）。
- 分數是先在每個場景內平均，再對場景取 `nanmean`（L107）。

### 9.2 對 CO3D GT 的指標：`distill2/eval_gt.py`（Table 2）

- **相對位姿誤差**：對所有 i < j 的幀對，計算相對旋轉誤差與相對平移方向誤差（L45–62）。所有場景的幀對誤差先串接起來再算比例（L196–199），也就是依幀對數加權，不是先對每個場景平均（由程式碼推論）。
- **AUC@30**：只用旋轉，是 1° 到 30° 各門檻下 RRA 的平均：

`distill2/eval_gt.py` L67
```python
    auc = float(np.mean([(re < t).mean() for t in range(1, 31)])) if len(re) else float("nan")
```

  論文也註明這個 AUC@30 不能和同時計入平移的數字比較（來源：論文 §4.1）。
- **GT 座標轉換**：CO3D（PyTorch3D 慣例）轉 OpenCV 用 `flip_xy`（L74–89）。teacher 的外參用影像尺寸 (518, 518) 解碼（L136–139）。
- **幀數**：`--frames` 預設 8（L147），但論文的數字是每場景 10 幀，共 65 個場景（來源：`distill2/reports/eval_gt_co3d10_*.md`「每場景 10 幀」；論文 Table 2）。`--random_frames` 是隨機取幀的穩健性檢查（L151–152；`distill2/reports/eval_gt_co3d10rand_*.md`）。
- 指令範例（在 `distill2/` 下；`--out_prefix` 預設會寫到目前目錄的 `eval_gt_report.md`，L153、L220）：

  ```bash
  python eval_gt.py --ckpt ../weights/vggt_sliced_464M_agg256_ch512.pt --data_root <DATA_ROOT>/test \
      --num_scenes 65 --frames 10 --ref co3d --co3d_root <CO3D_ROOT> \
      --out_prefix reports/eval_gt_co3d10_ch512_rerun
  ```

### 9.3 各數字用的評估子集

| 子集 | 用在哪裡 | 程式依據 |
|---|---|---|
| 全部 65 個 test 場景 × 10 幀 | Table 2（GT） | `eval_gt.py --num_scenes 65 --frames 10` |
| 前 12 個 test 場景（7 類） | 探針、不重訓的切片、2.87° | `slicegpt_probe.py` L126、`slicegpt_slice.py` L471 |
| 前 8 個 test 場景（4 類） | Tables 1、3；Table 4 所有恢復值；INT8 的 5.20° → 5.21°；訓練中的選模 | `train.py` L50、`run_ablation.py` L54、`run_layer_pruning.py` L54、`train_sliced.py` L49、`quantize.py` L99 |

（7 類：論文 §4.1；4 類：作者開發紀錄，依 test split 排序後的場景名稱，見 §7.1）

---

## 10. 程式與論文描述的差異總表

| # | 主題 | 論文 | 程式與紀錄 | 來源 |
|---|---|---|---|---|
| 1 | 主線 epoch 數 | 20 epochs × 500 steps | `runs/distill_b1/best.pt`（＝`weights/vggt_teacher_init_842M.pt`）來自 40 epochs 的 run（`train.py` 預設 40、checkpoint `epoch` = 37、`last.pt` 是 39）；作者開發紀錄裡的指令寫 20，20 epochs 的是另一個 run `runs/distill_b1_full` | §2.6；`distill2/README.md` §四 |
| 2 | Table 4 的評估子集 | 全部是前 12 個 test 場景 | 只有探針、不重訓值和 2.87° 是 12 個；恢復值和 INT8 檢查是前 8 個；Tables 1、3 也是 8 個 | §7.1、§9.3 |
| 3 | 切片模型的參數量 | 464M / 464.4M | 早期報告寫 466M（錯誤）；state_dict 是 464,434,584 個元素（參數 464.16M） | §7.1 |
| 4 | 延遲漂移 | 「within about 7%」 | 同一模型跨 session 最多約 15%（scratch）；842M 約 8%、切片約 10% | §8.4 |
| 5 | teacher 的 fp32 大小 | 5025 MB | repo 內沒有對應的量測；1256.5M × 4 bytes 推算約 5026 MB；9.4× 不變 | §7.2 |
| 6 | 切片順序 assertion | 程式會檢查順序，任何其他順序都會報錯 | 只 assert「前端在 aggregator 之前」；camera trunk 的位置不檢查（它不與其他流共用模組） | §6.5 |
| 7 | Rule 7 的校準位置 | 「collected at the trunk input」 | trunk 輸入加上 4 個 trunk block 的輸出，而且涵蓋 4 次 refinement 迭代 | §6.1 |
| 8 | Rule 1 / Rule 7 的入口層 | 稱作「the linear that lifts front-end features」「its input projection」 | VGGT 原本沒有這兩層；程式新增無 bias 的 `nn.Linear`，初值為 Q_k（`aggregator.patch_embed.lin`、`camera_head.trunk.0`） | §6.3 |
| 9 | Rule 2 | 讀取層 W ← Q_kᵀW | 程式沒有單獨使用；所有讀取者都在 LayerNorm 之後，一律走 Rule 4(ii) | §6.3 |
| 10 | Rule 6 的時機 | de-slice 在串接之前 | 在 aggregator 已串接的輸出上從第 k 維切開，兩半各自 de-slice（數學上等價） | §6.4 |
| 11 | depth loss 的正規化 | 只說 rotation 有獨立 EMA；depth 權重 1.0 | depth 也有自己的 EMA；translation 與 FoV 共用一個 EMA | §4.3 |
| 12 | camera loss 的 stage 加總 | γ^(n−s−1) 加權 | 另外再除以 stage 數 n（常數倍） | §4.3 |
| 13 | student 的 depth confidence | 論文沒提 | `pred_conf` 沒被使用，student 的 depth conf 通道不受監督 | §4.3 |
| 14 | 選模方式 | 論文沒提 | `best.pt` 用 test split 前 8 個場景的 rot_err 挑選，與 Tables 1、3、4 恢復值的報告子集相同 | §2.5、§7.1 |
| 15 | 對 teacher 的 RRA@15 定義 | 稱為 RRA@15 | `eval.py` 是逐幀對 teacher 計算（相對第 0 幀），不是所有幀對；Table 2 的 `eval_gt.py` 才是所有幀對 | §9.1 |
| 16 | 「12 equidistant frames」 | 每場景 12 幀 | 程式是 `min(12, n)` 幀，並包含第 0 幀；作者資料集的 test 場景都有 29 幀以上，所以實際都是 12 幀 | §9.1 |
| 17 | INT8 不量化 de-slicing 矩陣 | 「the de-slicing matrices stay unquantized」 | 只有 `aggregator.deslice_Q`（buffer）不被量化；camera trunk 的 de-slice 摺在 `pose_branch.fc1`（`nn.Linear`）裡，會被量化（`weights/vggt_sliced_464M_int8.pt` 可驗證） | §7.2 |
| 18 | Table 7 的 S 網格 | 32 之後每次加 8 | `S_LIST` 在 80 之後跳到 96，沒有 88；刊出的數值不受影響 | §8.2 |
| 19 | 恢復期間前端是否全凍結 | Table 6 把前端（含 adapter）標為 frozen | DINOv2 本體凍結；0.26M 的入口 adapter 可訓練 | §7.1 |
| 20 | 程式註解裡的表號 | 最終版 Table 5 / Table 7 | 註解寫的「Table 3/6」「Table 6」是早期草稿的編號 | §0 |
