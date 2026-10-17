# GT 直接微調實驗（中性/負面結果，如實記錄）

假設：恢復/蒸餾以 teacher soft label 為目標，換成 CO3D GT 直接微調或許能提升 GT 指標。

設定：`make_gt_labels.py` 產生全部 492 場景的 `gt_extrinsics.npy`（flip_xy，覆蓋率 100%）；
`--gt_pose 1` 使 pose 目標改用 GT 外參（first-frame 正規化與 depth 監督不變）；
4 epochs、lr 1e-5、從各自最佳權重續訓；GT 評估 65 場景 × 10 幀。

| 模型 | 微調前 RRA@15/RTA@15/AUC@30 | GT 微調後 | 判定 |
|---|---|---|---|
| Path B（842M） | 0.964 / 0.893 / 0.876 | 0.960 / 0.898 / 0.856 | 持平偏降 |
| agg256+ch512（466M） | 0.883 / 0.814 / 0.767 | 0.883 / 0.821 / 0.770 | 持平 |

## 結論

GT 直接微調**無實質增益**（±0.02 內）。原因：teacher 在本資料上 RRA@15=1.000——
teacher soft label 幾乎就是 GT，換目標拿不到新資訊。
**學生與 GT 的差距來自容量/壓縮，不是監督品質**——蒸餾 teacher 已是充分的訓練訊號。
（權重：`runs/gtft/{pathb,sliced}/last.pt`；不更新主比較表。）
