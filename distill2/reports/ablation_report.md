# 消融實驗（Teacher-init 各元件的貢獻）

所有組合皆為同寬度 1024（~842M）；差別只在「初始化」，藉此隔離各元件貢獻。

| 組合 | 複製元件 | freeze PE | rot_err | RRA@15 | abs_rel | delta_1 |
|---|---|---|---|---|---|---|
| random | none | 1 | 76.18° | 0.188 | 0.235 | 0.642 |
| patch_only | patch_embed | 1 | 75.59° | 0.198 | 0.260 | 0.616 |
| agg_only | aggregator | 1 | 75.58° | 0.208 | 0.265 | 0.574 |
| head_only | camera_head | 1 | 75.48° | 0.208 | 0.247 | 0.578 |
| full | patch_embed,aggregator,camera_head | 1 | 4.32° | 1.000 | 0.189 | 0.746 |
| full_ft_pe | patch_embed,aggregator,camera_head | 0 | 75.68° | 0.219 | 0.248 | 0.632 |
