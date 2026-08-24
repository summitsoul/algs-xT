# ALGS xT — Apex 电竞点位价值与 IGL 指挥评分模型

把足球里的 **xT (Expected Threat, 预期威胁)** 迁移到 **ALGS (Apex Legends Global Series)** 职业赛：
量化「此刻某支队伍站在地图某个点位」这件事，未来能转化为多少**排名分 + 击杀分**，并据此给 **IGL（指挥）** 的能力打分——不看他嘴上怎么指挥，看他把队伍带到了多值钱的位置。

## 当前状态

- **xT 模型暂时稳定**：`β_p + φ` 的分解、6 档 `rel_bin`、毒里软处理、速度锚点（守点优先）都定下来了，可作为点位价值的基线继续用。
- **IGL 指挥能力评定仍非常初步**：现在的 L1 站位 / L2 兑现只是第一版尝试，如何真正从 xT 里剥离出「指挥能力」还需要进一步改善——圈刷新后的转移决策、残差口径、与赛果回测校准等方向都还没做扎实。

## 模型

```
xT(点位 p, 第 m 圈) = β_p + φ_m(rel_bin)      当 p 在圈内 (rel ≤ 1)
                   = β_p + φ_m(圈内边缘档)   短暂进出毒(≤15s) 或 圈1/圈2
                   = 0                        圈3+ 持续在毒里 (rel > 1)
```

- **β_p** — 点位固有价值（掩体/地形/可防守性）。ridge 回归（λ=3）+ 队伍固定效应，从「点位停留占比」解出。
- **φ_m(rel_bin)** — 圈相对价值。Bellman 值迭代（γ=0.99），状态 =（圈阶段，相对圈心距离档）。
- **rel_bin** — `rel = 点位到圈心距离 / 圈半径`，分 6 档：圈心 / 圈内 / 圈内边缘 / 圈外贴边 / 圈外 / 远圈外。
- 击杀与排名分开建模：`xT = xT_kill + xT_place`，β 和 φ 各拆两套。

### IGL 评分

```
IGL 分 = 跨场平均 [ Σ_{每圈} ( 你队锚点 xT − 全场同时刻平均 xT ) ]
```

- 只测「**站位**」：同一时刻，你比全场平均站得好多少。不剥落地运气、不除圈运。
- 双层拆解（`scripts/igl/igl_two_layer.py`）：
  - **L1 站位** — 上面的残差，位置选得多好。
  - **L2 兑现** — 队伍固定效应，同一粗位置里谁长期拿得比该位置均值多（位置打出来多少）。
  - 错位案例：VK Gaming（积分第 2，站位为负，兑现强 = 贴边打法）vs NAH（站位为正，兑现弱 = 好位置打不出）。

## 目录结构

```
scripts/
├─ fetch/      抓数据（ALGS 赛事 + replay，浏览器控制台脚本）
├─ etl/        提取 → data/ 中间产物（long_table、点位）
├─ model/      xT 模型拟合 + 诊断（β_p、φ、拆分、交叉验证）
├─ igl/        IGL 评分（下游消费 xT，不改模型）
├─ viz/        可视化（xT 热点图、队伍轨迹动态网页）
└─ analysis/   旧 POI-draft 选点顺序分析报告

data/          模型产物（.npy / .json / .csv，小文件已入库；long_table 等原始大文件见下）
```

## 数据管线

```bash
# 0) 抓数据（需网络 / 浏览器）
python3 scripts/fetch/fetch_data.py          # ALGS 赛事、积分、POI draft
#   replay 数据：在 apexlegendsstatus.com 的 replay 页控制台跑 scripts/fetch/fetch_replay3.js

# 1) 提取
python3 scripts/etl/extract_long.py          # replay JSON → data/long_table.jsonl
python3 scripts/etl/extract_positions.py     # → data/positions.csv（点位）

# 2) 拟合 xT 模型
python3 scripts/model/xT_points.py           # → β_p + φ（写 data/phi.json、*.npy）
python3 scripts/model/phi_split.py           # φ 拆击杀/排名
python3 scripts/model/points_split.py        # β_p 拆击杀/排名

# 3) IGL 评分（亚太南 21 场示例）
python3 scripts/igl/score_igl2.py            # 站位残差榜 → output/igl_apac_s_residual.*
python3 scripts/igl/igl_two_layer.py         # 站位 × 兑现 二维榜

# 4) 可视化（任意一场比赛）
python3 scripts/viz/apply_game_xT.py "replay/storm point/sp_apac-s_d__g6_abfd4cf4.json"   # xT 热点图
python3 scripts/viz/gen_trajectory_demo.py "replay/storm point/sp_apac-s_d__g6_abfd4cf4.json"  # 队伍 xT 轨迹网页
```

## 依赖

- Python 3.8+
- `numpy` `scipy` `matplotlib` `Pillow` `scikit-learn`（`extract_positions.py` 用 DBSCAN）

## 数据来源与复现

- 数据来自 [apexlegendsstatus.com](https://apexlegendsstatus.com) 的 ALGS replay API。
- 原始大文件**未入库**（需重新抓取）：`replay/`（777MB）、`data/long_table.jsonl`（101MB，超 GitHub 单文件上限）、`data/raw/`、`output/`（生成产物）。
- `map/storm point.png`（20MB 游戏地图底图）也**未入库**——出图脚本需要它，放到 `map/` 目录即可。

## 说明

- 圈编号采用 Apex 口径：从**圈 1** 开始，数据里 `stage 0 → 圈1`、`stage 2 → 圈3`，共 6 圈。
- **毒里软处理**：玩家偶尔会短暂穿过毒区（转点/贴边），此时不把 xT 硬压成 0，而是按「圈内边缘档」计值——条件是**连续毒内停留 ≤15s**，或**圈 1/圈 2 之前**（`phase < 2`，早期毒弱、穿越无损）。只有圈 3+ 且持续泡在毒里才清零。
- IGL 评分是 xT 模型的**只读下游**：只消费模型产物（`data/phi.json`、`*.npy`），不重新拟合、不改写模型。
