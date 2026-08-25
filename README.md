# ALGS xT — Apex 电竞点位价值与 IGL 指挥评分模型

把足球里的 **xT (Expected Threat, 预期威胁)** 迁移到 **ALGS (Apex Legends Global Series)** 职业赛：
量化「此刻某支队伍站在地图某个点位」这件事，未来能转化为多少**排名分 + 击杀分**，并据此给 **IGL（指挥）** 的能力打分——不看他嘴上怎么指挥，看他把队伍带到了多值钱的位置。

## 当前状态

- **xT 模型暂时稳定**：击杀通道 = 点位×圈的每秒击杀数（斜率），沿轨迹累计积分（单调递增、换点只改斜率不清零）；排名通道 = β_p_place + φ_place（非折现 γ=1.0 线性解、非负），其中 β_p_place 再拆成「掩体/地形 + 转点/连通」两个子项（纯拆解，粗块转移前瞻递归），并在应用时乘两个权重——$w_{\mathrm{zone}}$（圈外衰减）+ $w_{\mathrm{stage}}$（圈阶段递增，早期圈压低点位固有价值）。6 档 `rel_bin`、速度锚点（守点优先）都定下来了，可作为点位价值的基线继续用。
- **IGL 指挥能力评定仍非常初步**：现在的 L1 站位 / L2 兑现只是第一版尝试，如何真正从 xT 里剥离出「指挥能力」还需要进一步改善——圈刷新后的转移决策、残差口径、与赛果回测校准等方向都还没做扎实。
- **粗块合并暂缓**：点位仍是 451 个细点；`scripts/viz/coarse_blocks.py` 可单独预览粗块合并效果，等观察覆盖/稀疏性后再决定是否降参。

## 模型

把一支队伍的轨迹 $t \mapsto \big(p(t),\, m(t)\big)$（时刻 $t$ 处在点位 $p$、第 $m$ 圈）拆成两条通道：

$$
\mathrm{xT}(t) = \underbrace{\mathrm{xT_{kill}}(t)}_{\text{累计击杀}} + \underbrace{\mathrm{xT_{place}}\big(p(t), m(t)\big)}_{\text{排名潜力}}
$$

两条通道**不对称**（刻意为之）：击杀是「即时奖励、随站随攒」，所以是**累计量**；排名分是「终局才兑现的终态奖励」，所以是**当前站位的瞬时量**。

### 击杀通道：斜率 × 累计积分（无回归、无递推）

**第 1 步** — 点位 $p$ 在第 $m$ 圈的历史击杀强度，用**每秒击杀数（斜率）**刻画：

$$
\mathrm{kill\_slope}(p, m) = \frac{K(p, m)}{T(p, m)}
$$

$$
K(p,m)=\sum_{\substack{\text{采样行}\\ \text{落在}(p,m)}}\mathrm{kills},
\qquad
T(p,m)=N(p,m)\cdot\Delta t,\qquad \Delta t = 5\,\text{s}
$$

- $K(p,m)$ — 历史落在「点位 $p$、第 $m$ 圈」的所有采样行的击杀数之和。
- $N(p,m)$ — 该「点位 × 圈」被采样的行数（每行 = 队伍锚点停留一个 $\Delta t$）；$T(p,m)$ 即被站住的总时长。
- 分母用 `max(1, T)` 仅防止 $T=0$ 除零（无收缩）。

击杀只跟地形/掩体 + 当前圈阶段有关——很多点要后期才打得起来，前期待着也拿不到人头——跟转去哪/从哪来无关，所以不需要 Bellman 状态转移。

**第 2 步** — 队伍轨迹上的**累计击杀**（单调递增）：

$$
\mathrm{xT_{kill}}(t) = \int_0^t \mathrm{kill\_slope}\big(p(\tau), m(\tau)\big)\,d\tau
\;\approx\;\sum_{k\,:\,t_k \le t} \mathrm{kill\_slope}(p_k, m_k)\cdot\Delta t
$$

换点只让被积函数里的斜率从 $\mathrm{kill\_slope}(p_a,m)$ 跳到 $\mathrm{kill\_slope}(p_b,m)$，已经累计的部分**不清零**。

> **击杀守恒（自检）**：全量 59 场对斜率积分 $\sum \mathrm{slope}\cdot\Delta t = 3599.0$，等于真实击杀总数 3599，误差 0.000%。$\mathrm{kill\_slope}$ 实测 $0\sim0.30$ 击杀/秒（最热 ≈ 18 头/分钟）。

### 排名通道：非折现 Bellman + 非负

$$
\mathrm{xT_{place}}(p, m) = \max\Big(0,\ w_{\mathrm{zone}}(r)\cdot w_{\mathrm{stage}}(m)\cdot\beta_p^{\mathrm{place}} + \varphi_m^{\mathrm{place}}\big(\mathrm{rel\_bin}(r)\big)\Big),
\qquad
r = \frac{\|a - c_m^{\mathrm{new}}\|}{R_m^{\mathrm{new}}}
$$

$a$ = 速度锚点（过去 15s 位移 < 400 单位的「踩点/守家」玩家质心；全队都在动时退回最靠拢两名的质心）。

- $r$ 用**新圈（安全区）**基准：$c_m^{\mathrm{new}}$ = 该圈 `finishedClosing` 的中心、$R_m^{\mathrm{new}}$ = 该圈 `endRadius`（不是正在缩的毒环半径）。圈刷新后新圈才决定「哪里是圈内」，旧圈（毒环）不再算。
- $w_{\mathrm{zone}}(r)$ — **β_p 圈外平滑衰减权重**：$r\le 1$（新圈内）全额 $=1$；$r$ 从 $1\to1.6$ 线性降到 $0$；$r\ge1.6$（远圈外）归零。理由：点位固有排名价值 $\beta_p$ 只在点位被圈覆盖时才兑现，圈外站不住。**最后圈例外**：`endRadius < 500` 时没有安全区（毒圈缩到一点），$w_{\mathrm{zone}}=1$ 不衰减。
- $w_{\mathrm{stage}}(m)$ — **β_p 圈阶段权重**：$= 1 - R_m^{\mathrm{new}}/R_0$（$R_0=31000$ = 圈1 半径），随圈从 0 爬到 1（圈1=0、圈2≈0.52、圈3≈0.74、圈4≈0.87、圈5≈0.94、圈6≈1.0）。理由：点位固有价值只在圈缩到足以「争抢」该点时才兑现——早期圈半径巨大、「圈内」形同虚设，站在一个离决赛圈很远的高 β_p 点不该全额记 β_p；只有圈缩到决赛圈附近，站哪才真正决定名次。这是对 $w_{\mathrm{zone}}$ 的**时间维**补充（前者管「圈内/外」，这个管「圈缩到哪一步」）。

**① $\varphi_m^{\mathrm{place}}(b)$ — 圈相对排名价值（非折现 Bellman 线性解）**

状态 $s=(m,b)$（第 $m$ 圈 + `rel_bin` 档 $b$），对每个状态统计：

$$
h(s)=\frac{D(s)}{V(s)},\qquad
pl(s)=\frac{PL(s)}{D(s)},\qquad
P(s'\,|\,s)=\frac{T(s,s')}{\sum_{s''}T(s,s'')}
$$

- $V(s)$ — 到访 $s$ 的采样行数；$D(s)$ — 其中被淘汰的行数；$PL(s)$ — 这些被淘汰行拿到的排名分之和；$T(s,s')$ — 从 $s$ 存活转移到 $s'$ 的行数。

> **采样行**：`data/long_table.jsonl` 里的一行 = 一支队伍在一个 5 秒时间步的状态快照（每 5 秒 × 每支活着的队伍写一行，全量 59 场共 220312 行）。所以 $V(s)$ = 队伍在「第 $m$ 圈 + 位置档 $b$」被观测到的 5 秒快照总数，$D(s)$ = 其中恰好这一步被淘汰的快照数。

非折现（$\gamma=1.0$）Bellman 方程写成一阶线性系统（$M[s][s']=(1-h(s))\,P(s'|s)$）：

$$
V(s)=(1-h(s))\sum_{s'}P(s'|s)\,V(s')+h(s)\,pl(s)
\;\Longleftrightarrow\;
(I-M)\,V = h\cdot pl
\;\Longrightarrow\;
\varphi_m^{\mathrm{place}}(b)=V[(m,b)]
$$

> 用线性解而非值迭代：$\gamma=1.0$ 时值迭代收敛慢；而只要 $h(s)>0$，$(I-M)$ 严格对角占优、必有唯一解。

**② $\beta_p^{\mathrm{place}}$ — 点位排名偏离（ridge 回归 + 队伍固定效应）**

$$
\min_{\beta,\delta}\ \Big\|\, y - X\beta - Z\delta \,\Big\|_2^2 + \lambda\,\|\beta\|_2^2,\qquad \lambda = 3
$$

$$
y_i = \underbrace{\mathrm{placement\_pts}_i}_{\text{队 }i\text{ 最终排名分}}
 - \underbrace{\frac{1}{T_i}\sum_{t} \varphi_{m(t)}^{\mathrm{place}}\big(\mathrm{rel\_bin}(t)\big)}_{\text{整场圈位期望排名（用 }\varphi\text{ 自身当基线）}}
$$

- $x_{i,p}$ — 队伍 $i$ 在第 $p$ 个点位的停留时间占比（`extract_stays` 的 stay 段；排除落地/搜刮的前 120s）。
- $Z\delta$ — 队伍固定效应（吸收队伍强弱，让 $\beta_p$ 只反映点位本身的偏离）。
- 目标用 $\varphi$ 自身当圈位基线，保证 $\beta_p$ 的零点跟 $\varphi$ 同口径。

**③ $\mathrm{rel\_bin}(r)$ — 圈相对位置分 6 档**（阈值 $\theta=[0.3,0.7,1.0,1.15,1.6]$）：

$$
\mathrm{rel\_bin}(r)=\min\big\{k : r < \theta_k\big\}
$$

对应「圈心 / 圈内 / 圈内边缘 / 圈外贴边 / 圈外 / 远圈外」。**毒里不硬记 0**：非折现口径下毒区档位本身有「贴着圈边、还有机会转进去」的小正数期望排名分。

**④ β 的拆解：掩体/地形 + 转点/连通（折中，纯拆解）**

**点位转点/连通价值本来就在 β 里、不在 φ 里**。φ 的 Bellman 递归只在「圈缩」这一条宏观通道上迭代（5 秒后圈到哪、相对位置档怎么变），它衡量的是「圈 + 相对位置」，**不含**「从这点能转到哪个具体点位」的点对点迭代。而 $β_p$ 是 ridge 直接对「最终排名分 − 圈位期望」回归出的点级残差：一个点如果转点/连通性好（四通八达、容易转到好点），踩它的队伍最终分会系统性偏高，这部分价值就被回归**隐式地折进了 $β_p$**。所以「转点」不需要在 φ 里做点对点迭代——它已经在 β 里了，只是没显式拆出来。

为了把这一层价值显式标出来，把 β_p 拆成两个子项（**纯拆解，合起来恒等，模型输出不变**）：

$$
\beta_p^{\mathrm{place}} = \beta_p^{\mathrm{terrain}} + \beta^{\mathrm{rotation}}\big(\mathrm{block}(p)\big)
$$

- $\beta_p^{\mathrm{terrain}}$ — **掩体/地形价值**（点级残差）：点位本身的防守价值（高地、掩体、视野）。
- $\beta^{\mathrm{rotation}}(b)$ — **转点/连通价值**（粗块级前瞻）：从这块能转到多好的地方。

转点项用**粗块转移 + 前瞻递归**估。先把 451 点按空间网格（边长 8000 世界单位）合并成 $\approx85$ 个粗块，再数块间转移（复用 `extract_stays` 的 stay 序列，已排除落地/搜刮前 120s，且**只取圈 3+**——开局圈 1/2 大家还在搜刮、随便转点，转移无连通性决策意义；圈 3+ 好点位被踩满、转点才难，此时「从这块能转到哪」才体现连通性价值）：

$$
P(b'\,|\,b)=\frac{T(b\to b')}{\sum_{b''}T(b\to b'')+\text{death/end}}
$$

$T(b\to b')$ = 圈 3+ 相邻 stay 段从块 $b$ 转到块 $b'$ 的次数（全量 59 场共约 1662 个圈 3+ stay 段）。因为死亡/结束不转移，$P$ 是**子随机**矩阵（每行和 $\le 1$）。然后解前瞻线性系统：

$$
R=\big(I-\gamma_{\mathrm{rot}}P\big)^{-1}\beta^{\mathrm{block}},
\qquad
\beta^{\mathrm{rotation}}(b)=R(b)-\beta^{\mathrm{block}}(b)
$$

其中 $\beta^{\mathrm{block}}(b)$ = 块内 $\beta_p$ 的覆盖加权均值（块的「即时」价值），$\gamma_{\mathrm{rot}}=0.8$（转点前瞻折扣）。$I-\gamma_{\mathrm{rot}}P$ 严格对角占优，必有唯一解。

> **这是纯拆解**：$\beta^{\mathrm{terrain}}+\beta^{\mathrm{rotation}}=\beta^{\mathrm{place}}$ 恒成立（校验误差 $2.2\times10^{-16}$），模型预测**完全不变**，只是把「转点价值」从 β 里显式标出来。实测 $\beta^{\mathrm{rotation}}\in[-0.88,+1.07]$、$\beta^{\mathrm{terrain}}\in[-1.80,+2.37]$，二者相关 $-0.09$（基本正交——两个子项各管一维：地形价值和连通性价值不再互相挤占）。

> **实测值域**：$\beta_p^{\mathrm{place}}\in[-1.53,+3.05]$（= $\beta^{\mathrm{terrain}}\in[-1.80,+2.37]$ + $\beta^{\mathrm{rotation}}\in[-0.88,+1.07]$），$\varphi_m^{\mathrm{place}}\in[1.49,8.10]$；$\mathrm{kill\_slope}$ 与 $\beta_p^{\mathrm{place}}$ 的相关系数 $+0.125$（两通道分离度高）。

> **为什么 φ 现在还留着（数据瓶颈 → 长期方向）**：φ 的「圈 + 相对位置」本质上是个**暂时、粗略的代理**——真正决定点位价值的应是「点位之间的连通性」（从这点能转到哪些点、那些点又值多少）。要做**点对点转点迭代**（在 $451\times451$ 的转移矩阵 $P(p'|p)$ 上跑 Bellman），需要每个点对都有足够转移样本；而 59 场只有约 2638 个 stay 段，点对点转移极稀疏（绝大多数点对从未观测到转移），硬算只会是噪声。所以当前折成两层：**β_rotation 先在 85 个粗块上做连通性前瞻递归**（把样本聚合到粗块、转移才勉强稳），φ 继续兜住「圈缩」这条跨场可统计的宏观通道。**长期方向**：数据量上去（几十上百场、点对点转移密度达标）后，把粗块递归细化到**点位级连通性迭代**

$$
R(p)=\beta^{\mathrm{terrain}}(p)+\gamma\sum_{p'}P(p'\,|\,p)\,R(p'),
$$

到那时 φ 的「圈缩」宏观项即可**逐步取消**、被真正的点位连通性迭代吸收。

### IGL 评分(初步，未完成)

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
python3 scripts/model/points_split.py        # → 击杀斜率(点位×圈, 击杀/秒) + 排名(β_p_place + φ_place, 非折现)

# 3) IGL 评分（亚太南 21 场示例）
python3 scripts/igl/score_igl2.py            # 站位残差榜 → output/igl_apac_s_residual.*
python3 scripts/igl/igl_two_layer.py         # 站位 × 兑现 二维榜

# 4) 可视化（任意一场比赛）
python3 scripts/viz/apply_game_xT.py "replay/storm point/sp_apac-s_d__g6_abfd4cf4.json"   # xT 热点图
python3 scripts/viz/gen_trajectory_demo.py "replay/storm point/sp_apac-s_d__g6_abfd4cf4.json"  # 队伍 xT 轨迹网页
python3 scripts/viz/coarse_blocks.py         # 粗块合并预览（单独看，不进模型）
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
- **毒区取值**：排名通道的 φ_place 是「期望最终排名分」，非折现口径下毒区档位本身就有小正数（贴着圈边、还有机会转进去），因此不做「短暂进出/持续在毒」的硬清零——毒里的 xT 自然地偏低但非 0。圈外判定用**新圈（安全区）基准**（`finishedClosing` 中心 + `endRadius`），β_p 圈外按 $w_{\mathrm{zone}}$ 平滑衰减、再按圈阶段 $w_{\mathrm{stage}}$ 递增（早期圈压低点位固有价值）。
- IGL 评分是 xT 模型的**只读下游**：只消费模型产物（`data/phi.json`、`*.npy`），不重新拟合、不改写模型。
