#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_igl2.py — IGL 评分 (亚太南 21 场), 残差版 (用当前 xT 模型)

口径 (与用户定稿):
  xT(p, 圈) = β_p + φ_m(rel_bin),  rel<=1 (圈内); rel>1 (毒里) = 0
  IGL 分 = 跨场平均 [ Σ_{每圈} ( 你队锚点 xT − 全场同时刻平均 xT ) ]

  不剥落地、不除运气 (t 从 0 起), 只跑 region=='apac-s' 的 21 场。

与旧 score_igl.py 的区别:
  - 旧版只加 φ (同心环, 同圈同档所有队一个值, 全挤在 1.2~1.4);
  - 这版用 β_p(点位固有, 按最近点) + φ, 且做"同时刻减全场平均"的残差。
"""
import json
from collections import defaultdict
import numpy as np

LONG = "data/long_table.jsonl"
PHI = "data/phi.json"
OUT_CSV = "output/igl_apac_s_residual.csv"
OUT_PNG = "output/igl_apac_s_residual.png"
MIN_GAMES = 3

# ---------- 模型 ----------
phi = json.load(open(PHI, encoding='utf-8'))
pk = {(int(a), int(b)): v for a, b, v in
      (k.split(',') + [v] for k, v in phi['phi_kill'].items())}
pp = {(int(a), int(b)): v for a, b, v in
      (k.split(',') + [v] for k, v in phi['phi_place'].items())}
bk = np.load("data/beta_kill.npy")
bp = np.load("data/beta_place.npy")
pos_arr = np.load("data/points_pos.npy")

phi_kill = np.zeros((6, 6)); phi_place = np.zeros((6, 6))
for (p, b), v in pk.items():
    phi_kill[p, b] = v
for (p, b), v in pp.items():
    phi_place[p, b] = v


def nearest_idx(xs, ys):
    """给 (xs, ys) 找最近点位索引 (向量化)."""
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pos_arr)
        return tree.query(np.c_[xs, ys])[1]
    except Exception:
        X = np.c_[xs, ys]
        out = np.empty(len(X), int)
        for i in range(0, len(X), 20000):
            d = np.hypot(X[i:i + 20000, None, 0] - pos_arr[:, 0],
                         X[i:i + 20000, None, 1] - pos_arr[:, 1])
            out[i:i + 20000] = d.argmin(1)
        return out


# ---------- 读 long table (apac-s) ----------
rows = []
for line in open(LONG, encoding='utf-8'):
    r = json.loads(line)
    if r['region'] != 'apac-s':
        continue
    rows.append(r)
print(f"apac-s 行数 {len(rows)}")

xs = np.array([r['ax'] for r in rows])
ys = np.array([r['ay'] for r in rows])
phase = np.array([r['phase'] for r in rows])
relbin = np.array([r['rel_bin'] for r in rows])
rel = np.array([r['rel'] for r in rows])
t = np.array([r['t'] for r in rows])
game = [r['game'] for r in rows]
team = [r['team_name'].lower() for r in rows]

j = nearest_idx(xs, ys)
inring = rel <= 1.0
xk = np.where(inring, bk[j] + phi_kill[phase, relbin], 0.0)
xp = np.where(inring, bp[j] + phi_place[phase, relbin], 0.0)
xt = xk + xp
print(f"xT 分布: total 均值 {xt.mean():.3f}, 圈内占比 {inring.mean():.3f}")

# ---------- 同时刻 (game,t) 全场平均 ----------
bucket_idx = defaultdict(list)
for i, (g, tt) in enumerate(zip(game, t)):
    bucket_idx[(g, tt)].append(i)

fa_k = {}; fa_p = {}; fa_t = {}
for key, idxs in bucket_idx.items():
    fa_k[key] = float(xk[idxs].mean())
    fa_p[key] = float(xp[idxs].mean())
    fa_t[key] = float(xt[idxs].mean())

res_k = xk - np.array([fa_k[(game[i], t[i])] for i in range(len(rows))])
res_p = xp - np.array([fa_p[(game[i], t[i])] for i in range(len(rows))])
res_t = xt - np.array([fa_t[(game[i], t[i])] for i in range(len(rows))])

# ---------- 聚合: 每队每场, 每圈平均残差 → 圈间求和/平均 ----------
# phase_bucket[(team, game)][phase] = list of (res_k, res_p, res_t)
phase_bucket = defaultdict(lambda: defaultdict(list))
for i in range(len(rows)):
    phase_bucket[(team[i], game[i])][int(phase[i])].append(
        (res_k[i], res_p[i], res_t[i]))

team_games = defaultdict(list)  # team -> list of (game, k_sum, p_sum, t_sum, k_mean, p_mean, t_mean)
for (nm, g), ph in phase_bucket.items():
    if not ph:
        continue
    sums = np.zeros(3); cnt = 0
    for m in sorted(ph):
        arr = np.array(ph[m])
        sums += arr.mean(0)          # 每圈平均残差
        cnt += 1
    k_sum, p_sum, t_sum = sums            # 圈间求和 = "Σ每圈"
    k_mean, p_mean, t_mean = sums / cnt   # 圈间平均 = 控"活得久"
    team_games[nm].append((g, k_sum, p_sum, t_sum, k_mean, p_mean, t_mean))

# ---------- 最终分 (跨场平均) ----------
rec = []
for nm, gs in team_games.items():
    n = len(gs)
    rec.append({
        'team': nm,
        'games': n,
        'k_sum': float(np.mean([g[1] for g in gs])),
        'p_sum': float(np.mean([g[2] for g in gs])),
        't_sum': float(np.mean([g[3] for g in gs])),
        'k_mean': float(np.mean([g[4] for g in gs])),
        'p_mean': float(np.mean([g[5] for g in gs])),
        't_mean': float(np.mean([g[6] for g in gs])),
    })
rec.sort(key=lambda x: -x['t_sum'])

# ---------- 输出 CSV ----------
with open(OUT_CSV, 'w', encoding='utf-8') as f:
    f.write("rank,team,games,kill_sum,place_sum,total_sum,kill_mean,place_mean,total_mean\n")
    for i, r in enumerate(rec):
        f.write(f"{i+1},{r['team']},{r['games']},"
                f"{r['k_sum']:.3f},{r['p_sum']:.3f},{r['t_sum']:.3f},"
                f"{r['k_mean']:.3f},{r['p_mean']:.3f},{r['t_mean']:.3f}\n")

print(f"\n亚太南队伍 {len(rec)} 支 (IGL 分 = 每圈站位残差, 跨场平均):")
print(f"{'#':>2} {'队伍':<20} {'场':>2} {'击杀':>8} {'排名':>8} {'合计':>8}  {'每圈均':>8}")
for i, r in enumerate(rec):
    flag = "  *" if r['games'] < MIN_GAMES else ""
    print(f"{i+1:>2} {r['team']:<20} {r['games']:>2} {r['k_sum']:>8.3f} "
          f"{r['p_sum']:>8.3f} {r['t_sum']:>8.3f} {r['t_mean']:>8.3f}{flag}")
print("  * = 场次 < 3, 样本少")

# ---------- PNG: 堆叠横条 (排名残差 + 击杀残差) ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for _fp in ['/mnt/c/Windows/Fonts/simhei.ttf',
            '/mnt/c/Windows/Fonts/msyh.ttc',
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf']:
    try:
        font_manager.fontManager.addfont(_fp)
    except Exception:
        pass
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                   'Droid Sans Fallback', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

top = [r for r in rec if r['games'] >= MIN_GAMES]
names = [r['team'] for r in top]
place = [r['p_sum'] for r in top]
kill = [r['k_sum'] for r in top]
totals = [r['t_sum'] for r in top]

fig, ax = plt.subplots(figsize=(8, 0.30 * len(top) + 2.4))
y = np.arange(len(top))
ax.barh(y, place, color='#4c78a8', edgecolor='#222', height=0.7, label='排名 xT 残差')
ax.barh(y, kill, left=place, color='#e45756', edgecolor='#222', height=0.7,
        label='击杀 xT 残差')
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9); ax.invert_yaxis()
ax.set_xlabel('IGL 分 = 每圈 (你队 xT − 全场平均) 加总, 跨场平均', fontsize=10)
ax.set_title('亚太南 21 场 — IGL 指挥能力榜 (站位残差)', fontsize=12)
ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax.axvline(0, color='#666', lw=1)
for yi, tt in zip(y, totals):
    ax.text(tt + 0.02, yi, f'{tt:.2f}', va='center', ha='left', fontsize=8)
lim = max(abs(min(totals + [0])), abs(max(totals + [0]))) * 1.2
ax.set_xlim(-lim, lim)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=130)
print(f"\n已写 {OUT_CSV} | {OUT_PNG} (主榜 {len(top)} 支, 场次>={MIN_GAMES})")
