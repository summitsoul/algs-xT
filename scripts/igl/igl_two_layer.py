#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
igl_two_layer.py — IGL 能力拆两层, 解决"错位"

单层 IGL 分(站位残差)只测"你站到多好的位置", 不测"你把这些位置兑现成多少分"。
错位:
  VK Gaming  分第2, 站位负  -> 贴边打法, 位置差但能打(兑现强)
  NAH/Ctrl Alt 分低, 站位正 -> 位置好但打不出(兑现弱)

两层 (同一套 xT 模型, 互不冲突):
  L1 站位 (position)  = 每圈 (你队锚点 xT − 同时刻全场平均 xT) 圈间加总, 跨场平均
                        -> "你比别人站得好多少"  (score_igl2 已算)
  L2 兑现 (conversion)= 点位兑现残差 = 队伍固定效应
                        实际分 − 圈相对期望 − 点位停留价值(β_p) 之后, 队伍仍系统性地
                        多拿/少拿多少分 -> "同一粗位置里谁长期拿得比该位置均值多"
                        (用 xT_points 里 ridge 的 team fixed effect, 但只在 apac-s 估)

错位归位:
  L1 负 + L2 正 = VK   (贴边, 位置差但能打)
  L1 正 + L2 负 = NAH  (位置好但打不出)
  L1 正 + L2 正 = 真冠军
"""
import json, math
from collections import defaultdict
import numpy as np

LONG = "data/long_table.jsonl"
PHI = "data/phi.json"
OUT_CSV = "output/igl_two_layer.csv"
OUT_PNG = "output/igl_two_layer.png"
MIN_GAMES = 3
LAM = 3.0

# ---------- 模型 ----------
phi = json.load(open(PHI, encoding='utf-8'))
pk = {(int(a), int(b)): v for a, b, v in
      (k.split(',') + [v] for k, v in phi['phi_kill'].items())}
pp = {(int(a), int(b)): v for a, b, v in
      (k.split(',') + [v] for k, v in phi['phi_place'].items())}
ef_k = {(int(a), int(b)): v for a, b, v in
        (k.split(',') + [v] for k, v in phi['E_final_kill'].items())}
ef_p = {(int(a), int(b)): v for a, b, v in
        (k.split(',') + [v] for k, v in phi['E_final_place'].items())}
E_final = {s: ef_k[s] + ef_p[s] for s in ef_k}

bk = np.load("data/beta_kill.npy")
bp = np.load("data/beta_place.npy")
pos_arr = np.load("data/points_pos.npy")
NP = len(bk)

phi_kill = np.zeros((6, 6)); phi_place = np.zeros((6, 6))
for (p, b), v in pk.items():
    phi_kill[p, b] = v
for (p, b), v in pp.items():
    phi_place[p, b] = v


def nearest_idx(xs, ys):
    try:
        from scipy.spatial import cKDTree
        return cKDTree(pos_arr).query(np.c_[xs, ys])[1]
    except Exception:
        X = np.c_[xs, ys]
        out = np.empty(len(X), int)
        for i in range(0, len(X), 20000):
            d = np.hypot(X[i:i + 20000, None, 0] - pos_arr[:, 0],
                         X[i:i + 20000, None, 1] - pos_arr[:, 1])
            out[i:i + 20000] = d.argmin(1)
        return out


# ---------- 读 apac-s ----------
rows = []
for line in open(LONG, encoding='utf-8'):
    r = json.loads(line)
    if r['region'] == 'apac-s':
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

# 毒里软处理: 短暂进出毒(连续毒内 ≤BRIEF_SEC) 或 圈1/圈2(phase<2) 不强制清零,
# 按"圈内边缘"(rel_bin=SOFT_BIN) 计值; 圈3+ 且持续在毒里才清零。
BRIEF_SEC = 15
SOFT_BIN = 2


def poison_brief_mask(game, team, t, rel, brief_sec=BRIEF_SEC):
    """每行是否处于'短暂进出毒'(连续毒内停留 ≤brief_sec)。按 (game,team,t) 排序。"""
    n = len(rel)
    order = sorted(range(n), key=lambda i: (game[i], team[i], t[i]))
    brief = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        gi, ti = game[order[i]], team[order[i]]
        j = i
        while j < n and game[order[j]] == gi and team[order[j]] == ti:
            j += 1
        k = i
        while k < j:
            if rel[order[k]] > 1.0:
                m = k
                while m < j and rel[order[m]] > 1.0:
                    m += 1
                if (m - k) * 5 <= brief_sec:      # 采样间隔 5s
                    brief[order[k:m]] = True
                k = m
            else:
                k += 1
        i = j
    return brief


j = nearest_idx(xs, ys)
inring = rel <= 1.0
brief = poison_brief_mask(game, team, t, rel)
soft = (~inring) & (brief | (phase < 2))        # 短暂进出 或 圈1/圈2
xk = np.where(inring, bk[j] + phi_kill[phase, relbin], 0.0)
xp = np.where(inring, bp[j] + phi_place[phase, relbin], 0.0)
xk = np.where(soft, bk[j] + phi_kill[phase, SOFT_BIN], xk)
xp = np.where(soft, bp[j] + phi_place[phase, SOFT_BIN], xp)
xt = xk + xp
print(f"xT 分布: total 均值 {xt.mean():.3f}, 毒内软处理 {int(soft.sum())}/{int((~inring).sum())}")

# ---------- L1: 同时刻 (game,t) 全场平均 → 站位残差 ----------
bucket_idx = defaultdict(list)
for i, (g, tt) in enumerate(zip(game, t)):
    bucket_idx[(g, tt)].append(i)
fa_t = {key: float(xt[idxs].mean()) for key, idxs in bucket_idx.items()}
res_t = xt - np.array([fa_t[(game[i], t[i])] for i in range(len(rows))])

phase_res = defaultdict(lambda: defaultdict(list))
for i in range(len(rows)):
    phase_res[(team[i], game[i])][int(phase[i])].append(res_t[i])

# ---------- L2: 点位兑现残差 = 队伍固定效应 (ridge, 同 xT_points 口径) ----------
# 按 (game, team_id) 分组, 算 final / ring_exp / occ
tg = defaultdict(list)
for r in rows:
    tg[(r['game'], r['team'])].append(r)


def extract_stays(rws, R=150, T=6, TMIN=120):
    stays = []
    i, n = 0, len(rws)
    while i < n:
        j, pts = i, []
        while j < n:
            pts.append((rws[j]['ax'], rws[j]['ay']))
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            if all(math.hypot(p[0] - cx, p[1] - cy) <= R for p in pts):
                j += 1
            else:
                pts.pop()
                break
        if len(pts) >= T and rws[i]['t'] >= TMIN:
            stays.append((sum(p[0] for p in pts) / len(pts),
                          sum(p[1] for p in pts) / len(pts), len(pts)))
        i = max(i + 1, j)
    return stays


units = []
for (gid, tid), rws in tg.items():
    rws = sorted(rws, key=lambda r: r['t'])
    final = sum(r['kills'] + r['placement_pts'] for r in rws)
    ring_exp = sum(E_final[(r['phase'], r['rel_bin'])] for r in rws) / len(rws)
    stays = extract_stays(rws)
    occ = defaultdict(float)
    tot = sum(s[2] for s in stays)
    if tot > 0:
        for (cx, cy, nb) in stays:
            d = np.hypot(pos_arr[:, 0] - cx, pos_arr[:, 1] - cy)
            occ[int(np.argmin(d))] += nb / tot
    units.append({'name': rws[0]['team_name'].lower(), 'y': final - ring_exp,
                  'occ': dict(occ), 'final': final})

team_names = sorted({u['name'] for u in units})
team_idx = {t: i for i, t in enumerate(team_names)}
N, Tdim = len(units), len(team_names)
X = np.zeros((N, NP + Tdim - 1))
y = np.zeros(N)
for i, u in enumerate(units):
    y[i] = u['y']
    for p, f in u['occ'].items():
        X[i, p] = f
    ti = team_idx[u['name']]
    if ti > 0:
        X[i, NP + ti - 1] = 1.0

A = X.T @ X
b = X.T @ y
Areg = A.copy()
Areg[:NP, :NP] += LAM * np.eye(NP)
beta = np.linalg.solve(Areg, b)
fe = np.zeros(Tdim)
fe[1:] = beta[NP:]
fe[0] = 0.0
fe -= fe.mean()  # 中心化: 队伍固定效应(兑现残差), 0 = 平均兑现

# ---------- 汇总 ----------
rec = defaultdict(lambda: {'L1': [], 'games': 0, 'pts': 0.0})
for (nm, g), ph in phase_res.items():
    l1 = sum(float(np.mean(ph[m])) for m in sorted(ph))
    rec[nm]['L1'].append(l1)
    rec[nm]['games'] += 1

for u in units:
    rec[u['name']]['pts'] += u['final']

out = []
for nm, d in rec.items():
    out.append({
        'team': nm,
        'games': d['games'],
        'L1': float(np.mean(d['L1'])),
        'L2': float(fe[team_idx.get(nm, 0)]),
        'pts': d['pts'],
    })
out.sort(key=lambda x: -x['pts'])

with open(OUT_CSV, 'w', encoding='utf-8') as f:
    f.write("team,games,L1_position,L2_conversion,total_pts\n")
    for r in out:
        f.write(f"{r['team']},{r['games']},{r['L1']:.3f},{r['L2']:.3f},"
                f"{r['pts']:.0f}\n")

print(f"\n亚太南 {len(out)} 支 | L1=站位(相对全场) L2=兑现(队伍固定效应, 0=平均)")
print(f"{'队伍':<20}{'场':>3}{'总分':>6}{'L1站位':>8}{'L2兑现':>8}")
for r in out:
    star = " *" if r['games'] < MIN_GAMES else ""
    print(f"{r['team']:<20}{r['games']:>3}{r['pts']:>6.0f}{r['L1']:>8.2f}"
          f"{r['L2']:>8.2f}{star}")

# ---------- 2D 散点 ----------
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

main = [r for r in out if r['games'] >= MIN_GAMES]
x = [r['L1'] for r in main]
y = [r['L2'] for r in main]
s = [max(r['pts'], 1) * 6 for r in main]
c = [r['pts'] for r in main]

fig, ax = plt.subplots(figsize=(12, 8))
sc = ax.scatter(x, y, s=s, c=c, cmap='YlOrRd', alpha=0.85, edgecolors='#333',
                linewidths=0.6)
for r in main:
    ax.annotate(r['team'], (r['L1'], r['L2']), xytext=(4, 4),
                textcoords='offset points', fontsize=8)
ax.axhline(0, color='#666', lw=1)
ax.axvline(0, color='#666', lw=1)
ax.set_xlabel('L1 站位 (每圈 你队xT − 全场平均, 跨场平均)  → 位置选得多好')
ax.set_ylabel('L2 兑现 (点位兑现残差 / 队伍固定效应, 0=平均)  → 位置打出来多少')
ax.set_title('亚太南 21 场 — IGL 双层能力: 站位 × 兑现 (点大小/颜色 = 总积分)', fontsize=13)
ax.text(0.02, 0.97, '好位置·打不出\n(NAH/Ctrl Alt)', transform=ax.transAxes,
        ha='left', va='top', fontsize=9, color='#666')
ax.text(0.98, 0.97, '好位置·能兑现\n(真冠军)', transform=ax.transAxes,
        ha='right', va='top', fontsize=9, color='#666')
ax.text(0.02, 0.03, '差位置·打不出', transform=ax.transAxes,
        ha='left', va='bottom', fontsize=9, color='#666')
ax.text(0.98, 0.03, '差位置·能兑现\n(VK 贴边)', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=9, color='#666')
cb = plt.colorbar(sc, ax=ax)
cb.set_label('21 场总积分')
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=130)
print(f"\n已写 {OUT_CSV} | {OUT_PNG}")
