#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xT_points.py — 点位 xT (足球式分块版)

xT(点位p, 第m圈z) = β_p + φ_m(rel_bin)   当 p 在圈内 (rel<=1)
                    FLOOR(≈0)             当 p 在毒里 (rel>1)

  β_p   = 点位固有价值 (掩体/地形/可防守), 跨场 ridge + 队伍固定效应解
  φ_m   = 圈相对价值 (值迭代 Bellman, 状态 (m, rel_bin))
  rel   = 点位到圈心距离 / 圈半径; rel<=1 圈内, >1 毒

输出:
  data/points_value.csv   点位 β_p
  output/points_beta.png  β_p 基值热力图 (圈无关)
  output/points_xT.png    示例圈下的完整 xT (圈内 β_p+φ, 毒区清零)
"""
import json, math
from collections import defaultdict
import numpy as np

GAMMA = 0.99
REL_BINS = [0.3, 0.7, 1.0, 1.15, 1.6]
REL_NAMES = ['圈心', '圈内', '圈内边缘', '圈外贴边', '圈外', '远圈外']
LONG = "data/long_table.jsonl"
POS = "data/positions.csv"
OUT_CSV = "data/points_value.csv"
RATIO = 24.93
MAP_PNG = "map/storm point.png"
SIZE = 4096
LAM = 3.0
MIN_COV = 2        # 点位至少被几支 team-game 踩过才显示


def rel_bin(rel):
    for i, th in enumerate(REL_BINS):
        if rel < th:
            return i
    return len(REL_BINS)


rows = [json.loads(l) for l in open(LONG, encoding='utf-8')]

# 点位 (cx, cy)
pos = []
for line in open(POS, encoding='utf-8'):
    if line.startswith('id'):
        continue
    p = line.strip().split(',')
    pos.append((float(p[1]), float(p[2])))
pos_arr = np.array(pos)
NP = len(pos)

# ---- φ_m(r): 值迭代 (状态 (phase, rel_bin)) ----
visits = defaultdict(int)
k_sum = defaultdict(float)
deaths = defaultdict(int)
pl_sum = defaultdict(float)
trans = defaultdict(lambda: defaultdict(int))
for r in rows:
    s = (r['phase'], r['rel_bin'])
    visits[s] += 1
    k_sum[s] += r['kills']
    if r['died']:
        deaths[s] += 1
        pl_sum[s] += r['placement_pts']
    else:
        trans[s][(r['next_phase'], r['next_rel_bin'])] += 1

states = sorted(visits)
k = {s: k_sum[s] / visits[s] for s in states}
h = {s: deaths[s] / visits[s] for s in states}
pl = {s: (pl_sum[s] / deaths[s] if deaths[s] else 0.0) for s in states}
P = {}
for s in states:
    tot = sum(trans[s].values())
    P[s] = {s2: trans[s][s2] / tot for s2 in trans[s]} if tot else {}

V = {s: 0.0 for s in states}
for _ in range(5000):
    Vn, delta = {}, 0.0
    for s in states:
        fut = sum(P[s].get(s2, 0) * V[s2] for s2 in P[s]) if P[s] else 0.0
        Vn[s] = k[s] + GAMMA * ((1 - h[s]) * fut + h[s] * pl[s])
        delta = max(delta, abs(Vn[s] - V[s]))
    V = Vn
    if delta < 1e-6:
        break
phi = {s: V[s] for s in states}

# E_final (不折现期望终分, 用于 β_p 回归的圈位置控制)
tg = defaultdict(list)
final = defaultdict(float)
for r in rows:
    tg[(r['game'], r['team'])].append(r)
    final[(r['game'], r['team'])] += r['kills'] + r['placement_pts']
ef_sum = defaultdict(float)
for r in rows:
    ef_sum[(r['phase'], r['rel_bin'])] += final[(r['game'], r['team'])]
E_final = {s: ef_sum[s] / visits[s] for s in states}


# ---- 停留段 (落地后 t>=120s, R=150, T=6) ----
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


# ---- 每 team-game: ring_exp + 点位停留占比 ----
units = []
for (gid, tid), rws in tg.items():
    rws = sorted(rws, key=lambda r: r['t'])
    ring_exp = sum(E_final[(r['phase'], r['rel_bin'])] for r in rws) / len(rws)
    stays = extract_stays(rws)
    occ = defaultdict(float)
    tot = sum(s[2] for s in stays)
    if tot > 0:
        for (cx, cy, nb) in stays:
            d = np.hypot(pos_arr[:, 0] - cx, pos_arr[:, 1] - cy)
            occ[int(np.argmin(d))] += nb / tot
    units.append({'name': rws[0]['team_name'], 'ring_exp': ring_exp,
                  'final': final[(gid, tid)], 'occ': dict(occ)})

# ---- ridge: 残差 ~ 点位停留占比 + 队伍固定效应 ----
team_names = sorted({u['name'] for u in units})
team_idx = {t: i for i, t in enumerate(team_names)}
N, Tdim = len(units), len(team_names)
X = np.zeros((N, NP + Tdim - 1))
y = np.zeros(N)
for i, u in enumerate(units):
    y[i] = u['final'] - u['ring_exp']
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
beta_p = beta[:NP]

# 覆盖: 每个点位被多少 team-game 踩过
cov = np.zeros(NP, int)
for u in units:
    for p in u['occ']:
        cov[p] += 1

# 保存
with open(OUT_CSV, 'w', encoding='utf-8') as f:
    f.write("id,cx,cy,beta_p,coverage\n")
    for i in range(NP):
        f.write(f"{i},{pos[i][0]:.1f},{pos[i][1]:.1f},{beta_p[i]:.4f},{cov[i]}\n")

print(f"点位 {NP} 个 | team-game {N} | teams {Tdim} | λ={LAM}")
print(f"β_p 范围: {beta_p.min():+.2f} ~ {beta_p.max():+.2f} (负=废点, 正=好点)")
print(f"覆盖≥{MIN_COV} 的点位: {(cov >= MIN_COV).sum()}/{NP}")
top = np.argsort(-beta_p)[:8]
bot = np.argsort(beta_p)[:8]
print("β_p 最高 (好点位):")
for i in top:
    print(f"  ({pos[i][0]:8.0f},{pos[i][1]:8.0f}) β={beta_p[i]:+.2f} 覆盖{cov[i]}")
print("β_p 最低 (废点位):")
for i in bot:
    print(f"  ({pos[i][0]:8.0f},{pos[i][1]:8.0f}) β={beta_p[i]:+.2f} 覆盖{cov[i]}")
print(f"已写 {OUT_CSV}")

np.save("data/points_beta.npy", beta_p)
np.save("data/points_pos.npy", pos_arr)

# 保存 φ / E_final (圈相对价值表), 供 apply 阶段复用
json.dump({'phi': {f"{a},{b}": v for (a, b), v in phi.items()},
           'E_final': {f"{a},{b}": v for (a, b), v in E_final.items()}},
          open("data/phi.json", "w", encoding='utf-8'))
print("已写 data/phi.json")
