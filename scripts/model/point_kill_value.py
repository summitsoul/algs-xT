#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
point_kill_value.py — 点位击杀价值 β_kill(位置)

按用户纠正: 击杀期望分应按"踩到这个位置的队伍, 在该位置时获得的击杀分"算,
而不是按 (圈, 离圈心距离) 这种圈相对状态取全场平均。

结论先行(见运行输出): 451 个点位太细, 击杀是稀有事件(每桶~0.016), 点级击杀率
几乎全是噪声, 相关性反而更差。要落到"具体位置", 得用足球 xT 那套——把地图切成
粗网格(cell), 每个 cell 聚合击杀/时长, 再用收缩(拉向全局均值)降噪。

β_kill(cell) = (击杀分 + α·全局击杀率) / (被踩桶数 + α)   # α=伪观测收缩强度
"""
import json
from collections import defaultdict
import numpy as np

LONG = "data/long_table.jsonl"
POS = "data/positions.csv"
PHI_JSON = "data/phi.json"
TMIN = 120
MIN_GAMES = 3
ALPHA = 30          # 收缩伪观测桶数
GRIDS = [8, 12, 16, 20]

pos = []
for line in open(POS, encoding='utf-8'):
    if line.startswith('id'):
        continue
    p = line.strip().split(',')
    pos.append((float(p[1]), float(p[2])))
pos_arr = np.array(pos)
NP = len(pos)

rows = [json.loads(l) for l in open(LONG, encoding='utf-8')]
apac = [r for r in rows if r['region'] == 'apac-s' and r['t'] >= TMIN]

j = json.load(open(PHI_JSON, encoding='utf-8'))
pk_old = {(int(k.split(',')[0]), int(k.split(',')[1])): v
          for k, v in j['phi_kill'].items()}

xmin, xmax = pos_arr[:, 0].min(), pos_arr[:, 0].max()
ymin, ymax = pos_arr[:, 1].min(), pos_arr[:, 1].max()

# ---- 全局击杀率 (收缩先验) ----
tot_k = sum(r['kills'] for r in apac)
tot_v = sum(1 for r in apac if r['rel'] <= 1.0)
global_rate = tot_k / max(tot_v, 1)


def team_ak():
    """每队实际击杀/场"""
    a = defaultdict(lambda: {'k': 0.0, 'g': set()})
    for r in apac:
        nm = r['team_name'].lower()
        a[nm]['k'] += r['kills']
        a[nm]['g'].add(r['game'])
    return {nm: a[nm]['k'] / len(a[nm]['g']) for nm in a
            if len(a[nm]['g']) >= MIN_GAMES}


def corr(scores):
    ak = team_ak()
    names = sorted(ak)
    xs = [ak[n] for n in names]
    ys = [scores.get(n, 0.0) for n in names]
    return np.corrcoef(xs, ys)[0, 1]


# ---- 1) 旧口径: 圈相对 φ_kill ----
s_old = defaultdict(lambda: {'s': 0.0, 'w': 0})
for r in apac:
    s_old[r['team_name'].lower()]['w'] += 1
    if r['rel'] <= 1.0:
        s_old[r['team_name'].lower()]['s'] += pk_old.get((r['phase'], r['rel_bin']), 0.0)
scores_old = {nm: a['s'] / a['w'] for nm, a in s_old.items()}
print(f"旧口径 (圈相对 φ_kill):            r = {corr(scores_old):+.3f}")


# ---- 2) 451 点位 (无收缩) ----
kills = np.zeros(NP); visits = np.zeros(NP)
for r in apac:
    if r['rel'] > 1.0:
        continue
    d = np.hypot(pos_arr[:, 0] - r['ax'], pos_arr[:, 1] - r['ay'])
    jpt = int(np.argmin(d))
    kills[jpt] += r['kills']; visits[jpt] += 1
beta_pts = np.where(visits > 0, kills / np.where(visits == 0, 1, visits), 0.0)
s_pts = defaultdict(lambda: {'s': 0.0, 'w': 0})
for r in apac:
    s_pts[r['team_name'].lower()]['w'] += 1
    if r['rel'] <= 1.0:
        d = np.hypot(pos_arr[:, 0] - r['ax'], pos_arr[:, 1] - r['ay'])
        s_pts[r['team_name'].lower()]['s'] += beta_pts[int(np.argmin(d))]
scores_pts = {nm: a['s'] / a['w'] for nm, a in s_pts.items()}
print(f"451 点位 (无收缩):                r = {corr(scores_pts):+.3f}")


# ---- 3) 粗网格 + 收缩 ----
def grid_beta(GRID):
    nk = np.zeros((GRID, GRID)); nv = np.zeros((GRID, GRID))
    for r in apac:
        if r['rel'] > 1.0:
            continue
        ci = int(min(GRID - 1, max(0, (r['ax'] - xmin) / (xmax - xmin) * GRID)))
        cj = int(min(GRID - 1, max(0, (r['ay'] - ymin) / (ymax - ymin) * GRID)))
        nk[ci, cj] += r['kills']; nv[ci, cj] += 1
    beta = (nk + ALPHA * global_rate) / (nv + ALPHA)
    return beta


for G in GRIDS:
    beta_g = grid_beta(G)
    s_g = defaultdict(lambda: {'s': 0.0, 'w': 0})
    for r in apac:
        s_g[r['team_name'].lower()]['w'] += 1
        if r['rel'] <= 1.0:
            ci = int(min(G - 1, max(0, (r['ax'] - xmin) / (xmax - xmin) * G)))
            cj = int(min(G - 1, max(0, (r['ay'] - ymin) / (ymax - ymin) * G)))
            s_g[r['team_name'].lower()]['s'] += beta_g[ci, cj]
    scores_g = {nm: a['s'] / a['w'] for nm, a in s_g.items()}
    print(f"{G}x{G} 网格 + 收缩(α={ALPHA}):          r = {corr(scores_g):+.3f}")