#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regional_kill_compare.py — 击杀价值是"位置属性"还是"队伍属性"? 分区域检验

用户假设: 击杀分应是"点位"来的(踩到哪个点→该点击杀期望), 而不是圈相对平均;
低强度区(东南亚)信号弱是因为队伍参差, 高强度区(世界赛/北美)点位信号应更强。

口径(全部 t>=120s, 毒里 rel>1 清零, 不剔弱队):
  圈相对 k(s)   = 全场在该 (phase, rel_bin) 的击杀/桶   -> 队伍取轨迹均值
  451点位 β(p)  = 该点击杀/桶 (最近点归属)             -> 队伍取轨迹均值, 留一
  NxN 网格 β(c) = 该格击杀/桶, α 收缩                 -> 队伍取轨迹均值, 留一
每个口径与"该队实际击杀/场"求 Pearson r。点位/网格用留一(剔除本队)去自证循环。
"""
import json, csv
from collections import defaultdict
import numpy as np

LONG = "data/long_table.jsonl"
POS = "data/positions.csv"
TMIN = 120
MIN_GAMES = 3
REGIONS = ["global", "na", "apac-s"]
GRID = 16
ALPHA = 30

# 固定网格包围盒(全地图, 保证跨区域可比)
XB = (-46141.0, 42361.0)
YB = (-41306.0, 48120.0)

# 451 点位
pts = []
for row in csv.DictReader(open(POS, encoding='utf-8')):
    pts.append((float(row['cx']), float(row['cy'])))
pts = np.array(pts)
NP = len(pts)


def cell(r):
    ci = int(min(GRID - 1, max(0, (r['ax'] - XB[0]) / (XB[1] - XB[0]) * GRID)))
    cj = int(min(GRID - 1, max(0, (r['ay'] - YB[0]) / (YB[1] - YB[0]) * GRID)))
    return ci, cj


def nearest(r):
    d = np.hypot(pts[:, 0] - r['ax'], pts[:, 1] - r['ay'])
    return int(np.argmin(d))


rows = [json.loads(l) for l in open(LONG, encoding='utf-8')]


def analyze(region):
    R = [r for r in rows if r['region'] == region and r['t'] >= TMIN]
    inring = [r for r in R if r['rel'] <= 1.0]
    n_bucket = len(inring)

    # 实际击杀/场
    ak = defaultdict(lambda: {'k': 0.0, 'g': set()})
    for r in R:
        ak[r['team_name'].lower()]['k'] += r['kills']
        ak[r['team_name'].lower()]['g'].add(r['game'])
    ak = {n: a['k'] / len(a['g']) for n, a in ak.items() if len(a['g']) >= MIN_GAMES}
    names = sorted(ak)
    n_team = len(names)

    # ---- 圈相对 k(s) ----
    ks = defaultdict(lambda: [0.0, 0])
    for r in inring:
        s = (r['phase'], r['rel_bin'])
        ks[s][0] += r['kills']; ks[s][1] += 1
    k = {s: ks[s][0] / ks[s][1] for s in ks}
    sc_ring = defaultdict(lambda: [0.0, 0])
    for r in inring:
        sc_ring[r['team_name'].lower()][0] += k[(r['phase'], r['rel_bin'])]
        sc_ring[r['team_name'].lower()][1] += 1
    sc_ring = {n: sc_ring[n][0] / sc_ring[n][1] for n in sc_ring}

    # ---- 451 点位 β(p), 留一 ----
    K = np.zeros(NP); V = np.zeros(NP)
    tk = defaultdict(lambda: np.zeros(NP)); tv = defaultdict(lambda: np.zeros(NP))
    for r in inring:
        j = nearest(r)
        K[j] += r['kills']; V[j] += 1
        tk[r['team_name'].lower()][j] += r['kills']
        tv[r['team_name'].lower()][j] += 1
    sc_pt = {}
    for n in names:
        Kl = K - tk[n]; Vl = V - tv[n]
        b = np.where(Vl > 0, Kl / np.where(Vl == 0, 1, Vl), 0.0)
        w = tv[n].sum(); s = (tv[n] * b).sum()
        sc_pt[n] = s / w if w else 0.0

    # ---- 网格 β(c), 留一 ----
    GK = np.zeros((GRID, GRID)); GV = np.zeros((GRID, GRID))
    tk2 = defaultdict(lambda: np.zeros((GRID, GRID)))
    tv2 = defaultdict(lambda: np.zeros((GRID, GRID)))
    for r in inring:
        ci, cj = cell(r)
        GK[ci, cj] += r['kills']; GV[ci, cj] += 1
        tk2[r['team_name'].lower()][ci, cj] += r['kills']
        tv2[r['team_name'].lower()][ci, cj] += 1
    gr = GK.sum() / max(GV.sum(), 1)
    sc_grid = {}
    for n in names:
        Kl = GK - tk2[n] + ALPHA * gr
        Vl = GV - tv2[n] + ALPHA
        b = Kl / Vl
        w = tv2[n].sum(); s = (tv2[n] * b).sum()
        sc_grid[n] = s / w if w else 0.0

    xs = [ak[n] for n in names]
    def r_(sc):
        ys = [sc.get(n, 0.0) for n in names]
        return np.corrcoef(xs, ys)[0, 1]

    return {
        'region': region, 'n_team': n_team, 'n_games': len({r['game'] for r in R}),
        'n_bucket': n_bucket, 'avg_kill_pg': np.mean(xs),
        'ring': r_(sc_ring), 'point': r_(sc_pt), 'grid': r_(sc_grid),
    }


results = [analyze(r) for r in REGIONS]
print("击杀价值预测实际击杀的相关性 (留一交叉验证, 不剔弱队):")
print(f"{'region':<8} {'队':>3} {'场':>3} {'圈内桶':>7} {'击杀/场':>7} | "
      f"{'圈相对':>7} {'451点位':>7} {'16x16格':>7}")
for m in results:
    print(f"{m['region']:<8} {m['n_team']:>3} {m['n_games']:>3} {m['n_bucket']:>7} "
          f"{m['avg_kill_pg']:>7.2f} | {m['ring']:>+7.3f} {m['point']:>+7.3f} "
          f"{m['grid']:>+7.3f}")

# 点位相对圈相对的提升幅度
print("\n点位口径相对圈相对口径的提升 (Δr):")
for m in results:
    print(f"  {m['region']:<8} 451点位 Δ={m['point']-m['ring']:+.3f}   "
          f"16x16网格 Δ={m['grid']-m['ring']:+.3f}")