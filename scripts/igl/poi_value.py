#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
poi_value.py — 点位级 xT 价值 + 点位级 IGL 评分

把位置从"离圈心多远"升级到"踩的是哪个点位":
  1. 每个队伍-桶锚点 -> 判定落在哪个 POI (点在世界坐标多边形内), 落不到 = open 野区
  2. POI 价值用回归解出 (净掉圈位置 + 队伍实力两个混淆):
       最终分 - 环形期望E_final = Σ(在该POI停留占比)·β_poi + 队伍固定效应 + ε
     β_poi = 该点位的"地皮价值" (掩体/高度/搜刮/转点), 独立于圈和队伍
  3. IGL 评分 v3 = 位置质量(环形 + 点位) + 执行残差
"""
import json, math, numpy as np
from collections import defaultdict

RATIO = 24.93
REPLAY = "replay/storm point/sp_apac-s_d2_g6_99c6fb41.json"  # 拿 POI 几何(同图所有场一致)
LONG = "data/long_table.jsonl"
MIN_GAMES = 3

# ---- 1. POI 多边形 (像素 -> 世界) ----
d = json.load(open(REPLAY, encoding='utf-8'))
polys = d['pois']['payload']['polygons']


def pix_to_world(px, py):
    return ((px - 8192) * RATIO / 4, -(py - 8192) * RATIO / 4)


pois = []
for p in polys:
    w = [pix_to_world(c['x'], c['y']) for c in p['coordinates']]
    xs = [q[0] for q in w]
    ys = [q[1] for q in w]
    pois.append({'name': p['name'], 'poly': w, 'bbox': (min(xs), min(ys), max(xs), max(ys))})


def point_in_poly(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def which_poi(ax, ay):
    for p in pois:
        x0, y0, x1, y1 = p['bbox']
        if x0 <= ax <= x1 and y0 <= ay <= y1 and point_in_poly((ax, ay), p['poly']):
            return p['name']
    return "open"


# ---- 2. 读长表, 每行落 POI ----
rows = [json.loads(l) for l in open(LONG, encoding='utf-8')]
poi_of_row = [which_poi(r['ax'], r['ay']) for r in rows]

final = defaultdict(float)
kp = defaultdict(float)
pp = defaultdict(float)
for i, r in enumerate(rows):
    key = (r['game'], r['team'])
    final[key] += r['kills'] + r['placement_pts']
    kp[key] += r['kills']
    pp[key] += r['placement_pts']

# 环形期望 E_final
visits = defaultdict(int)
ef = defaultdict(float)
for i, r in enumerate(rows):
    s = (r['phase'], r['rel_bin'])
    visits[s] += 1
    ef[s] += final[(r['game'], r['team'])]
E_final = {s: ef[s] / visits[s] for s in visits}

# ---- 3. 原始 POI 值 (bucket 级 mean residual, 未净队伍) ----
poi_raw_sum = defaultdict(float)
poi_raw_cnt = defaultdict(int)
for i, r in enumerate(rows):
    resid = final[(r['game'], r['team'])] - E_final[(r['phase'], r['rel_bin'])]
    poi_raw_sum[poi_of_row[i]] += resid
    poi_raw_cnt[poi_of_row[i]] += 1
poi_raw = {pn: poi_raw_sum[pn] / poi_raw_cnt[pn] for pn in poi_raw_cnt}

# ---- 4. 队伍固定效应回归 (ridge) ----
poi_predictors = [p['name'] for p in pois]  # open 做 baseline
teams_grp = defaultdict(list)
for i, r in enumerate(rows):
    teams_grp[(r['game'], r['team'])].append(i)

team_names = sorted({r['team_name'] for r in rows})
team_idx = {t: i for i, t in enumerate(team_names)}
P_ = len(poi_predictors)
T_ = len(team_names)

unit = []
for (gid, tid), idxs in teams_grp.items():
    rws = [rows[i] for i in idxs]
    pois_here = [poi_of_row[i] for i in idxs]
    ring_exp = sum(E_final[(r['phase'], r['rel_bin'])] for r in rws) / len(rws)
    cnt = defaultdict(int)
    for pn in pois_here:
        cnt[pn] += 1
    n = len(rws)
    frac = {pn: cnt.get(pn, 0) / n for pn in poi_predictors}
    unit.append({'name': rws[0]['team_name'], 'gid': gid, 'n': n,
                 'ring_exp': ring_exp, 'frac': frac, 'final': final[(gid, tid)],
                 'kp': kp[(gid, tid)], 'pp': pp[(gid, tid)]})

n = len(unit)
X = np.zeros((n, P_ + T_ - 1))
y = np.zeros(n)
for i, u in enumerate(unit):
    y[i] = u['final'] - u['ring_exp']
    for j, pn in enumerate(poi_predictors):
        X[i, j] = u['frac'][pn]
    ti = team_idx[u['name']]
    if ti > 0:
        X[i, P_ + ti - 1] = 1.0

lam = 0.5
A = X.T @ X
b = X.T @ y
Areg = A.copy()
Areg[:P_, :P_] += lam * np.eye(P_)
beta = np.linalg.solve(Areg, b)
beta_poi = beta[:P_]

pred = X @ beta
ss_res = ((y - pred) ** 2).sum()
ss_tot = ((y - y.mean()) ** 2).sum()
r2 = 1 - ss_res / ss_tot

# ---- 输出 1: POI 价值榜 ----
print("=" * 78)
print(f"点位价值 β_poi (净掉圈位置 + 队伍固定效应后; ridge λ={lam}; 回归 R²={r2:.2f})")
print("  含义: 该点位每停留 100% 时间, 比 open 野区多拿的分")
print("=" * 78)
print(f"{'点位':<20}{'桶样本':>7}{'原始值':>9}{'净队伍后':>10}")
rank = []
for j, pn in enumerate(poi_predictors):
    rank.append((pn, poi_raw.get(pn, 0.0), beta_poi[j], poi_raw_cnt.get(pn, 0)))
rank.sort(key=lambda x: -x[2])
for pn, raw, b, c in rank:
    print(f"{pn:<20}{c:>7}{raw:>+9.2f}{b:>+10.2f}")
print(f"{'open(野区)':<20}{poi_raw_cnt.get('open',0):>7}{poi_raw.get('open',0):>+9.2f}{'0.00 (基准)':>10}")

# ---- 输出 2: POI 级 IGL 评分 ----
print("\n" + "=" * 78)
print(f"点位级 IGL 评分 (只显示 >= {MIN_GAMES} 场)")
print("  位置质量 = 环形期望 + 点位价值 (IGL 带队的选点+转点能力)")
print("  执行残差 = 真实终分 - 位置质量 (选点之外的对枪/执行)")
print("=" * 78)
poi_val = {pn: beta_poi[j] for j, pn in enumerate(poi_predictors)}

by_name = defaultdict(lambda: {'games': 0, 'ring': 0.0, 'poi': 0.0, 'final': 0.0})
for u in unit:
    nm = u['name']
    pv = sum(poi_val[pn] * u['frac'][pn] for pn in poi_predictors)
    d = by_name[nm]
    d['games'] += 1
    d['ring'] += u['ring_exp']
    d['poi'] += pv
    d['final'] += u['final']

ranking = []
for nm, dd in by_name.items():
    g = dd['games']
    ranking.append((nm, g, dd['ring'] / g, dd['poi'] / g, dd['final'] / g,
                    (dd['final'] - dd['ring'] - dd['poi']) / g))

multi = [x for x in ranking if x[1] >= MIN_GAMES]
multi.sort(key=lambda x: -(x[2] + x[3]))  # 按位置质量排
print(f"{'队伍':<22}{'场':>3}{'环形期望':>8}{'点位价值':>8}{'位置质量':>8}{'真实终分':>8}{'执行残差':>8}")
for nm, g, ring, poi, f, res in multi:
    print(f"{nm:<22}{g:>3}{ring:>8.2f}{poi:>+8.2f}{ring+poi:>8.2f}{f:>8.2f}{res:>+8.2f}")

# 保存
with open("data/poi_value.csv", "w") as f:
    f.write("poi,buckets,raw_value,fe_value\n")
    for pn, raw, b, c in rank:
        f.write(f"{pn},{c},{raw:.4f},{b:.4f}\n")
    f.write(f"open,{poi_raw_cnt.get('open',0)},{poi_raw.get('open',0):.4f},0.0\n")

with open("data/igl_ranking_poi.csv", "w") as f:
    f.write("team,games,ring_quality,poi_quality,position_quality,final_score,execution_residual\n")
    for nm, g, ring, poi, fin, res in ranking:
        f.write(f"{nm},{g},{ring:.4f},{poi:.4f},{ring+poi:.4f},{fin:.4f},{res:.4f}\n")

print("\n已保存: data/poi_value.csv, data/igl_ranking_poi.csv")
