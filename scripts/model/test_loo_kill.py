#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_loo_kill.py — 留一交叉验证: 点位击杀价值的"自证循环"检验

怀疑: 451 点位 β 是拿同一批竞争队自己的击杀算的, 队 A 在某点杀得多 → 该点 β 高
→ 队 A 又常去那点 → 相关被抬高(循环)。

留一: 对每支队 T, 用"排除 T 自己的击杀/访问"后的 β_{-T} 来给它打分。
如果位置真带独立信号, 留一后 r 应该仍明显 > 0; 若塌到 ~0, 说明之前是循环。
"""
import json, glob, os
from collections import defaultdict
import numpy as np

LONG = "data/long_table.jsonl"
POS = "data/positions.csv"
TMIN = 120
MIN_GAMES = 5
MAX_PLACE = 11

# 竞争队
place = defaultdict(lambda: {'s': 0, 'n': 0})
for path in glob.glob('replay/storm point/*.json'):
    if os.path.basename(path).split('_')[1] != 'apac-s':
        continue
    s = json.load(open(path, encoding='utf-8'))['summary']
    for t in s['teams']:
        nm = t.get('teamName', '?').lower()
        place[nm]['s'] += t['placement']
        place[nm]['n'] += 1
comp = {nm for nm, a in place.items()
        if a['n'] >= MIN_GAMES and a['s'] / a['n'] <= MAX_PLACE}

pos = []
for line in open(POS, encoding='utf-8'):
    if line.startswith('id'):
        continue
    p = line.strip().split(',')
    pos.append((float(p[1]), float(p[2])))
pos_arr = np.array(pos)
NP = len(pos_arr)

rows = [json.loads(l) for l in open(LONG, encoding='utf-8')]
apac = [r for r in rows if r['region'] == 'apac-s' and r['t'] >= TMIN
        and r['team_name'].lower() in comp]

# 每队: 实际击杀/场, 以及 (点id -> 击杀, 访问) 供留一扣除
teams = defaultdict(lambda: {'k': 0.0, 'g': set(), 'pt_k': defaultdict(float),
                             'pt_v': defaultdict(float)})
for r in apac:
    nm = r['team_name'].lower()
    t = teams[nm]
    t['k'] += r['kills']
    t['g'].add(r['game'])
    if r['rel'] <= 1.0:
        d = np.hypot(pos_arr[:, 0] - r['ax'], pos_arr[:, 1] - r['ay'])
        j = int(np.argmin(d))
        t['pt_k'][j] += r['kills']
        t['pt_v'][j] += 1

names = sorted(teams)
ak = {n: teams[n]['k'] / len(teams[n]['g']) for n in names}

# 全场聚合 (含所有竞争队)
K = np.zeros(NP); V = np.zeros(NP)
for n in names:
    for j in teams[n]['pt_k']:
        K[j] += teams[n]['pt_k'][j]
        V[j] += teams[n]['pt_v'][j]
beta_all = np.where(V > 0, K / np.where(V == 0, 1, V), 0.0)

# 口径1: 无留一 (循环版)
s_all = {}
for n in names:
    w = 0; s = 0.0
    for j in teams[n]['pt_k']:
        s += beta_all[j] * teams[n]['pt_v'][j]
        w += teams[n]['pt_v'][j]
    s_all[n] = s / w
r_all = np.corrcoef([ak[n] for n in names], [s_all[n] for n in names])[0, 1]
print(f"451 点位 无留一(循环):   r = {r_all:+.3f}")

# 口径2: 留一 (每队用自己的 β_{-T})
s_loo = {}
for n in names:
    Kl = K.copy(); Vl = V.copy()
    for j in teams[n]['pt_k']:
        Kl[j] -= teams[n]['pt_k'][j]
        Vl[j] -= teams[n]['pt_v'][j]
    b = np.where(Vl > 0, Kl / np.where(Vl == 0, 1, Vl), 0.0)
    w = 0; s = 0.0
    for j in teams[n]['pt_k']:
        s += b[j] * teams[n]['pt_v'][j]
        w += teams[n]['pt_v'][j]
    s_loo[n] = s / w
r_loo = np.corrcoef([ak[n] for n in names], [s_loo[n] for n in names])[0, 1]
print(f"451 点位 留一交叉验证:   r = {r_loo:+.3f}")

# 网格版留一 (16x16 + 收缩), 作为稳健对照
xmin, xmax = pos_arr[:, 0].min(), pos_arr[:, 0].max()
ymin, ymax = pos_arr[:, 1].min(), pos_arr[:, 1].max()
G = 16
ALPHA = 30


def cell(r):
    ci = int(min(G - 1, max(0, (r['ax'] - xmin) / (xmax - xmin) * G)))
    cj = int(min(G - 1, max(0, (r['ay'] - ymin) / (ymax - ymin) * G)))
    return ci, cj


# 每队每格 击杀/访问
tk = defaultdict(lambda: np.zeros((G, G)))
tv = defaultdict(lambda: np.zeros((G, G)))
for r in apac:
    if r['rel'] <= 1.0:
        ci, cj = cell(r)
        tk[r['team_name'].lower()][ci, cj] += r['kills']
        tv[r['team_name'].lower()][ci, cj] += 1

KK = sum(tk[n] for n in names)
VV = sum(tv[n] for n in names)
gr = KK.sum() / max(VV.sum(), 1)
beta_g_all = (KK + ALPHA * gr) / (VV + ALPHA)

s_g_all = {}
s_g_loo = {}
for n in names:
    for lbl, beta in [('all', beta_g_all),
                      ('loo', (KK - tk[n] + ALPHA * gr) / (VV - tv[n] + ALPHA))]:
        w = tv[n].sum(); s = (tv[n] * beta).sum()
        (s_g_all if lbl == 'all' else s_g_loo)[n] = s / max(w, 1)
r_g_all = np.corrcoef([ak[n] for n in names], [s_g_all[n] for n in names])[0, 1]
r_g_loo = np.corrcoef([ak[n] for n in names], [s_g_loo[n] for n in names])[0, 1]
print(f"16x16网格 无留一:        r = {r_g_all:+.3f}")
print(f"16x16网格 留一交叉:     r = {r_g_loo:+.3f}")