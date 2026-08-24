#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_competitive_kill.py — 只留竞争队伍, 看点位击杀价值是否有信号

用户: 击杀分预测差可能因为"数据太少 + 有些队太菜"。这里把场均名次差(>=11名)
的队伍剔掉, 只在竞争队伍里重算: 圈相对 φ_kill vs 点位 β_kill 预测实际击杀的相关性。
"""
import json, glob, os
from collections import defaultdict
import numpy as np

LONG = "data/long_table.jsonl"
PHI_JSON = "data/phi.json"
POS = "data/positions.csv"
TMIN = 120
MAX_PLACE = 11      # 场均名次 <= 该值 算竞争队
MIN_GAMES = 5

# ---- 场均名次 (replay summary) ----
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

rows = [json.loads(l) for l in open(LONG, encoding='utf-8')]
apac = [r for r in rows if r['region'] == 'apac-s' and r['t'] >= TMIN
        and r['team_name'].lower() in comp]
print(f"竞争队伍 {len(comp)} 支 (场均名次<={MAX_PLACE}, 场次>={MIN_GAMES})")

j = json.load(open(PHI_JSON, encoding='utf-8'))
pk_old = {(int(k.split(',')[0]), int(k.split(',')[1])): v
          for k, v in j['phi_kill'].items()}


def team_ak():
    a = defaultdict(lambda: {'k': 0.0, 'g': set()})
    for r in apac:
        nm = r['team_name'].lower()
        a[nm]['k'] += r['kills']
        a[nm]['g'].add(r['game'])
    return {nm: a[nm]['k'] / len(a[nm]['g']) for nm in a}


def corr(scores):
    ak = team_ak()
    names = sorted(ak)
    return np.corrcoef([ak[n] for n in names],
                       [scores.get(n, 0.0) for n in names])[0, 1]


# 圈相对 φ_kill
s_old = defaultdict(lambda: {'s': 0.0, 'w': 0})
for r in apac:
    s_old[r['team_name'].lower()]['w'] += 1
    if r['rel'] <= 1.0:
        s_old[r['team_name'].lower()]['s'] += pk_old.get((r['phase'], r['rel_bin']), 0.0)
scores_old = {n: a['s'] / a['w'] for n, a in s_old.items()}
print(f"圈相对 φ_kill:            r = {corr(scores_old):+.3f}")

# 451 点位 β_kill (无收缩)
kills = np.zeros(len(pos_arr)); visits = np.zeros(len(pos_arr))
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
scores_pts = {n: a['s'] / a['w'] for n, a in s_pts.items()}
print(f"451 点位 (无收缩):        r = {corr(scores_pts):+.3f}")

# 16x16 网格 + 收缩
xmin, xmax = pos_arr[:, 0].min(), pos_arr[:, 0].max()
ymin, ymax = pos_arr[:, 1].min(), pos_arr[:, 1].max()
G = 16
nk = np.zeros((G, G)); nv = np.zeros((G, G))
for r in apac:
    if r['rel'] > 1.0:
        continue
    ci = int(min(G - 1, max(0, (r['ax'] - xmin) / (xmax - xmin) * G)))
    cj = int(min(G - 1, max(0, (r['ay'] - ymin) / (ymax - ymin) * G)))
    nk[ci, cj] += r['kills']; nv[ci, cj] += 1
global_rate = nk.sum() / max(nv.sum(), 1)
beta_g = (nk + 30 * global_rate) / (nv + 30)
s_g = defaultdict(lambda: {'s': 0.0, 'w': 0})
for r in apac:
    s_g[r['team_name'].lower()]['w'] += 1
    if r['rel'] <= 1.0:
        ci = int(min(G - 1, max(0, (r['ax'] - xmin) / (xmax - xmin) * G)))
        cj = int(min(G - 1, max(0, (r['ay'] - ymin) / (ymax - ymin) * G)))
        s_g[r['team_name'].lower()]['s'] += beta_g[ci, cj]
scores_g = {n: a['s'] / a['w'] for n, a in s_g.items()}
print(f"16x16 网格 + 收缩:       r = {corr(scores_g):+.3f}")