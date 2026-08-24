#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
team_xT_trajectory.py — 一场具体比赛里, 每支队伍的 xT 值随时间的轨迹

对每个时间桶, 取队伍锚点 -> 找最近点位 p -> xT(p, 当前圈) = β_p + φ_m(rel_bin)
rel>1 (毒里) 记 0 (该点当前无价值)。把吃鸡队伍高亮, 其余 19 队淡色铺底。
同时输出吃鸡队伍的 击杀/排名 分解。
"""
import json, os, math, bisect
import numpy as np

REL_BINS = [0.3, 0.7, 1.0, 1.15, 1.6]
BUCKET = 5
GAME = "replay/storm point/sp_global_d__g9_29ce1521.json"

phi = json.load(open("data/phi.json", encoding='utf-8'))
pk = {(int(a), int(b)): v for a, b, v in
      (k.split(',') + [v] for k, v in phi['phi_kill'].items())}
pp = {(int(a), int(b)): v for a, b, v in
      (k.split(',') + [v] for k, v in phi['phi_place'].items())}
bk = np.load("data/beta_kill.npy")
bp = np.load("data/beta_place.npy")
pos_arr = np.load("data/points_pos.npy")


def rel_bin(rel):
    for i, th in enumerate(REL_BINS):
        if rel < th:
            return i
    return len(REL_BINS)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def centroid(pts):
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def team_anchor(ps):
    n = len(ps)
    if n == 0:
        return None
    if n == 1:
        return ps[0]
    if n == 2:
        return centroid(ps)
    p0, p1, p2 = ps
    d01, d02, d12 = dist(p0, p1), dist(p0, p2), dist(p1, p2)
    if d01 <= d02 and d01 <= d12:
        return centroid([p0, p1])
    if d02 <= d01 and d02 <= d12:
        return centroid([p0, p2])
    return centroid([p1, p2])


def pos_at(pts, ts, t):
    if t <= ts[0]:
        return (pts[0]['x'], pts[0]['y'])
    if t >= ts[-1]:
        return (pts[-1]['x'], pts[-1]['y'])
    i = bisect.bisect_left(ts, t)
    a, b = pts[i - 1], pts[i]
    frac = (t - a['tsGame']) / (b['tsGame'] - a['tsGame'])
    return (a['x'] + frac * (b['x'] - a['x']), a['y'] + frac * (b['y'] - a['y']))


d = json.load(open(GAME, encoding='utf-8'))
s = d['summary']
dur, g0 = s['duration'], s['gameStartTs']

# 圈时间线
sc, fc = {}, {}
for r in s['ringPhases']:
    if r['type'] == 'startClosing':
        sc[r['stage']] = r
    elif r['type'] == 'finishedClosing':
        fc[r['stage']] = r
order = sorted(sc.keys())
stages = {}
for k in order:
    a, b = sc[k], fc.get(k)
    c0 = (a['center']['x'], a['center']['y'])
    c1 = (b['center']['x'], b['center']['y']) if b else c0
    stages[k] = {'c0': c0, 'c1': c1, 'r0': a['startRadius'], 'r1': a['endRadius'],
                 't0': a['timestamp'] - g0, 't1': (b['timestamp'] - g0) if b else dur}


def ring_at(t):
    for i in range(len(order) - 1, -1, -1):
        st = stages[order[i]]
        if t >= st['t0']:
            p = min(1.0, (t - st['t0']) / max(st['t1'] - st['t0'], 1e-6))
            cx = st['c0'][0] + (st['c1'][0] - st['c0'][0]) * p
            cy = st['c0'][1] + (st['c1'][1] - st['c0'][1]) * p
            r = st['r0'] + (st['r1'] - st['r0']) * p
            return i, (cx, cy), r
    st = stages[order[0]]
    return 0, st['c0'], st['r0']


# 队伍轨迹 + 名次 + 队名
teams = {}
for t in d['pathing']['teams']:
    tid = t['teamId']
    players, max_last = [], 0
    for pl in t['players']:
        pts = pl.get('points') or []
        if not pts:
            continue
        pts.sort(key=lambda p: p['tsGame'])
        players.append({'ts': [p['tsGame'] for p in pts], 'pts': pts})
        max_last = max(max_last, pts[-1]['tsGame'])
    teams[tid] = {'players': players, 'elim': min(max_last, dur)}
for st in s['teams']:
    tid = st['teamId']
    if tid in teams:
        teams[tid]['rank'] = st['placement']
        teams[tid]['name'] = st.get('teamName', f"T{tid}")


def xT_of(x, y, t):
    """返回 (xT_kill, xT_place, rel)"""
    ph, (cx, cy), r = ring_at(t)
    rel = math.hypot(x - cx, y - cy) / r
    if rel > 1.0:
        return 0.0, 0.0, rel
    j = int(np.argmin(np.hypot(pos_arr[:, 0] - x, pos_arr[:, 1] - y)))
    b = rel_bin(rel)
    return bk[j] + pk[(ph, b)], bp[j] + pp[(ph, b)], rel


# 逐队轨迹
traj = {}
for tid, team in teams.items():
    ts = []
    n_buckets = int(dur // BUCKET) + 1
    for b in range(n_buckets):
        t = b * BUCKET
        if team['elim'] < t:
            break
        alive = [pos_at(pl['pts'], pl['ts'], t) for pl in team['players']
                 if pl['ts'][0] <= t <= pl['ts'][-1]]
        if not alive:
            continue  # 队员开局前/复活间隙缺轨迹, 跳过该桶而非截断整条轨迹
        a = team_anchor(alive)
        xk, xp, rel = xT_of(a[0], a[1], t)
        ts.append((t, xk, xp, xk + xp, rel))
    traj[tid] = ts

winner = [tid for tid, t in teams.items() if t.get('rank') == 1][0]
wname = teams[winner]['name']

# 圈阶段边界时间 (画竖线)
ring_ts = [stages[k]['t1'] for k in order]

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

# 图1: 20 队 xT_total 轨迹
fig, ax = plt.subplots(figsize=(14, 6))
names = {}
for tid, t in teams.items():
    names[tid] = t.get('name', f"T{tid}")
for tid, ts in traj.items():
    if not ts:
        continue
    tt = [r[0] for r in ts]
    vv = [r[3] for r in ts]
    if tid == winner:
        ax.plot(tt, vv, color='#d62728', lw=3, zorder=5, label=f'{wname} (冠军)')
    else:
        ax.plot(tt, vv, color='#999', lw=1, alpha=0.55, zorder=2)
for rt in ring_ts:
    ax.axvline(rt, color='cyan', ls='--', lw=1, alpha=0.5)
ax.set_xlabel('比赛时间 (秒)')
ax.set_ylabel('点位 xT (击杀+排名)')
ax.set_title(f'世界赛决赛 Game#9 — 20 队点位 xT 值随圈型变化 (冠军 {wname} 高亮)')
ax.legend(loc='upper left', fontsize=9)
ax.set_ylim(bottom=0)
fig.tight_layout()
out1 = "output/team_xT_trajectory.png"
plt.savefig(out1, dpi=110, bbox_inches='tight')
print("已写", out1)

# 图2: 吃鸡队伍 击杀/排名 分解
fig, ax = plt.subplots(figsize=(14, 6))
tt = [r[0] for r in traj[winner]]
ax.plot(tt, [r[1] for r in traj[winner]], color='#d62728', lw=2,
        label='击杀价值 xT_kill')
ax.plot(tt, [r[2] for r in traj[winner]], color='#1f77b4', lw=2,
        label='排名价值 xT_place')
ax.plot(tt, [r[3] for r in traj[winner]], color='#333', lw=1.5, ls=':',
        label='合计 xT')
for rt in ring_ts:
    ax.axvline(rt, color='cyan', ls='--', lw=1, alpha=0.5)
ax.set_xlabel('比赛时间 (秒)')
ax.set_ylabel('xT 价值')
ax.set_title(f'冠军 {wname} — 击杀价值 vs 排名价值 随时间')
ax.legend(loc='upper left', fontsize=9)
fig.tight_layout()
out2 = "output/winner_xT_decompose.png"
plt.savefig(out2, dpi=110, bbox_inches='tight')
print("已写", out2)

# 数值摘要: 各队 后期(圈3后) 平均 xT
print(f"\n冠军: {wname}")
LATE_T0 = stages[order[2]]['t0']  # Apex 圈3 开始 (后期 = 圈3+; stage 2→圈3)
late = [t for t in ring_ts if t > LATE_T0]
print(f"各队 后期(圈3+, t>={LATE_T0:.0f}s)平均 xT_total (按名次排序):")
ranked = sorted(teams.items(), key=lambda kv: kv[1].get('rank', 99))
for tid, t in ranked:
    ts = traj.get(tid, [])
    vals = [r[3] for r in ts if r[0] >= LATE_T0]
    if vals:
        print(f"  #{t.get('rank',99):>2} {names[tid]:20} 后期均xT={np.mean(vals):.2f}  峰值={max(vals):.2f}")