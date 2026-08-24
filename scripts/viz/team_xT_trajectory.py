#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
team_xT_trajectory.py — 一场具体比赛里, 每支队伍的 xT 值随时间的轨迹

对每个时间桶, 取队伍锚点 -> 找最近点位 p -> xT(p, 当前圈) = β_p + φ_m(rel_bin)
rel>1 (毒里) 软处理: 短暂进出(≤15s)或圈1/圈2 按圈内边缘计值, 圈3+ 持续在毒里才记 0。
把吃鸡队伍高亮, 其余 19 队淡色铺底。
同时输出吃鸡队伍的 击杀/排名 分解。
"""
import json, os, math, bisect
import numpy as np

REL_BINS = [0.3, 0.7, 1.0, 1.15, 1.6]
BUCKET = 5
BRIEF_SEC = 15   # 毒内连续停留 ≤15s 视为"短暂进出", 不强制清零
SOFT_BIN = 2     # 软处理时按"圈内边缘"(rel_bin=2) 计值
VEL_WINDOW = 15  # 速度窗口(秒): 判"踩点/守家"看过去15s位移
STAY_VEL = 400   # 位移 <400 单位 => 算踩点; 否则算出去动/转点
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


def team_anchor(alive, t):
    """队伍锚点 = '踩点的人'(过去 VEL_WINDOW 内没怎么动)的质心。
    解决: 2人出去拿人头+1人守家时, 锚点应落在守家的人而非出去的2人。
    若都在动(整队转点), 退回旧逻辑'最靠拢两名的质心'。"""
    pos = [pos_at(pl['pts'], pl['ts'], t) for pl in alive]
    n = len(pos)
    if n == 0:
        return None
    if n == 1:
        return pos[0]
    stay = [p for pl, p in zip(alive, pos)
            if dist(pos_at(pl['pts'], pl['ts'], t - VEL_WINDOW), p) < STAY_VEL]
    if stay:
        return centroid(stay)
    if n == 2:
        return centroid(pos)
    p0, p1, p2 = pos
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


def xT_of(x, y, t, brief=False):
    """返回 (xT_kill, xT_place, rel)"""
    ph, (cx, cy), r = ring_at(t)
    rel = math.hypot(x - cx, y - cy) / r
    j = int(np.argmin(np.hypot(pos_arr[:, 0] - x, pos_arr[:, 1] - y)))
    if rel <= 1.0:
        b = rel_bin(rel)
        return bk[j] + pk[(ph, b)], bp[j] + pp[(ph, b)], rel
    if brief or ph < 2:     # 短暂进出 或 圈1/圈2: 不强制清零, 按圈内边缘计值
        return bk[j] + pk[(ph, SOFT_BIN)], bp[j] + pp[(ph, SOFT_BIN)], rel
    return 0.0, 0.0, rel


# 逐队轨迹
def mark_brief(rels, bucket_sec=BUCKET, brief_sec=BRIEF_SEC):
    """连续毒内停留 ≤brief_sec 的桶标记为'短暂进出'(不清零)。rels 含 None 表缺桶。"""
    brief = [False] * len(rels)
    i = 0
    while i < len(rels):
        if rels[i] is not None and rels[i] > 1.0:
            j = i
            while j < len(rels) and rels[j] is not None and rels[j] > 1.0:
                j += 1
            if (j - i) * bucket_sec <= brief_sec:
                for k in range(i, j):
                    brief[k] = True
            i = j
        else:
            i += 1
    return brief


traj = {}
for tid, team in teams.items():
    n_buckets = int(dur // BUCKET) + 1
    # 第一遍: 算每个桶的锚点 + rel, 用来判"短暂进出毒"
    anchors, rels = [], []
    for b in range(n_buckets):
        t = b * BUCKET
        if team['elim'] < t:
            break
        alive = [pl for pl in team['players']
                 if pl['ts'][0] <= t <= pl['ts'][-1]]
        if not alive:
            anchors.append(None); rels.append(None)
            continue  # 队员开局前/复活间隙缺轨迹, 跳过该桶而非截断整条轨迹
        a = team_anchor(alive, t)
        ph, (cx, cy), r = ring_at(t)
        anchors.append(a)
        rels.append(math.hypot(a[0] - cx, a[1] - cy) / r)
    brief = mark_brief(rels)
    # 第二遍: 算 xT
    ts = []
    for b in range(len(anchors)):
        a = anchors[b]
        if a is None:
            continue
        t = b * BUCKET
        xk, xp, rel = xT_of(a[0], a[1], t, brief=brief[b])
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