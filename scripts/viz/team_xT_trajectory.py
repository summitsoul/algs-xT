#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
team_xT_trajectory.py — 一场具体比赛里, 每支队伍的 xT 值随时间的轨迹

对每个时间桶, 取队伍锚点 -> 找最近点位 p ->
xT(p, 圈) = kill(直接局部均值, 按当前圈分档) + place(β_p_place + φ_place, 非折现, 毒里不再硬记0, 非负)。
把吃鸡队伍高亮, 其余 19 队淡色铺底。
同时输出吃鸡队伍的 击杀/排名 分解。
"""
import json, os, math, bisect
import numpy as np

REL_BINS = [0.3, 0.7, 1.0, 1.15, 1.6]
BUCKET = 5
VEL_WINDOW = 15  # 速度窗口(秒): 判"踩点/守家"看过去15s位移
STAY_VEL = 400   # 位移 <400 单位 => 算踩点; 否则算出去动/转点
GAME = "replay/storm point/sp_global_d__g9_29ce1521.json"

phi = json.load(open("data/phi.json", encoding='utf-8'))
pp = {(int(a), int(b)): v for a, b, v in
      (k.split(',') + [v] for k, v in phi['phi_place'].items())}
bk = np.load("data/beta_kill.npy")     # 击杀 = 直接局部均值
bp = np.load("data/beta_place.npy")
pos_arr = np.load("data/points_pos.npy")


def rel_bin(rel):
    for i, th in enumerate(REL_BINS):
        if rel < th:
            return i
    return len(REL_BINS)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


MIN_ZONE_R = 500.0  # 新圈(安全区)半径 < 此值视为无安全区(最后缩到一点), rel 退回毒环、β_p 不衰减
R0 = 31000.0        # 圈1 新圈(安全区)半径, w_stage 归一化基准


def zone_rel(a, c1, r1, c_ring, r_ring):
    """新圈基准 rel(圈内/圈外判据)。常规圈用新圈(安全区)中心/半径; 最后圈(r1≈0)退回正在缩的毒环。"""
    if r1 >= MIN_ZONE_R:
        return dist(a, c1) / r1
    return dist(a, c_ring) / max(r_ring, 1.0)


def zone_w(rel, r1):
    """β_p 圈外平滑衰减权重。最后圈不衰减=1.0; 常规圈 rel<=1 全额, rel 1→1.6 线性降到0, rel>=1.6 归零。"""
    if r1 < MIN_ZONE_R:
        return 1.0
    return max(0.0, min(1.0, (1.6 - rel) / 0.6))


def stage_w(r1):
    """β_p 圈阶段权重: 圈越大(越早)点位固有价值兑现越低, 缩到决赛圈才全额。
    w_stage = 1 - r1/R0 → 圈1=0, 圈2≈0.52, 圈3≈0.74, 圈4≈0.87, 圈5≈0.94, 圈6≈1.0。"""
    return max(0.0, 1.0 - r1 / R0)


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


def xT_of(x, y, t):
    """返回 (xT_kill, xT_place, rel)"""
    ph, (cx, cy), r = ring_at(t)
    c1, r1 = stages[ph]['c1'], stages[ph]['r1']    # 新圈(安全区)中心/半径
    rel = zone_rel((x, y), c1, r1, (cx, cy), r)
    j = int(np.argmin(np.hypot(pos_arr[:, 0] - x, pos_arr[:, 1] - y)))
    xk = bk[j][ph]                                # 击杀: 直接局部均值(按当前圈分档)
    xp = max(0.0, zone_w(rel, r1) * stage_w(r1) * bp[j] + pp[(ph, rel_bin(rel))])  # 排名: β_p 圈外衰减 + 圈阶段权重 + 非折现 + 非负
    return xk, xp, rel


# 逐队轨迹
traj = {}
for tid, team in teams.items():
    n_buckets = int(dur // BUCKET) + 1
    ts = []
    for b in range(n_buckets):
        t = b * BUCKET
        if team['elim'] < t:
            break
        alive = [pl for pl in team['players']
                 if pl['ts'][0] <= t <= pl['ts'][-1]]
        if not alive:
            continue  # 队员开局前/复活间隙缺轨迹, 跳过该桶而非截断整条轨迹
        a = team_anchor(alive, t)
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