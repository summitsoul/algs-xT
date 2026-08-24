#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spike_xT.py — 单场 xT 实验 (Storm Point, 1 场)

用 Bellman 递推: V(s) = k(s) + γ·[ (1−h(s))·E[V(s')] + h(s)·pl(s) ]
  - k(s)  = 状态 s 的期望击杀分/桶
  - h(s)  = 状态 s 下一桶被淘汰的概率
  - pl(s) = 从状态 s 被淘汰时拿到的排名分
  - P(s'|s) = 存活转移概率

状态 s = (第几圈 phase, 圈相对位置 rel_bin)
  圈相对: rel = 到当前圈中心距离 / 当前圈起始半径(startRadius)
  rel_bin: 0=中心(<0.3) 1=中内(0.3-0.7) 2=圈边(0.7-1.1) 3=圈外(>1.1)

验证: 圈心 > 圈边 > 圈外 是否成立; V 是否随圈递增
"""
import json, math, sys, bisect
from collections import defaultdict

PLACEMENT_PTS = {1: 12, 2: 9, 3: 7, 4: 5, 5: 4, 6: 3, 7: 3,
                 8: 2, 9: 2, 10: 2, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1}
BUCKET = 5          # 时间桶(秒)
GAMMA = 0.99
REL_BINS = [0.3, 0.7, 1.1]
REL_NAMES = ['中心', '中内', '圈边', '圈外']


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def centroid(pts):
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def team_anchor(positions):
    """队伍锚点: 取最靠拢的两名队员的质心(忽略单走的第三人)"""
    n = len(positions)
    if n == 0:
        return None
    if n == 1:
        return positions[0]
    if n == 2:
        return centroid(positions)
    p0, p1, p2 = positions
    d01, d02, d12 = dist(p0, p1), dist(p0, p2), dist(p1, p2)
    if d01 <= d02 and d01 <= d12:
        return centroid([p0, p1])
    if d02 <= d01 and d02 <= d12:
        return centroid([p0, p2])
    return centroid([p1, p2])


def pos_at(pts, ts, t):
    """线性插值求 t 时刻位置"""
    if t <= ts[0]:
        return (pts[0]['x'], pts[0]['y'])
    if t >= ts[-1]:
        return (pts[-1]['x'], pts[-1]['y'])
    i = bisect.bisect_left(ts, t)
    a, b = pts[i - 1], pts[i]
    frac = (t - a['tsGame']) / (b['tsGame'] - a['tsGame'])
    return (a['x'] + frac * (b['x'] - a['x']), a['y'] + frac * (b['y'] - a['y']))


def rel_bin(rel):
    for i, th in enumerate(REL_BINS):
        if rel < th:
            return i
    return len(REL_BINS)


def main(path):
    d = json.load(open(path, encoding='utf-8'))
    summary = d['summary']
    duration = summary['duration']
    g0 = summary['gameStartTs']

    # ---- 圈时间线 (只用 startClosing: 中心 + 起始半径) ----
    stages = {}
    for r in summary['ringPhases']:
        if r['type'] == 'startClosing':
            stages[r['stage']] = {
                'center': (r['center']['x'], r['center']['y']),
                'radius': r['startRadius'],
                'start': r['timestamp'] - g0,
            }
    order = sorted(stages.keys())
    starts = [stages[k]['start'] for k in order]
    centers = [stages[k]['center'] for k in order]
    radii = [stages[k]['radius'] for k in order]

    def active_phase(t):
        for i in range(len(starts) - 1, -1, -1):
            if t >= starts[i]:
                return i
        return 0

    # ---- 队伍: 淘汰时间(最后一名队员轨迹结束), 名次, 轨迹 ----
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
        teams[tid] = {'players': players, 'elim': max_last}

    for st in summary['teams']:
        tid = st['teamId']
        if tid in teams:
            teams[tid]['rank'] = st['placement']

    # ---- 击杀 每队每桶 (playerKilled.actor = 击杀者) ----
    team_kills = defaultdict(lambda: defaultdict(int))
    for e in d['events']:
        if e.get('category') == 'playerKilled':
            team_kills[e['actor']['teamId']][int(e['tsGame'] // BUCKET)] += 1

    # ---- 逐桶收集 (状态, 击杀, 是否淘汰, 下一状态) ----
    n_buckets = int(duration // BUCKET) + 1
    visits = defaultdict(int)
    k_sum = defaultdict(float)
    deaths = defaultdict(int)
    pl_sum = defaultdict(float)
    trans = defaultdict(lambda: defaultdict(int))

    for b in range(n_buckets):
        t, t_next = b * BUCKET, (b + 1) * BUCKET
        ph, ph_next = active_phase(t), active_phase(t_next)
        c, r = centers[ph], radii[ph]
        c_next, r_next = centers[ph_next], radii[ph_next]

        for tid, team in teams.items():
            if team['elim'] < t:
                continue
            # 本队存活队员在 t 的位置
            alive = []
            for pl in team['players']:
                if pl['ts'][0] <= t <= pl['ts'][-1]:
                    alive.append(pos_at(pl['pts'], pl['ts'], t))
            if not alive:
                continue
            anchor = team_anchor(alive)
            s = (ph, rel_bin(dist(anchor, c) / r))
            visits[s] += 1
            k_sum[s] += team_kills[tid][b]

            if team['elim'] < t_next:
                # 本桶被淘汰 -> 终态
                deaths[s] += 1
                pl_sum[s] += PLACEMENT_PTS.get(team.get('rank', 20), 0)
            else:
                # 活到下一桶 -> 转移
                alive_next = []
                for pl in team['players']:
                    if pl['ts'][0] <= t_next <= pl['ts'][-1]:
                        alive_next.append(pos_at(pl['pts'], pl['ts'], t_next))
                if not alive_next:
                    continue
                a_next = team_anchor(alive_next)
                s_next = (ph_next, rel_bin(dist(a_next, c_next) / r_next))
                trans[s][s_next] += 1

    # ---- 估计 k, h, pl, P ----
    states = sorted(visits.keys())
    k = {s: k_sum[s] / visits[s] for s in states}
    h = {s: deaths[s] / visits[s] for s in states}
    pl = {s: (pl_sum[s] / deaths[s] if deaths[s] else 0.0) for s in states}
    P = {}
    for s in states:
        tot = sum(trans[s].values())
        P[s] = {s2: trans[s][s2] / tot for s2 in trans[s]} if tot else {}

    # ---- value iteration ----
    V = {s: 0.0 for s in states}
    it, delta = 0, float('inf')
    for it in range(200):
        Vn, delta = {}, 0.0
        for s in states:
            future = sum(P[s].get(s2, 0) * V[s2] for s2 in P[s]) if P[s] else 0.0
            Vn[s] = k[s] + GAMMA * ((1 - h[s]) * future + h[s] * pl[s])
            delta = max(delta, abs(Vn[s] - V[s]))
        V = Vn
        if delta < 1e-6:
            break

    # ---- 输出 ----
    print(f"比赛: {summary['headerDisplayName']}")
    print(f"状态数 {len(states)} | 迭代 {it} 次收敛 (delta={delta:.2e}) | 圈时间线(tsGame): {[int(x) for x in starts]}")
    print(f"\n{'圈':<3}{'位置':<4}{'样本':>5}{'击杀k':>7}{'淘汰率h':>8}{'终态pl':>7}{'V':>8}")
    for ph in range(6):
        for rb in range(4):
            s = (ph, rb)
            if s not in visits:
                continue
            print(f"{ph+1:<3}{REL_NAMES[rb]:<5}{visits[s]:>5}{k[s]:>7.3f}{h[s]:>8.3f}{pl[s]:>7.2f}{V[s]:>8.3f}")

    print("\n=== sanity: V 是否 中心>中内>圈边>圈外 ? ===")
    for ph in range(6):
        vals = [V.get((ph, rb), float('nan')) for rb in range(4)]
        mono = all(vals[i] >= vals[i + 1] for i in range(3))
        print(f"圈{ph+1}: " + " | ".join(f"{REL_NAMES[i]}={v:.3f}" for i, v in enumerate(vals))
              + ("   ✓单调递减" if mono else "   ✗"))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/mnt/j/rank/replay/storm point/sp_apac-s_d2_g6_99c6fb41.json')
