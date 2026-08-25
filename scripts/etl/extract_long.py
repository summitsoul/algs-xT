#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_long.py — 跨场提取: 22 场 replay JSON -> 长表 data/long_table.jsonl

每行 = 一个 (game, team, 时间桶) 观测。状态由 model 脚本统一拼装, 这里只出原始量:
  game, region, team, team_name, b, t, phase, rel, rel_bin, ax, ay (锚点世界坐标),
  kills (本桶击杀分), died (本桶是否被淘汰), placement_pts (被淘汰时的排名分),
  next_phase, next_rel_bin, next_ax, next_ay (存活转移用)

绝对位置(踩的位置)的网格分桶不在本脚本做 —— 需要全局包围盒, 由 model 脚本按 meta 统一切。
"""
import json, glob, os, math, bisect
from collections import defaultdict

PLACEMENT_PTS = {1: 12, 2: 9, 3: 7, 4: 5, 5: 4, 6: 3, 7: 3,
                 8: 2, 9: 2, 10: 2, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1}
BUCKET = 5
REL_BINS = [0.3, 0.7, 1.0, 1.15, 1.6]
REL_NAMES = ['圈心', '圈内', '圈内边缘', '圈外贴边', '圈外', '远圈外']
MIN_ZONE_R = 500.0  # 新圈(安全区)半径 < 此值视为无安全区(最后缩到一点), rel 退回毒环、β_p 不衰减
R0 = 31000.0        # 圈1 新圈(安全区)半径, w_stage 归一化基准
VEL_WINDOW = 15    # 速度窗口(秒): 判"踩点/守家"看过去15s位移
STAY_VEL = 400     # 位移 <400 单位 => 算踩点; 否则算出去动/转点
GLOB = "replay/storm point/*.json"
OUT = "data/long_table.jsonl"
META = "data/long_table.meta.json"


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def zone_rel(a, c1, r1, c_ring, r_ring):
    """新圈基准 rel(圈内/圈外判据)。常规圈(r1>=MIN_ZONE_R)用新圈(安全区)中心/半径;
    最后圈(r1≈0, 无安全区)退回正在缩的毒环。"""
    if r1 >= MIN_ZONE_R:
        return dist(a, c1) / r1
    return dist(a, c_ring) / max(r_ring, 1.0)


def zone_w(rel, r1):
    """β_p 圈外平滑衰减权重。最后圈不衰减=1.0; 常规圈 rel<=1 全额, rel 1→1.6 线性降到0,
    rel>=1.6(远圈外)归零。理由: 点位固有排名价值只在点位被圈覆盖时才兑现, 圈外站不住。"""
    if r1 < MIN_ZONE_R:
        return 1.0
    return max(0.0, min(1.0, (1.6 - rel) / 0.6))


def stage_w(r1):
    """β_p 圈阶段权重: 圈越大(越早)点位固有价值兑现越低, 缩到决赛圈才全额。
    w_stage = 1 - r1/R0 → 圈1=0, 圈2≈0.52, 圈3≈0.74, 圈4≈0.87, 圈5≈0.94, 圈6≈1.0。
    理由: 点位固有排名价值只在圈缩到足以争抢该点时才兑现, 早期圈"在圈内"形同虚设。"""
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


def rel_bin(rel):
    for i, th in enumerate(REL_BINS):
        if rel < th:
            return i
    return len(REL_BINS)


def extract_game(path):
    d = json.load(open(path, encoding='utf-8'))
    s = d['summary']
    dur, g0 = s['duration'], s['gameStartTs']
    name = os.path.basename(path)
    region = name.split('_')[1]

    # 圈时间线 (startClosing -> finishedClosing, 中心/半径随时间漂移+收缩)
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
        """返回 (phase, 当前中心, 当前半径); 圈收缩期线性插值, 收缩间隙停在终点"""
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

    # 队伍: 淘汰时间 + 名次 + 队名 + 轨迹
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

    # 击杀 每队每桶 (playerKilled.actor = 击杀者)
    tk = defaultdict(lambda: defaultdict(int))
    for e in d['events']:
        if e.get('category') == 'playerKilled':
            tk[e['actor']['teamId']][int(e['tsGame'] // BUCKET)] += 1

    # 逐桶
    rows = []
    n_buckets = int(dur // BUCKET) + 1
    for b in range(n_buckets):
        t, t_next = b * BUCKET, (b + 1) * BUCKET
        ph, c, r = ring_at(t)
        ph_next, c_next, r_next = ring_at(t_next)
        c1, r1 = stages[ph]['c1'], stages[ph]['r1']          # 新圈(安全区)中心/半径
        c1n, r1n = stages[ph_next]['c1'], stages[ph_next]['r1']

        for tid, team in teams.items():
            if team['elim'] < t:
                continue
            alive = [pl for pl in team['players']
                     if pl['ts'][0] <= t <= pl['ts'][-1]]
            if not alive:
                continue
            a = team_anchor(alive, t)
            rel = zone_rel(a, c1, r1, c, r)
            row = {'game': name, 'region': region, 'team': tid,
                   'team_name': team.get('name', f"T{tid}"),
                   'b': b, 't': t, 'phase': ph, 'rel': rel,
                   'rel_bin': rel_bin(rel), 'w_zone': zone_w(rel, r1), 'w_stage': stage_w(r1),
                   'ax': a[0], 'ay': a[1], 'kills': tk[tid][b],
                   'cx': c[0], 'cy': c[1], 'r': r}

            if team['elim'] < t_next:
                row['died'] = 1
                row['placement_pts'] = PLACEMENT_PTS.get(team.get('rank', 20), 0)
                rows.append(row)
                continue

            alive_next = [pl for pl in team['players']
                          if pl['ts'][0] <= t_next <= pl['ts'][-1]]
            if not alive_next:  # 理论上存活队不会发生(elim>=t_next 必有队员覆盖 t_next)
                continue
            an = team_anchor(alive_next, t_next)
            row['died'] = 0
            row['placement_pts'] = 0
            row['next_phase'] = ph_next
            row['next_rel_bin'] = rel_bin(zone_rel(an, c1n, r1n, c_next, r_next))
            row['next_ax'] = an[0]
            row['next_ay'] = an[1]
            row['next_cx'] = c_next[0]
            row['next_cy'] = c_next[1]
            row['next_r'] = r_next
            rows.append(row)

    return rows


def main():
    files = sorted(glob.glob(GLOB))
    all_rows = []
    for f in files:
        rows = extract_game(f)
        all_rows.extend(rows)
        print(f"  {os.path.basename(f):40} -> {len(rows)} rows")

    xs = [r['ax'] for r in all_rows]
    ys = [r['ay'] for r in all_rows]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    meta = {'n': len(all_rows), 'bounds': {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax},
            'bucket': BUCKET, 'rel_bins': REL_BINS, 'n_files': len(files)}
    json.dump(meta, open(META, 'w'))

    print(f"\n总行数 {len(all_rows)} | 锚点世界坐标包围盒 x=[{xmin:.0f},{xmax:.0f}] y=[{ymin:.0f},{ymax:.0f}]")
    print(f"长表已写 {OUT} | meta 已写 {META}")


if __name__ == '__main__':
    main()
