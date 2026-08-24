#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_trajectory_demo.py — 生成「队伍 xT 轨迹」动态网页 demo (自包含 HTML)

左: Storm Point 地图 + 时间轴播放/拖动, 圈随游戏时间实时收缩漂移,
    点位按当前圈实时着色 xT, 吃鸡队(或选中队)路径逐帧推进, 20 队实时位置。
右: 20 队 xT 轨迹线图(带 now 游标) + 吃鸡队 击杀/排名 分解 + 实时读数 + 排名表。

xT(p, 圈) = β_p + φ_m(rel_bin), 毒里清零。数据全部内联, 无外部依赖。
"""
import json, math, bisect, base64, io, os, re, sys
import numpy as np
from PIL import Image

REL_BINS = [0.3, 0.7, 1.0, 1.15, 1.6]
BUCKET = 5
RATIO = 24.93
GAME = sys.argv[1] if len(sys.argv) > 1 else "replay/storm point/sp_global_d__g9_29ce1521.json"
MAP_PNG = "map/storm point.png"
OUT = sys.argv[2] if len(sys.argv) > 2 else "output/demo_" + os.path.basename(GAME).replace(".json", "") + ".html"


def game_label(path):
    b = os.path.basename(path).replace(".json", "")
    m = re.match(r"sp_(.+?)_d(.*?)_(g\d+)_", b)
    if not m:
        return b
    region = {"apac-s": "亚太南", "apac-n": "亚太北", "global": "世界赛",
              "na": "北美", "emea": "欧洲"}.get(m.group(1), m.group(1))
    day = m.group(2).strip("_")
    day_s = f" Day{day}" if day else ""
    return f"{region}{day_s} {m.group(3).replace('g', 'Game#')}"
MAP_SIZE = 2048

# ---------- 模型 ----------
phi = json.load(open("data/phi.json", encoding='utf-8'))
pk = {(int(a), int(b)): v for a, b, v in
      (k.split(',') + [v] for k, v in phi['phi_kill'].items())}
pp = {(int(a), int(b)): v for a, b, v in
      (k.split(',') + [v] for k, v in phi['phi_place'].items())}
bk = np.load("data/beta_kill.npy")
bp = np.load("data/beta_place.npy")
pos_arr = np.load("data/points_pos.npy")
NP = len(bk)


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


# ---------- 解析比赛 ----------
d = json.load(open(GAME, encoding='utf-8'))
s = d['summary']
dur, g0 = s['duration'], s['gameStartTs']

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
    ph, (cx, cy), r = ring_at(t)
    rel = math.hypot(x - cx, y - cy) / r
    if rel > 1.0:
        return 0.0, 0.0, rel, ph
    j = int(np.argmin(np.hypot(pos_arr[:, 0] - x, pos_arr[:, 1] - y)))
    b = rel_bin(rel)
    return bk[j] + pk[(ph, b)], bp[j] + pp[(ph, b)], rel, ph


# ---------- 逐队轨迹 + 路径 ----------
series_map, path_map = {}, {}
for tid, team in teams.items():
    ts, pt = [], []
    n_buckets = int(dur // BUCKET) + 1
    for b in range(n_buckets):
        t = b * BUCKET
        if team['elim'] < t:
            break
        alive = [pos_at(pl['pts'], pl['ts'], t) for pl in team['players']
                 if pl['ts'][0] <= t <= pl['ts'][-1]]
        if not alive:
            continue
        a = team_anchor(alive)
        xk, xp, rel, ph = xT_of(a[0], a[1], t)
        ts.append([t, round(xk, 3), round(xp, 3), round(xk + xp, 3)])
        pt.append([t, round(a[0], 1), round(a[1], 1)])
    series_map[tid] = ts
    path_map[tid] = pt

winner = [tid for tid, t in teams.items() if t.get('rank') == 1][0]
wname = teams[winner]['name']

# ---------- 后期排名表 ----------
# Apex 圈从 1 开始: 数据 stage 0→圈1, stage 2→圈3。后期 = 圈3+ = stage[2] 起。
LATE_T0 = stages[order[2]]['t0']
late_rows = []
for tid, t in sorted(teams.items(), key=lambda kv: kv[1].get('rank', 99)):
    ts = series_map.get(tid, [])
    vals = [r[3] for r in ts if r[0] >= LATE_T0]
    if vals:
        late_rows.append({
            'rank': t.get('rank', 99), 'name': t.get('name', f"T{tid}"),
            'mean': round(float(np.mean(vals)), 2),
            'peak': round(float(max(vals)), 2),
        })

# ---------- 地图 base64 ----------
img = Image.open(MAP_PNG).convert('RGB').resize((MAP_SIZE, MAP_SIZE), Image.LANCZOS)
buf = io.BytesIO()
img.save(buf, 'JPEG', quality=70)
map_b64 = base64.b64encode(buf.getvalue()).decode()

# ---------- 色阶 vmax (各指标在所有完成圈下的点位 xT 上限) ----------
vmax = {'total': 0.0, 'kill': 0.0, 'place': 0.0}
for m in order:
    cx, cy = stages[m]['c1']
    r = stages[m]['r1']
    for i in range(NP):
        rel = math.hypot(pos_arr[i, 0] - cx, pos_arr[i, 1] - cy) / r
        if rel <= 1.0:
            b = rel_bin(rel)
            vmax['kill'] = max(vmax['kill'], bk[i] + pk[(m, b)])
            vmax['place'] = max(vmax['place'], bp[i] + pp[(m, b)])
            vmax['total'] = max(vmax['total'], bk[i] + bp[i] + pk[(m, b)] + pp[(m, b)])

# ---------- 组装数据 ----------
DATA = {
    'game': game_label(GAME),
    'winner': wname,
    'dur': dur,
    'vmax': {k: round(vmax[k], 2) for k in vmax},
    'relbins': REL_BINS,
    'rings': [{'t0': stages[m]['t0'], 't1': stages[m]['t1'],
               'c0': [stages[m]['c0'][0], stages[m]['c0'][1]],
               'c1': [stages[m]['c1'][0], stages[m]['c1'][1]],
               'r0': stages[m]['r0'], 'r1': stages[m]['r1']} for m in order],
    'points': {'x': [round(p[0], 1) for p in pos_arr],
               'y': [round(p[1], 1) for p in pos_arr],
               'bk': [round(v, 3) for v in bk],
               'bp': [round(v, 3) for v in bp]},
    'phi': {'kill': {f"{a},{b}": round(v, 4) for (a, b), v in pk.items()},
            'place': {f"{a},{b}": round(v, 4) for (a, b), v in pp.items()}},
    'teams': [{'name': teams[tid].get('name', f"T{tid}"),
               'rank': teams[tid].get('rank', 99),
               'series': series_map.get(tid, []),
               'path': path_map.get(tid, [])}
              for tid in sorted(teams, key=lambda t: teams[t].get('rank', 99))],
    'late': late_rows,
}

all_xT = [r[3] for t in series_map.values() for r in t]
YMAX = math.ceil(max(all_xT) * 1.15)

tmpl = open("scripts/viz/trajectory_template.html", encoding='utf-8').read()
html = tmpl.replace('__DATA__', json.dumps(DATA, ensure_ascii=False)) \
           .replace('__MAP__', map_b64) \
           .replace('__YMAX__', str(YMAX)) \
           .replace('__DUR__', str(dur)) \
           .replace('__GAME__', game_label(GAME)) \
           .replace('__WINNER__', wname)
open(OUT, 'w', encoding='utf-8').write(html)
print(f"已写 {OUT} ({len(html)//1024} KB, 点位 {NP}, YMAX {YMAX}, vmax {DATA['vmax']})")