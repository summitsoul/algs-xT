#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_trajectory_demo.py — 生成「队伍 xT 轨迹」动态网页 demo (自包含 HTML)

左: Storm Point 地图 + 时间轴播放/拖动, 圈随游戏时间实时收缩漂移,
    点位按当前圈实时着色 xT, 吃鸡队(或选中队)路径逐帧推进, 20 队实时位置。
右: 20 队 xT 轨迹线图(带 now 游标) + 吃鸡队 击杀/排名 分解 + 实时读数 + 排名表。

xT(p, 圈) = xT_kill(p, 圈) + xT_place(p, 圈); kill=该点该圈每秒击杀数(斜率),
沿轨迹积分得累计击杀(单调递增, 换点只改斜率不清零); place=β_p_place+φ_place
(非折现, 毒里不再硬记0, 非负)。数据全部内联。
"""
import json, math, bisect, base64, io, os, re, sys
from collections import defaultdict
import numpy as np
from PIL import Image

REL_BINS = [0.3, 0.7, 1.0, 1.15, 1.6]
BUCKET = 5
VEL_WINDOW = 15  # 速度窗口(秒): 判"踩点/守家"看过去15s位移
STAY_VEL = 400   # 位移 <400 单位 => 算踩点; 否则算出去动/转点
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
pp = {(int(a), int(b)): v for a, b, v in
      (k.split(',') + [v] for k, v in phi['phi_place'].items())}
bk = np.load("data/beta_kill.npy")     # 击杀斜率 = 每秒击杀数
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


MIN_ZONE_R = 500.0  # 新圈(安全区)半径 < 此值视为无安全区(最后缩到一点), rel 退回毒环、β_p 不衰减
R0 = 31000.0        # 圈1 新圈(安全区)半径, w_stage 归一化基准
GAP_THRESHOLD = 60.0  # 尾端孤点判定: 末段 gap 超此值 => 死亡/断线后补记的清理快照, 丢弃


def zone_rel(a, c1, r1, c_ring, r_ring_start):
    """新圈基准 rel(圈内/圈外判据)。常规圈(r1>=MIN_ZONE_R)用新圈(安全区)中心/半径;
    最后圈(r1<MIN_ZONE_R, 无安全区)退回毒环中心 + 毒环起始半径(冻结在开始收缩时的大小,
    不随收缩缩小——否则分母→1、rel 爆炸、所有人挤进远圈外)。"""
    if r1 >= MIN_ZONE_R:
        return dist(a, c1) / r1
    return dist(a, c_ring) / max(r_ring_start, 1.0)


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


def alive_at(pl, t):
    """t 时刻该玩家是否在场。用 playerKilled 当死亡真相: 被击杀后、直到轨迹恢复(复活)之间
    算不在场, 排除出 alive 集——复活空窗期按剩下的人记锚点, 不把死人插值进去。"""
    ts = pl['ts']
    if not ts or t < ts[0] or t > ts[-1]:
        return False
    for D in pl['deaths']:
        if D >= t:
            break
        i = bisect.bisect_right(ts, D)
        if i >= len(ts) or ts[i] > t:
            return False  # 死后到 t 尚未复活
    return True


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


# 每玩家死亡时刻 (playerKilled.target -> 玩家名 -> [tsGame, ...] 升序)
deaths = defaultdict(list)
for e in d['events']:
    if e.get('category') == 'playerKilled':
        tg = e.get('target', {})
        nm = tg.get('playerName') or tg.get('name')
        if nm:
            deaths[nm].append(e['tsGame'])
for nm in deaths:
    deaths[nm].sort()

teams = {}
for t in d['pathing']['teams']:
    tid = t['teamId']
    players, max_last = [], 0
    for pl in t['players']:
        pts = pl.get('points') or []
        if not pts:
            continue
        pts.sort(key=lambda p: p['tsGame'])
        # 丢弃「死亡/断线后补记的尾端孤点」: 末段 gap 过大说明前面已死, 此点是清理快照(非真实位置)。
        while len(pts) >= 2 and pts[-1]['tsGame'] - pts[-2]['tsGame'] > GAP_THRESHOLD:
            pts.pop()
        players.append({'ts': [p['tsGame'] for p in pts], 'pts': pts,
                        'deaths': deaths.get(pl.get('playerName'), [])})
        max_last = max(max_last, pts[-1]['tsGame'])
    teams[tid] = {'players': players, 'elim': min(max_last, dur)}
for st in s['teams']:
    tid = st['teamId']
    if tid in teams:
        teams[tid]['rank'] = st['placement']
        teams[tid]['name'] = st.get('teamName', f"T{tid}")


def xT_of(x, y, t):
    ph, (cx, cy), r = ring_at(t)
    c1, r1 = stages[ph]['c1'], stages[ph]['r1']    # 新圈(安全区)中心/半径
    rel = zone_rel((x, y), c1, r1, (cx, cy), stages[ph]['r0'])
    j = int(np.argmin(np.hypot(pos_arr[:, 0] - x, pos_arr[:, 1] - y)))
    slope = bk[j][ph]                              # 击杀斜率: 每秒击杀数(按当前圈分档)
    xp = max(0.0, zone_w(rel, r1) * stage_w(r1) * bp[j] + pp[(ph, rel_bin(rel))])  # 排名: β_p 圈外衰减 + 圈阶段权重 + 非折现 + 非负
    return slope, xp, rel, ph


# ---------- 逐队轨迹 + 路径 ----------
series_map, path_map = {}, {}
for tid, team in teams.items():
    n_buckets = int(dur // BUCKET) + 1
    ts, pt = [], []
    xk_acc = 0.0
    for b in range(n_buckets):
        t = b * BUCKET
        if team['elim'] < t:
            break
        alive = [pl for pl in team['players'] if alive_at(pl, t)]
        if not alive:
            continue
        a = team_anchor(alive, t)
        slope, xp, rel, ph = xT_of(a[0], a[1], t)
        xk_acc += slope * BUCKET                 # 累计击杀 (单调递增, 换点只改斜率)
        ts.append([t, round(xk_acc, 3), round(xp, 3), round(xk_acc + xp, 3)])
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
vmax = {'total': 0.0, 'kill': float(bk.max()), 'place': 0.0}
for m in order:
    c1, r1 = stages[m]['c1'], stages[m]['r1']
    for i in range(NP):
        rel = zone_rel((pos_arr[i, 0], pos_arr[i, 1]), c1, r1, c1, stages[m]['r0'])
        xp = max(0.0, zone_w(rel, r1) * stage_w(r1) * bp[i] + pp[(m, rel_bin(rel))])
        vmax['place'] = max(vmax['place'], xp)
        vmax['total'] = max(vmax['total'], bk[i][m] + xp)

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
               'bk': [[round(v, 3) for v in bk[i]] for i in range(NP)],
               'bp': [round(v, 3) for v in bp]},
    'phi': {'kill': {f"{a},{b}": 0.0 for (a, b), v in pp.items()},
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