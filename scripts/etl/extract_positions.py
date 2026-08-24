#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_positions.py — 提取所有"点位"(可踩/会停留的位置)

思路(按用户要求, 先暂停 xT 建模, 先把点位全跑出来):
  点位 = 一支队伍在一个地方"呆着不动"(不是路过转移)的位置。
  1) 停留段检测: 在 R 世界单位内维持 T 个时间桶(=T*5 秒) -> 一个 stay 段, 记其质心。
  2) 去重聚类: 不同 stay 段若质心距离 < eps 则属同一个点位 (DBSCAN)。
  3) 每个点位输出: 质心世界坐标、总停留秒数、停留段数、覆盖队数、覆盖场数。

输出:
  data/positions.csv   点位列表 (id, cx, cy, dwell_sec, n_stays, n_teams, n_games)
  output/positions.png 地图叠加标记 (大小=总停留时长)
"""
import json, math, sys, glob, os
from collections import defaultdict
import numpy as np
from sklearn.cluster import DBSCAN
from PIL import Image, ImageDraw

RATIO = 24.93
R = 150          # 停留半径(世界单位), 桶内所有点相对质心不超过此值
T = 6            # 最少停留桶数 (6*5=30 秒)
TMIN = 120       # 停留段起始时间下限(秒): 排除落地/搜刮阶段, 只看进圈占点
EPS = 250        # 点位聚类半径(世界单位)
MIN_DWELL = 60   # 展示/输出时, 点位最少总停留秒数
LONG = "data/long_table.jsonl"
OUT_CSV = "data/positions.csv"
OUT_PNG = "output/positions.png"
MAP_PNG = "map/storm point.png"
SIZE = 4096


def extract_stays(rws, R=R, T=T):
    """单队单场 -> 停留段 [(cx, cy, n_buckets)]"""
    stays = []
    i, n = 0, len(rws)
    while i < n:
        j, pts = i, []
        while j < n:
            pts.append((rws[j]['ax'], rws[j]['ay']))
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            if all(math.hypot(p[0] - cx, p[1] - cy) <= R for p in pts):
                j += 1
            else:
                pts.pop()
                break
        if len(pts) >= T and rws[i]['t'] >= TMIN:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            stays.append((cx, cy, len(pts)))
        i = max(i + 1, j)
    return stays


def main():
    eps = float(sys.argv[1]) if len(sys.argv) > 1 else EPS
    rows = [json.loads(l) for l in open(LONG, encoding='utf-8')]
    traj = defaultdict(list)
    for r in rows:
        traj[(r['game'], r['team'])].append(r)
    for k in traj:
        traj[k].sort(key=lambda r: r['t'])

    # 1) 停留段
    stays = []  # (cx, cy, buckets, game, team)
    for (gid, tid), rws in traj.items():
        for (cx, cy, nb) in extract_stays(rws):
            stays.append((cx, cy, nb, gid, tid))

    # 2) DBSCAN 聚类成点位
    X = np.array([[s[0], s[1]] for s in stays])
    db = DBSCAN(eps=eps, min_samples=1).fit(X)
    labels = db.labels_

    clusters = defaultdict(lambda: {'w': 0.0, 'wx': 0.0, 'wy': 0.0, 'n_stays': 0,
                                    'teams': set(), 'games': set(), 'buckets': 0})
    for (cx, cy, nb, gid, tid), lab in zip(stays, labels):
        c = clusters[lab]
        w = nb  # 用停留桶数加权质心
        c['w'] += w
        c['wx'] += cx * w
        c['wy'] += cy * w
        c['n_stays'] += 1
        c['buckets'] += nb
        c['teams'].add((gid, tid))
        c['games'].add(gid)

    # 3) 输出
    pos = []
    for lab, c in clusters.items():
        if lab == -1:
            continue
        cx = c['wx'] / c['w']
        cy = c['wy'] / c['w']
        pos.append({
            'id': len(pos),
            'cx': cx, 'cy': cy,
            'dwell_sec': c['buckets'] * 5,
            'n_stays': c['n_stays'],
            'n_teams': len(c['teams']),
            'n_games': len(c['games']),
        })

    n_all = len(pos)
    pos.sort(key=lambda p: -p['dwell_sec'])
    pos = [p for p in pos if p['dwell_sec'] >= MIN_DWELL]

    with open(OUT_CSV, 'w', encoding='utf-8') as f:
        f.write("id,cx,cy,dwell_sec,n_stays,n_teams,n_games\n")
        for p in pos:
            f.write(f"{p['id']},{p['cx']:.1f},{p['cy']:.1f},{p['dwell_sec']:.0f},"
                    f"{p['n_stays']},{p['n_teams']},{p['n_games']}\n")

    print(f"停留段总数 {len(stays)} | eps={eps:.0f} -> 原始点位 {n_all} 个, "
          f"过滤后(停留>={MIN_DWELL}s) {len(pos)} 个")
    print(f"停留时长分布: 中位 {np.median([p['dwell_sec'] for p in pos]):.0f}s, "
          f"最大 {max(p['dwell_sec'] for p in pos):.0f}s")
    print(f"点位覆盖队数: 中位 {np.median([p['n_teams'] for p in pos]):.0f}, "
          f"最大 {max(p['n_teams'] for p in pos):.0f}")
    print(f"已写 {OUT_CSV}")

    # 4) 决赛圈: 每场最后一个 finishedClosing 圆 (中心 + 收缩后半径)
    finals = []  # (cx, cy, radius, region)
    for path in sorted(glob.glob("replay/storm point/*.json")):
        d = json.load(open(path, encoding='utf-8'))
        s = d['summary']
        sc, fc = {}, {}
        for r in s['ringPhases']:
            if r['type'] == 'startClosing':
                sc[r['stage']] = r
            elif r['type'] == 'finishedClosing':
                fc[r['stage']] = r
        last = sorted(sc.keys())[-1]
        a = sc[last]
        b = fc.get(last)
        # 决赛圈半径: ring6 的 endRadius 是 1(退化的点, 比赛未走到), 取起始 startRadius
        # 5段局(无 ring6) endRadius=2000 是真实收完圈 -> 取 endRadius
        rad = a['endRadius'] if a['endRadius'] > 100 else a['startRadius']
        cc = b['center'] if b else a['center']
        cx, cy = cc['x'], cc['y']
        region = os.path.basename(path).split('_')[1]
        finals.append((cx, cy, rad, region))

    # 5) 渲染: 先画决赛圈(细轮廓, 按地区着色), 再画点位标记
    img = Image.open(MAP_PNG).convert('RGBA')
    dr = ImageDraw.Draw(img)
    ring_col = {'na': (0, 210, 255, 170), 'apac-s': (255, 130, 0, 170)}  # NA青 / APAC-S橙
    for (cx, cy, rad, region) in finals:
        ix, iy = cx / RATIO + 2048, -cy / RATIO + 2048
        r = rad / RATIO
        dr.ellipse([ix - r, iy - r, ix + r, iy + r], outline=ring_col.get(region, (255, 255, 255, 170)),
                   width=3)
    vmax = max(p['dwell_sec'] for p in pos) if pos else 1.0
    from matplotlib import colormaps as _cm
    for p in pos:
        ix = p['cx'] / RATIO + 2048
        iy = -p['cy'] / RATIO + 2048
        t = math.sqrt(p['dwell_sec'] / vmax)   # 面积按时长开方
        rad = 6 + 34 * t
        col = _cm['YlOrRd'](0.25 + 0.7 * t)   # 单色序贯(浅黄->深红), 越红越久
        rgba = tuple(int(v * 255) for v in col[:3]) + (220,)
        dr.ellipse([ix - rad, iy - rad, ix + rad, iy + rad], fill=rgba,
                   outline=(0, 0, 0, 180), width=2)

    # 图例: 点位色带 + 决赛圈颜色
    from PIL import ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    except Exception:
        font = ImageFont.load_default()
    lx0, ly0 = 56, SIZE - 190
    dr.rectangle([lx0 - 18, ly0 - 18, lx0 + 560, ly0 + 172], fill=(255, 255, 255, 205))
    bw = 400
    for i in range(bw):
        t = i / (bw - 1)
        col = _cm['YlOrRd'](0.25 + 0.7 * t)
        dr.line([lx0 + i, ly0, lx0 + i, ly0 + 30], fill=tuple(int(v * 255) for v in col[:3]) + (255,))
    dr.text((lx0, ly0 + 40), "停留短 60s", fill=(30, 30, 30, 255), font=font)
    dr.text((lx0 + bw - 168, ly0 + 40), "停留长 2665s", fill=(30, 30, 30, 255), font=font)
    dr.ellipse([lx0 + 2, ly0 + 86, lx0 + 34, ly0 + 118], outline=(0, 210, 255, 255), width=3)
    dr.text((lx0 + 44, ly0 + 88), "NA 决赛圈", fill=(30, 30, 30, 255), font=font)
    dr.ellipse([lx0 + 200, ly0 + 86, lx0 + 232, ly0 + 118], outline=(255, 130, 0, 255), width=3)
    dr.text((lx0 + 242, ly0 + 88), "APAC-S 决赛圈", fill=(30, 30, 30, 255), font=font)

    img.save(OUT_PNG)
    from collections import Counter
    print(f"决赛圈 {len(finals)} 个已叠加 (NA {sum(1 for f in finals if f[3]=='na')} 场 / "
          f"APAC-S {sum(1 for f in finals if f[3]=='apac-s')} 场)")
    print(f"地图叠加 {OUT_PNG}")


if __name__ == '__main__':
    main()
