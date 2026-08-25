#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_points_split.py — 点位价值拆成 击杀/排名 热力图 (叠加 Storm Point)

左: 后期点位击杀价值 kill_rate(p, 圈5)   (击杀跟当前圈有关, 这里取圈5)
右: 点位排名偏离 β_p_place               (圈无关, 红=好活点 蓝=废点)
点大小 = 覆盖 team-game 数。
"""
import json, math
from collections import defaultdict
import numpy as np
from PIL import Image

RATIO = 24.93
MAP = "map/storm point.png"
POS = "data/positions.csv"
LONG = "data/long_table.jsonl"
LATE_PHASE = 4   # 圈5 (阶段索引 4)
MIN_COV = 3

bk = np.load("data/beta_kill.npy")     # (NP, 6)
bp = np.load("data/beta_place.npy")

pts = []
for row in open(POS, encoding='utf-8'):
    if row.startswith('id'):
        continue
    p = row.strip().split(',')
    pts.append((float(p[1]), float(p[2])))
pts = np.array(pts)


# 覆盖度 = 每点位被多少 team-game 踩过 (与 points_split.py 同口径)
def extract_stays(rws, R=150, T=6, TMIN=120):
    stays = []
    i, n = 0, len(rws)
    while i < n:
        j, ps = i, []
        while j < n:
            ps.append((rws[j]['ax'], rws[j]['ay']))
            cx = sum(p[0] for p in ps) / len(ps)
            cy = sum(p[1] for p in ps) / len(ps)
            if all(math.hypot(p[0] - cx, p[1] - cy) <= R for p in ps):
                j += 1
            else:
                ps.pop()
                break
        if len(ps) >= T and rws[i]['t'] >= TMIN:
            stays.append((sum(p[0] for p in ps) / len(ps),
                          sum(p[1] for p in ps) / len(ps), len(ps)))
        i = max(i + 1, j)
    return stays


tg = defaultdict(list)
for line in open(LONG, encoding='utf-8'):
    r = json.loads(line)
    tg[(r['game'], r['team'])].append(r)

cov = np.zeros(len(bk), int)
for (gid, tid), rws in tg.items():
    rws = sorted(rws, key=lambda r: r['t'])
    for (cx, cy, nb) in extract_stays(rws):
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        cov[int(np.argmin(d))] += 1


def to_img(x, y):
    return (x / RATIO + 2048, -y / RATIO + 2048)


img = Image.open(MAP).convert("RGB")

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

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
ix = np.array([to_img(x, y)[0] for x, y in pts])
iy = np.array([to_img(x, y)[1] for x, y in pts])
mask = cov >= MIN_COV
sizes = np.clip(np.sqrt(cov) * 8, 16, 180)

# 左: 后期击杀价值 (圈5) — 非负, 用顺序色
ax = axes[0]
Vk = bk[:, LATE_PHASE]
vmax_k = Vk[mask].max()
ax.imshow(img, extent=[0, 4096, 4096, 0], zorder=0)
sc = ax.scatter(ix[mask], iy[mask], c=Vk[mask], cmap='YlOrRd', s=sizes[mask],
                edgecolors='#222', linewidths=0.3, alpha=0.9, zorder=3,
                vmin=0, vmax=vmax_k)
cb = plt.colorbar(sc, ax=ax, shrink=0.8)
cb.set_label('圈5 击杀价值', fontsize=10)
ax.set_title('后期点位击杀价值 kill_rate(p, 圈5)', fontsize=12)
ax.set_axis_off()
ax.set_xlim(0, 4096); ax.set_ylim(4096, 0)

# 右: 排名偏离 β_p_place — 有正有负, 用发散色
ax = axes[1]
vmax_p = np.abs(bp[mask]).max()
ax.imshow(img, extent=[0, 4096, 4096, 0], zorder=0)
sc = ax.scatter(ix[mask], iy[mask], c=bp[mask], cmap='RdBu_r', s=sizes[mask],
                edgecolors='#222', linewidths=0.3, alpha=0.9, zorder=3,
                vmin=-vmax_p, vmax=vmax_p)
cb = plt.colorbar(sc, ax=ax, shrink=0.8)
cb.set_label('点位排名偏离 β_p_place', fontsize=10)
ax.set_title('点位排名偏离 β_p_place (圈无关)', fontsize=12)
ax.set_axis_off()
ax.set_xlim(0, 4096); ax.set_ylim(4096, 0)

fig.suptitle('点位价值分解: 击杀(按圈, 取圈5) × 排名(圈无关)', fontsize=14, y=0.98)
plt.tight_layout()
out = "output/points_split.png"
plt.savefig(out, dpi=110, bbox_inches='tight')
print("已写", out)
