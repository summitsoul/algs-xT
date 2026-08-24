#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_late_point_value.py — 后期点位价值热力图 (叠加 Storm Point 底图)

左: 击杀价值 V_kill (期望未来击杀分)   右: 排名价值 V_place (期望未来排名分)
点大小 = 该点后期访问量(数据覆盖度), 颜色 = 价值(红/蓝 = 高)。
"""
import csv
import numpy as np
from PIL import Image

RATIO = 24.93
MAP = "map/storm point.png"
POS = "data/positions.csv"

V_kill = np.load("data/late_V_kill.npy")
V_place = np.load("data/late_V_place.npy")
V_total = np.load("data/late_V_total.npy")
visits = np.load("data/late_visits.npy")

pts = []
for row in csv.DictReader(open(POS, encoding='utf-8')):
    pts.append((float(row['cx']), float(row['cy'])))
pts = np.array(pts)


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
iy = np.array([to_img(x, y)[1] for x, y in pts])
ix = np.array([to_img(x, y)[0] for x, y in pts])
# 只画有覆盖的点
mask = visits >= 5
sizes = np.clip(np.sqrt(visits) * 6, 12, 160)

for ax, V, cmap, title, cbar_lab in [
    (axes[0], V_kill, 'Reds', '后期点位击杀价值 V_kill', '期望未来击杀分'),
    (axes[1], V_place, 'Blues', '后期点位排名价值 V_place', '期望未来排名分'),
]:
    ax.imshow(img, extent=[0, 4096, 4096, 0], zorder=0)
    sc = ax.scatter(ix[mask], iy[mask], c=V[mask], cmap=cmap, s=sizes[mask],
                    edgecolors='none', alpha=0.85, zorder=3, vmin=0)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label(cbar_lab, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()
    ax.set_xlim(0, 4096); ax.set_ylim(4096, 0)

fig.suptitle('后期(圈3+) 各点位价值 —— 指挥选点依据', fontsize=14, y=0.98)
plt.tight_layout()
out = "output/late_point_value.png"
plt.savefig(out, dpi=110, bbox_inches='tight')
print("已写", out)