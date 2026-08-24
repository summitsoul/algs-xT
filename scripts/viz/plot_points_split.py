#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_points_split.py — 点位固有价值 β_p 拆成 击杀/排名 热力图 (叠加 Storm Point)

左: β_p_kill (点位固有击杀价值)   右: β_p_place (点位固有排名价值)
颜色: 红=正(好点), 蓝=负(废点), 白≈0; 点大小=覆盖 team-game 数。
这是定稿模型 xT = β_p + φ_m(rel_bin) 里的 β_p 项, 跨场不变的点位固有价值。
"""
import csv
import numpy as np
from PIL import Image

RATIO = 24.93
MAP = "map/storm point.png"
POS = "data/positions.csv"

bk = np.load("data/beta_kill.npy")
bp = np.load("data/beta_place.npy")
cov = np.load("data/points_kill_cov.npy") if False else None
# 覆盖度从 points_value.csv 读
cov = np.zeros(len(bk), int)
for row in csv.DictReader(open("data/points_value.csv", encoding='utf-8')):
    cov[int(row['id'])] = int(float(row['coverage']))

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
ix = np.array([to_img(x, y)[0] for x, y in pts])
iy = np.array([to_img(x, y)[1] for x, y in pts])
mask = cov >= 3
sizes = np.clip(np.sqrt(cov) * 8, 16, 180)

for ax, V, title, cbar_lab in [
    (axes[0], bk, '后期点位击杀价值 β_p_kill', '点位固有击杀价值'),
    (axes[1], bp, '后期点位排名价值 β_p_place', '点位固有排名价值'),
]:
    vmax = np.abs(V[mask]).max()
    ax.imshow(img, extent=[0, 4096, 4096, 0], zorder=0)
    sc = ax.scatter(ix[mask], iy[mask], c=V[mask], cmap='RdBu_r', s=sizes[mask],
                    edgecolors='#222', linewidths=0.3, alpha=0.9, zorder=3,
                    vmin=-vmax, vmax=vmax)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label(cbar_lab, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()
    ax.set_xlim(0, 4096); ax.set_ylim(4096, 0)

fig.suptitle('点位固有价值 β_p 分解 (红=好点, 蓝=废点, 与圈无关)', fontsize=14, y=0.98)
plt.tight_layout()
out = "output/points_split.png"
plt.savefig(out, dpi=110, bbox_inches='tight')
print("已写", out)