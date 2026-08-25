#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coarse_blocks.py — 粗块合并预览 (单独, 不进模型)

把 451 个细点位按空间网格合并成「粗块」, 看看合并后长什么样:
  - 每个粗块 = 地图上一块 N×N 网格 (块大小 BLOCK 世界单位)
  - 颜色 = 块内总覆盖 (n_games 之和, 即这块被多少 team-game 踩过)
  - 数字 = 块内的细点位数

目的只是「看一眼」, 帮助判断要不要真的做粗块合并(降参/提高每块覆盖)。
输出 output/coarse_blocks.png。
"""
import csv
from collections import defaultdict
import numpy as np
from PIL import Image

RATIO = 24.93
MAP = "map/storm point.png"
POS = "data/positions.csv"
BLOCK = 6000.0   # 粗块边长 (世界单位)

pts, ids = [], []
cov = []
for row in csv.DictReader(open(POS, encoding='utf-8')):
    ids.append(int(row['id']))
    pts.append((float(row['cx']), float(row['cy'])))
    cov.append(int(float(row['n_games'])))
pts = np.array(pts)
cov = np.array(cov, float)

x0, y0 = pts[:, 0].min(), pts[:, 1].min()
x1, y1 = pts[:, 0].max(), pts[:, 1].max()
nx = int(np.ceil((x1 - x0) / BLOCK)) + 1
ny = int(np.ceil((y1 - y0) / BLOCK)) + 1

# 每个块: 总覆盖 + 点数
blocks = defaultdict(lambda: [0.0, 0])
for (cx, cy), c in zip(pts, cov):
    gx = int((cx - x0) // BLOCK)
    gy = int((cy - y0) // BLOCK)
    blocks[(gx, gy)][0] += c
    blocks[(gx, gy)][1] += 1

occupied = list(blocks.keys())
print(f"细点位 {len(pts)} → 粗块 {len(occupied)} 个 "
      f"(块边长 {BLOCK:.0f}, 网格 {nx}x{ny})")
covers = [b[0] for b in blocks.values()]
print(f"每块总覆盖: min={min(covers):.0f} max={max(covers):.0f} "
      f"中位={np.median(covers):.0f}")


def to_img(x, y):
    return (x / RATIO + 2048, -y / RATIO + 2048)


img = Image.open(MAP).convert("RGB")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
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

fig, ax = plt.subplots(figsize=(11, 10))
ax.imshow(img, extent=[0, 4096, 4096, 0], zorder=0)

vmax = max(covers)
for (gx, gy), (c, n) in blocks.items():
    wx0 = x0 + gx * BLOCK
    wy0 = y0 + gy * BLOCK
    ix0, iy0 = to_img(wx0, wy0)
    ix1, iy1 = to_img(wx0 + BLOCK, wy0 + BLOCK)
    left = min(ix0, ix1); bottom = min(iy0, iy1)
    w = abs(ix1 - ix0); h = abs(iy1 - iy0)
    # 颜色按覆盖, 透明底
    t = min(1.0, c / vmax)
    cmap = plt.get_cmap('YlOrRd')
    r, g, b, _ = cmap(0.25 + 0.7 * t)
    ax.add_patch(Rectangle((left, bottom), w, h, facecolor=(r, g, b, 0.55),
                           edgecolor='#111', linewidth=1.0, zorder=2))
    if n > 0:
        ax.text((left + ix1) / 2, (bottom + iy1) / 2, str(n),
                ha='center', va='center', fontsize=7, color='#111',
                zorder=3, weight='bold')

ax.set_xlim(0, 4096); ax.set_ylim(4096, 0)
ax.set_axis_off()
ax.set_title(f'粗块合并预览: 451 细点位 → {len(occupied)} 块 '
             f'(块 {BLOCK:.0f} 世界单位, 颜色=块内总覆盖, 数字=点位数)',
             fontsize=13)
fig.tight_layout()
out = "output/coarse_blocks.png"
plt.savefig(out, dpi=110, bbox_inches='tight')
print("已写", out)
