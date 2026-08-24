#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
terrain_heatmap.py — 点位价值热力图 (30×30 网格 + 队伍固定效应)

把位置从"离圈心多远"升级到"踩在哪块地":
  每个 (game,team) 一个观测:  y = 最终分 - 环形期望(该队整场平均位置的质量)
                            X = 在各网格停留的时间占比(frac) + 队伍 one-hot
  ridge 只正则化网格块 -> β_grid = 每块地的"地皮价值"(净掉圈位置 + 队伍实力)

输出:
  data/terrain_30x30.csv   网格价值
  output/terrain_heatmap.png  半透明热力图叠加层(图像坐标 4096)
"""
import json, numpy as np
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFilter

GRID = 30
LAM = 3.0
MIN_COV = 2          # 网格至少被多少个 team-game 踩过才显示
VMAX = 2.0           # 颜色饱和点(分), 红=值钱点位 蓝=死亡点
RATIO = 24.93
LONG = "data/long_table.jsonl"
META = "data/long_table.meta.json"
VRING = "data/v_ring.json"
OUT_CSV = "data/terrain_30x30.csv"
OUT_PNG = "output/terrain_heatmap.png"
SIZE = 4096

rows = [json.loads(l) for l in open(LONG, encoding='utf-8')]
meta = json.load(open(META))
bnd = meta['bounds']
xmin, xmax, ymin, ymax = bnd['xmin'], bnd['xmax'], bnd['ymin'], bnd['ymax']
vr = json.load(open(VRING, encoding='utf-8'))
E_final = {tuple(map(int, k.split(','))): v for k, v in vr['E_final'].items()}


def grid(ax, ay):
    gx = int((ax - xmin) / (xmax - xmin) * GRID)
    gy = int((ay - ymin) / (ymax - ymin) * GRID)
    return min(max(gx, 0), GRID - 1), min(max(gy, 0), GRID - 1)


# 每 (game,team) 最终分
final = defaultdict(float)
for r in rows:
    final[(r['game'], r['team'])] += r['kills'] + r['placement_pts']

# unit = (game, team)
tg = defaultdict(list)
for r in rows:
    tg[(r['game'], r['team'])].append(r)

team_names = sorted({r['team_name'] for r in rows})
team_idx = {t: i for i, t in enumerate(team_names)}
P, T = GRID * GRID, len(team_names)

units = []
for (gid, tid), rws in tg.items():
    ring_exp = sum(E_final[(r['phase'], r['rel_bin'])] for r in rws) / len(rws)
    cnt = defaultdict(int)
    for r in rws:
        cnt[grid(r['ax'], r['ay'])] += 1
    n = len(rws)
    frac = {g: c / n for g, c in cnt.items()}
    units.append({'name': rws[0]['team_name'], 'ring_exp': ring_exp,
                  'final': final[(gid, tid)], 'frac': frac})

N = len(units)
X = np.zeros((N, P + T - 1))
y = np.zeros(N)
for i, u in enumerate(units):
    y[i] = u['final'] - u['ring_exp']
    for (gx, gy), f in u['frac'].items():
        X[i, gy * GRID + gx] = f
    ti = team_idx[u['name']]
    if ti > 0:
        X[i, P + ti - 1] = 1.0

A = X.T @ X
b = X.T @ y
Areg = A.copy()
Areg[:P, :P] += LAM * np.eye(P)
beta = np.linalg.solve(Areg, b)
beta_grid = beta[:P].reshape(GRID, GRID)

pred = X @ beta
ss_res = ((y - pred) ** 2).sum()
ss_tot = ((y - y.mean()) ** 2).sum()
r2 = 1 - ss_res / ss_tot

cov = np.zeros((GRID, GRID), int)
for u in units:
    for (gx, gy) in u['frac']:
        cov[gy, gx] += 1

print(f"N units={N} | grid={P} | teams={T} | λ={LAM} | R²={r2:.3f}")
print(f"β_grid 范围: {beta_grid.min():+.2f} ~ {beta_grid.max():+.2f}")
print(f"覆盖 (≥{MIN_COV} team-game) 的网格: {(cov >= MIN_COV).sum()}/{P}")

# 保存 CSV
with open(OUT_CSV, 'w') as f:
    f.write("gx,gy,value,coverage\n")
    for gy in range(GRID):
        for gx in range(GRID):
            f.write(f"{gx},{gy},{beta_grid[gy, gx]:.4f},{cov[gy, gx]}\n")

# ---- 渲染热力图 (图像坐标 4096) ----
vmax = VMAX
from matplotlib import colormaps as _colormaps
_cmap = _colormaps['RdBu_r']   # 红=高 白=0 蓝=低 (感知均匀, 中点近白)


def diverging(t):
    """t in [-1,1] -> 蓝(低) -> 白(0) -> 红(高)"""
    t = max(-1.0, min(1.0, t))
    r, g, b, _ = _cmap((t + 1) / 2)
    return (int(r * 255), int(g * 255), int(b * 255))


img = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
dr = ImageDraw.Draw(img)
cw = (xmax - xmin) / GRID          # 每格世界宽度
ch = (ymax - ymin) / GRID
for gy in range(GRID):
    for gx in range(GRID):
        if cov[gy, gx] < MIN_COV:
            continue
        v = beta_grid[gy, gx]
        t = v / vmax
        col = diverging(t)
        alpha = int(150 + 105 * abs(t))      # 越强越实
        # 世界 -> 图像: img_x = x/RATIO+2048, img_y = -y/RATIO+2048
        wx0 = xmin + gx * cw
        wx1 = xmin + (gx + 1) * cw
        wy0 = ymin + gy * ch
        wy1 = ymin + (gy + 1) * ch
        ix0 = wx0 / RATIO + 2048
        ix1 = wx1 / RATIO + 2048
        iy1 = -wy0 / RATIO + 2048       # 上边
        iy0 = -wy1 / RATIO + 2048       # 下边
        dr.rectangle([ix0, iy0, ix1, iy1], fill=col + (alpha,))

img = img.filter(ImageFilter.GaussianBlur(22))
img.save(OUT_PNG)
print(f"热力图 {OUT_PNG}  (vmax={vmax:.2f})")
