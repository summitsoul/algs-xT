#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_game_xT.py — 把点位 xT 套到一场具体比赛的圈上 (热点图)

xT(点位, 圈) = β_p + φ_m(rel_bin)   圈内(rel<=1) ; 毒里清零(灰)
β_p 拆成 β_p_kill / β_p_place, φ 拆成 φ_kill / φ_place。

用法:
  python3 scripts/viz/apply_game_xT.py "replay/storm point/sp_apac-s_d__g6_abfd4cf4.json"

从 replay 的 ringPhases 读 finished 圈型, 画 圈3/圈4/圈5 的 击杀|排名 热点图
(Apex 圈从 1 起; 数据 stage 0→圈1, 所以 stage 2/3/4 = 圈3/圈4/圈5)。
"""
import json, math, os, re, sys
import numpy as np
from PIL import Image

REL_BINS = [0.3, 0.7, 1.0, 1.15, 1.6]
RATIO = 24.93
MAP_PNG = "map/storm point.png"
SIZE = 4096
STAGES = [2, 3, 4]  # 圈3 / 圈4 / 圈5


def rel_bin(rel):
    for i, th in enumerate(REL_BINS):
        if rel < th:
            return i
    return len(REL_BINS)


def world2img(x, y):
    return x / RATIO + 2048, -y / RATIO + 2048


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


def parse_circles(path):
    d = json.load(open(path, encoding='utf-8'))
    sc, fc = {}, {}
    for r in d['summary']['ringPhases']:
        if r['type'] == 'startClosing':
            sc[r['stage']] = r
        elif r['type'] == 'finishedClosing':
            fc[r['stage']] = r
    circles = {}
    for k in sorted(sc):
        a = sc[k]
        b = fc.get(k)
        cx = b['center']['x'] if b else a['center']['x']
        cy = b['center']['y'] if b else a['center']['y']
        circles[k] = ((cx, cy), a['endRadius'])
    return circles


GAME_PATH = sys.argv[1] if len(sys.argv) > 1 else "replay/storm point/sp_global_d__g9_29ce1521.json"
CIRCLES = parse_circles(GAME_PATH)

phi = json.load(open("data/phi.json", encoding='utf-8'))
pk = {(int(a), int(b)): v for a, b, v in
      (k.split(',') + [v] for k, v in phi['phi_kill'].items())}
pp = {(int(a), int(b)): v for a, b, v in
      (k.split(',') + [v] for k, v in phi['phi_place'].items())}

bk = np.load("data/beta_kill.npy")
bp = np.load("data/beta_place.npy")
pos_arr = np.load("data/points_pos.npy")
NP = len(bk)

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

img = Image.open(MAP_PNG).convert("RGB")
stages = [s for s in STAGES if s in CIRCLES]
fig, axes = plt.subplots(len(stages), 2, figsize=(13, 4.4 * len(stages)))


def which_label(w):
    return '击杀价值 xT_kill' if w == 'kill' else '排名价值 xT_place'


def draw(ax, stage, which):
    (cx, cy), rr = CIRCLES[stage]
    bx = bk if which == 'kill' else bp
    phi_tbl = pk if which == 'kill' else pp
    vals = np.zeros(NP)
    for i in range(NP):
        rel = math.hypot(pos_arr[i, 0] - cx, pos_arr[i, 1] - cy) / rr
        if rel <= 1.0:
            vals[i] = bx[i] + phi_tbl[(stage, rel_bin(rel))]
        else:
            vals[i] = np.nan
    ix = np.array([world2img(x, y)[0] for x, y in pos_arr])
    iy = np.array([world2img(x, y)[1] for x, y in pos_arr])
    ax.imshow(img, extent=[0, SIZE, SIZE, 0], zorder=0)
    cxi, cyi = world2img(cx, cy)
    ax.add_patch(plt.Circle((cxi, cyi), rr / RATIO, fill=False,
                            edgecolor='cyan', lw=2, zorder=2))
    inmask = ~np.isnan(vals)
    vmax = np.nanmax(vals)
    ax.scatter(ix[inmask], iy[inmask], c=vals[inmask], cmap='YlOrRd',
               s=70, edgecolors='none', alpha=0.9, zorder=3,
               vmin=0, vmax=vmax)
    ax.scatter(ix[~inmask], iy[~inmask], c='#666', s=18, edgecolors='none',
               alpha=0.5, zorder=3)
    ax.set_xlim(0, SIZE); ax.set_ylim(SIZE, 0)
    ax.set_axis_off()
    ax.set_title(f'圈{stage + 1} {which_label(which)} (r={rr:.0f})', fontsize=11)


for ri, st in enumerate(stages):
    draw(axes[ri, 0], st, 'kill')
    draw(axes[ri, 1], st, 'place')

fig.suptitle(f'{game_label(GAME_PATH)} — 点位 xT 热点图 (圈内 β_p+φ, 毒区清零)',
             fontsize=15, y=0.995)
plt.tight_layout()
out = "output/xT_" + os.path.basename(GAME_PATH).replace(".json", "") + ".png"
plt.savefig(out, dpi=105, bbox_inches='tight')
print("已写", out)
