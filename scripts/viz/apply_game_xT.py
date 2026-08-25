#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_game_xT.py — 把点位 xT 套到一场具体比赛的圈上 (热点图)

xT(点位, 圈) = kill(直接局部均值, 按当前圈分档) + place(β_p_place + φ_place, 非折现, 毒里不再硬记0)。

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


MIN_ZONE_R = 500.0  # 新圈(安全区)半径 < 此值视为无安全区(最后缩到一点), β_p 不衰减
R0 = 31000.0        # 圈1 新圈(安全区)半径, w_stage 归一化基准


def zone_w(rel, r1):
    """β_p 圈外平滑衰减权重。最后圈不衰减=1.0; 常规圈 rel<=1 全额, rel 1→1.6 线性降到0, rel>=1.6 归零。"""
    if r1 < MIN_ZONE_R:
        return 1.0
    return max(0.0, min(1.0, (1.6 - rel) / 0.6))


def stage_w(r1):
    """β_p 圈阶段权重: 圈越大(越早)点位固有价值兑现越低, 缩到决赛圈才全额。
    w_stage = 1 - r1/R0 → 圈1=0, 圈2≈0.52, 圈3≈0.74, 圈4≈0.87, 圈5≈0.94, 圈6≈1.0。"""
    return max(0.0, 1.0 - r1 / R0)


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
pp = {(int(a), int(b)): v for a, b, v in
      (k.split(',') + [v] for k, v in phi['phi_place'].items())}

bk = np.load("data/beta_kill.npy")     # 击杀 = 直接局部均值
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
    vals = np.zeros(NP)
    for i in range(NP):
        rel = math.hypot(pos_arr[i, 0] - cx, pos_arr[i, 1] - cy) / rr
        if which == 'kill':
            vals[i] = bk[i][stage]
        else:
            vals[i] = max(0.0, zone_w(rel, rr) * stage_w(rr) * bp[i] + pp[(stage, rel_bin(rel))])
    ix = np.array([world2img(x, y)[0] for x, y in pos_arr])
    iy = np.array([world2img(x, y)[1] for x, y in pos_arr])
    ax.imshow(img, extent=[0, SIZE, SIZE, 0], zorder=0)
    cxi, cyi = world2img(cx, cy)
    ax.add_patch(plt.Circle((cxi, cyi), rr / RATIO, fill=False,
                            edgecolor='cyan', lw=2, zorder=2))
    vmax = vals.max()
    ax.scatter(ix, iy, c=vals, cmap='YlOrRd',
               s=70, edgecolors='none', alpha=0.9, zorder=3,
               vmin=0, vmax=vmax)
    ax.set_xlim(0, SIZE); ax.set_ylim(SIZE, 0)
    ax.set_axis_off()
    ax.set_title(f'圈{stage + 1} {which_label(which)} (r={rr:.0f})', fontsize=11)


for ri, st in enumerate(stages):
    draw(axes[ri, 0], st, 'kill')
    draw(axes[ri, 1], st, 'place')

fig.suptitle(f'{game_label(GAME_PATH)} — 点位 xT 热点图 (击杀直接均值 + 排名 β_p+φ 非折现)',
             fontsize=15, y=0.995)
plt.tight_layout()
out = "output/xT_" + os.path.basename(GAME_PATH).replace(".json", "") + ".png"
plt.savefig(out, dpi=105, bbox_inches='tight')
print("已写", out)
