#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_points_xT.py — 渲染点位 β_p 基值 + 示例圈下完整 xT (毒区清零)

输出:
  output/points_beta.png   β_p 基值 (圈无关, 红=好点位 蓝=废点位)
  output/points_xT.png     示例圈(ring4 收完)下 xT = β_p+φ(圈内) / 0(毒)
"""
import json, math
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps as _cm

GAMMA = 0.99
REL_BINS = [0.3, 0.7, 1.0, 1.15, 1.6]
LONG = "data/long_table.jsonl"
RATIO = 24.93
MAP_PNG = "map/storm point.png"
SIZE = 4096
VMAX_BETA = 3.0


def rel_bin(rel):
    for i, th in enumerate(REL_BINS):
        if rel < th:
            return i
    return len(REL_BINS)


def diverging(t, vmax):
    t = max(-1.0, min(1.0, t / vmax))
    r, g, b, _ = _cm['RdBu_r']((t + 1) / 2)
    return (int(r * 255), int(g * 255), int(b * 255))


rows = [json.loads(l) for l in open(LONG, encoding='utf-8')]
beta_p = np.load("data/points_beta.npy")
pos_arr = np.load("data/points_pos.npy")
NP = len(beta_p)

# ---- 重算 φ ----
visits = defaultdict(int)
k_sum = defaultdict(float)
deaths = defaultdict(int)
pl_sum = defaultdict(float)
trans = defaultdict(lambda: defaultdict(int))
for r in rows:
    s = (r['phase'], r['rel_bin'])
    visits[s] += 1
    k_sum[s] += r['kills']
    if r['died']:
        deaths[s] += 1
        pl_sum[s] += r['placement_pts']
    else:
        trans[s][(r['next_phase'], r['next_rel_bin'])] += 1
states = sorted(visits)
k = {s: k_sum[s] / visits[s] for s in states}
h = {s: deaths[s] / visits[s] for s in states}
pl = {s: (pl_sum[s] / deaths[s] if deaths[s] else 0.0) for s in states}
P = {}
for s in states:
    tot = sum(trans[s].values())
    P[s] = {s2: trans[s][s2] / tot for s2 in trans[s]} if tot else {}
V = {s: 0.0 for s in states}
for _ in range(5000):
    Vn, d = {}, 0.0
    for s in states:
        fut = sum(P[s].get(s2, 0) * V[s2] for s2 in P[s]) if P[s] else 0.0
        Vn[s] = k[s] + GAMMA * ((1 - h[s]) * fut + h[s] * pl[s])
        d = max(d, abs(Vn[s] - V[s]))
    V = Vn
    if d < 1e-6:
        break
phi = {s: V[s] for s in states}


def world2img(x, y):
    return x / RATIO + 2048, -y / RATIO + 2048


def draw_legend(dr, font, x0, y0, items):
    dr.rectangle([x0 - 18, y0 - 18, x0 + 560, y0 + 40 * len(items) + 6], fill=(255, 255, 255, 205))
    for i, (color, label) in enumerate(items):
        dr.ellipse([x0, y0 + i * 40, x0 + 30, y0 + i * 40 + 30], fill=color, outline=(0, 0, 0, 200), width=2)
        dr.text((x0 + 42, y0 + i * 40), label, fill=(20, 20, 20, 255), font=font)


try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
except Exception:
    font = ImageFont.load_default()

# ============ 图1: β_p 基值 ============
img = Image.open(MAP_PNG).convert('RGBA')
dr = ImageDraw.Draw(img)
for i in range(NP):
    ix, iy = world2img(pos_arr[i, 0], pos_arr[i, 1])
    t = beta_p[i]
    col = diverging(t, VMAX_BETA)
    r = 10
    dr.ellipse([ix - r, iy - r, ix + r, iy + r], fill=col + (230,), outline=(0, 0, 0, 160), width=2)
draw_legend(dr, font, 56, SIZE - 150,
            [((178, 34, 34), "β_p 高 (好点位)"), ((255, 255, 255), "β_p ≈ 0"),
             ((33, 102, 172), "β_p 低 (废点位)")])
img.save("output/points_beta.png")
print("已写 output/points_beta.png")

# ============ 图2: 示例圈 xT (ring4 收完, 毒区清零) ============
# 用 sp_na_d__g7 的 ring4(stage3) 收完状态: 中心(17818,-28712) 半径4000
RC = (17818.0, -28712.0)
RR = 4000.0
PHASE = 3
img = Image.open(MAP_PNG).convert('RGBA')
dr = ImageDraw.Draw(img)
# 先画圈
cx, cy = world2img(RC[0], RC[1])
rr = RR / RATIO
dr.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(0, 220, 255, 220), width=4)
for i in range(NP):
    ix, iy = world2img(pos_arr[i, 0], pos_arr[i, 1])
    rel = math.hypot(pos_arr[i, 0] - RC[0], pos_arr[i, 1] - RC[1]) / RR
    if rel <= 1.0:
        xT = beta_p[i] + phi[(PHASE, rel_bin(rel))]
        t = xT / 9.0
        col = _cm['YlOrRd'](0.15 + 0.75 * max(0.0, min(1.0, t)))
        col = tuple(int(v * 255) for v in col[:3])
        label_r = 10
    else:
        col = (90, 90, 90)   # 毒区 = 清零(灰)
        label_r = 8
    dr.ellipse([ix - label_r, iy - label_r, ix + label_r, iy + label_r], fill=col + (220,),
               outline=(0, 0, 0, 150), width=2)
draw_legend(dr, font, 56, SIZE - 190,
            [((178, 34, 34), "xT 高 (圈内好点)"), ((255, 230, 150), "xT 中"),
             ((90, 90, 90), "毒区 = 清零"), ((0, 220, 255), "示例圈 (ring4 收完)")])
img.save("output/points_xT.png")
print("已写 output/points_xT.png")
