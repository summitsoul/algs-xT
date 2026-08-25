#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_points_xT.py — 渲染点位 β_p_place 基值 + 示例圈下完整 xT

输出:
  output/points_beta.png   β_p_place 基值 (圈无关, 红=好活点 蓝=废点)
  output/points_xT.png     示例圈(ring4 收完)下 xT = kill斜率 + max(0, β_place+φ_place)
"""
import json, math
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps as _cm

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


def diverging(t, vmax):
    t = max(-1.0, min(1.0, t / vmax))
    r, g, b, _ = _cm['RdBu_r']((t + 1) / 2)
    return (int(r * 255), int(g * 255), int(b * 255))


rows = [json.loads(l) for l in open(LONG, encoding='utf-8')]
kill_slope = np.load("data/beta_kill.npy")   # 击杀斜率 = 每秒击杀数(按圈分档)
beta_place = np.load("data/beta_place.npy")
pos_arr = np.load("data/points_pos.npy")
NP = len(beta_place)

# ---- 重算 φ_place (非折现 γ=1.0 线性解) ----
visits = defaultdict(int)
deaths = defaultdict(int)
pl_sum = defaultdict(float)
trans = defaultdict(lambda: defaultdict(int))
for r in rows:
    s = (r['phase'], r['rel_bin'])
    visits[s] += 1
    if r['died']:
        deaths[s] += 1
        pl_sum[s] += r['placement_pts']
    else:
        trans[s][(r['next_phase'], r['next_rel_bin'])] += 1
states = sorted(visits)
h = {s: deaths[s] / visits[s] for s in states}
pl = {s: (pl_sum[s] / deaths[s] if deaths[s] else 0.0) for s in states}
P = {}
for s in states:
    tot = sum(trans[s].values())
    P[s] = {s2: trans[s][s2] / tot for s2 in trans[s]} if tot else {}

ns = len(states)
idx = {s: i for i, s in enumerate(states)}
A = np.eye(ns)
b = np.zeros(ns)
for s in states:
    i = idx[s]
    b[i] = h[s] * pl[s]
    for s2, pr in P[s].items():
        A[i, idx[s2]] -= (1 - h[s]) * pr
V = np.linalg.solve(A, b)
phi_place = {s: float(V[idx[s]]) for s in states}


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

# ============ 图1: β_p_place 基值 ============
img = Image.open(MAP_PNG).convert('RGBA')
dr = ImageDraw.Draw(img)
for i in range(NP):
    ix, iy = world2img(pos_arr[i, 0], pos_arr[i, 1])
    col = diverging(beta_place[i], VMAX_BETA)
    r = 10
    dr.ellipse([ix - r, iy - r, ix + r, iy + r], fill=col + (230,), outline=(0, 0, 0, 160), width=2)
draw_legend(dr, font, 56, SIZE - 150,
            [((178, 34, 34), "β_p_place 高 (好活点)"), ((255, 255, 255), "β_p_place ≈ 0"),
             ((33, 102, 172), "β_p_place 低 (废点位)")])
img.save("output/points_beta.png")
print("已写 output/points_beta.png (β_p_place 排名偏离)")

# ============ 图2: 示例圈 xT (ring4 收完) ============
# 用 sp_na_d__g7 的 ring4(stage3) 收完状态: 中心(17818,-28712) 半径4000
RC = (17818.0, -28712.0)
RR = 4000.0
PHASE = 3
img = Image.open(MAP_PNG).convert('RGBA')
dr = ImageDraw.Draw(img)
cx, cy = world2img(RC[0], RC[1])
rr = RR / RATIO
dr.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(0, 220, 255, 220), width=4)
for i in range(NP):
    ix, iy = world2img(pos_arr[i, 0], pos_arr[i, 1])
    rel = math.hypot(pos_arr[i, 0] - RC[0], pos_arr[i, 1] - RC[1]) / RR
    xp = max(0.0, zone_w(rel, RR) * stage_w(RR) * beta_place[i] + phi_place[(PHASE, rel_bin(rel))])
    xT = kill_slope[i][PHASE] + xp
    t = xT / 9.0
    col = _cm['YlOrRd'](0.15 + 0.75 * max(0.0, min(1.0, t)))
    col = tuple(int(v * 255) for v in col[:3])
    dr.ellipse([ix - 10, iy - 10, ix + 10, iy + 10], fill=col + (220,),
               outline=(0, 0, 0, 150), width=2)
draw_legend(dr, font, 56, SIZE - 190,
            [((178, 34, 34), "xT 高 (圈内好点)"), ((255, 230, 150), "xT 中"),
             ((90, 90, 90), "xT 低 (毒里/废点)"), ((0, 220, 255), "示例圈 (ring4 收完)")])
img.save("output/points_xT.png")
print("已写 output/points_xT.png")
