#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
late_point_value.py — 后期(圈3+)各点位的击杀/排名价值

回答指挥真正的问题: "到了后期, 我去哪个点位能拿更多人击杀分和排名分?"

做法(足球 xT 的空间版, 但 state = 我们之前提取的 451 个实际点位, 不是格子):
  - 只取后期桶 (phase >= LATE_PHASE, t >= 120)
  - 每桶归属到最近的 451 点位之一
  - 每个点位 p:
      k(p)  = 该点即时击杀分/桶
      h(p)  = 该点死亡概率/桶
      pl(p) = 该点死亡时的平均排名分
      P(p→p') = 存活桶的点位转移概率
  - Bellman:  V(p) = k(p) + γ[(1-h(p))·Σ P(p→p')V(p') + h(p)·pl(p)]
    线性分解: V_kill(击杀价值) + V_place(排名价值)
"""
import json, csv
from collections import defaultdict
import numpy as np

LONG = "data/long_table.jsonl"
POS = "data/positions.csv"
TMIN = 120
LATE_PHASE = 3
GAMMA = 0.99

pts = []
for row in csv.DictReader(open(POS, encoding='utf-8')):
    pts.append((float(row['cx']), float(row['cy'])))
pts = np.array(pts)
NP = len(pts)


def nearest(x, y):
    return int(np.argmin(np.hypot(pts[:, 0] - x, pts[:, 1] - y)))


rows = [json.loads(l) for l in open(LONG, encoding='utf-8')]
late = [r for r in rows if r['t'] >= TMIN and r['phase'] >= LATE_PHASE]
n_bucket = len(late)
print(f"后期桶(phase>={LATE_PHASE}): {n_bucket} | 点位 {NP}")

kills = np.zeros(NP); visits = np.zeros(NP)
deaths = np.zeros(NP); plsum = np.zeros(NP)
trans = defaultdict(lambda: defaultdict(float))

for r in late:
    j = nearest(r['ax'], r['ay'])
    visits[j] += 1
    kills[j] += r['kills']
    if r.get('died'):
        deaths[j] += 1
        plsum[j] += r.get('placement_pts', 0)
    else:
        j2 = nearest(r['next_ax'], r['next_ay'])
        trans[j][j2] += 1

# 各点位统计量
k = kills / np.where(visits > 0, visits, 1)
h = deaths / np.where(visits > 0, visits, 1)
pl = plsum / np.where(deaths > 0, deaths, 1)

# 转移矩阵 P[j][j2] = P(point j -> j2 | 存活)
P = np.zeros((NP, NP))
for j in range(NP):
    tot = sum(trans[j].values())
    if tot > 0:
        for j2, c in trans[j].items():
            P[j, j2] = c / tot

# Bellman 值迭代 + 线性分解
def solve(imm, term):
    V = np.zeros(NP)
    for _ in range(20000):
        Vn = imm + GAMMA * ((1 - h) * (P @ V) + term)
        if np.max(np.abs(Vn - V)) < 1e-8:
            break
        V = Vn
    return V

V_kill = solve(k, np.zeros(NP))
V_place = solve(np.zeros(NP), h * pl)
V_total = V_kill + V_place

# 覆盖度: 有足够访问的点位数
cov = {50: (visits >= 50).sum(), 20: (visits >= 20).sum(),
       5: (visits >= 5).sum()}
print(f"点位访问量: 总{NP} | >=5桶 {(visits>=5).sum()} | "
      f">=20桶 {(visits>=20).sum()} | >=50桶 {(visits>=50).sum()}")

# 数值量纲: 每桶期望分(分/5s), 换算成"分/分钟"便于读
print("\n后期点位价值分布 (分/分钟):")
for name, V in [('击杀', V_kill), ('排名', V_place), ('合计', V_total)]:
    vm = V * 12  # 5s桶 -> 分钟
    print(f"  {name}: 均值 {vm[visits>=5].mean():.2f}   "
          f"中位 {np.median(vm[visits>=5]):.2f}   最大 {vm.max():.2f}  @点 {vm.argmax()}")

# 前 15 高价值点
print("\n后期合计价值 TOP 15 点位:")
order = np.argsort(-V_total)
for rank, j in enumerate(order[:15]):
    if visits[j] < 5:
        continue
    print(f"  #{rank+1} 点{j:>3} (x={pts[j,0]:>8.0f}, y={pts[j,1]:>8.0f}) "
          f"桶={visits[j]:>4} | 击杀 {V_kill[j]*12:>5.2f} 排名 {V_place[j]*12:>5.2f} "
          f"合计 {V_total[j]*12:>5.2f} 分/分钟")

# 校验: 队伍后期轨迹均值 预测 实际(后期击杀 / 排名)
team = defaultdict(lambda: {'ak': 0.0, 'ap': 0.0, 'sk': 0.0, 'sp': 0.0,
                            'w': 0, 'g': set()})
for r in late:
    nm = r['team_name'].lower()
    a = team[nm]
    j = nearest(r['ax'], r['ay'])
    a['ak'] += r['kills']
    a['ap'] += r['placement_pts']
    a['sk'] += V_kill[j]
    a['sp'] += V_place[j]
    a['w'] += 1
    a['g'].add(r['game'])
rec = []
for nm, a in team.items():
    if len(a['g']) < 3:
        continue
    n = len(a['g'])
    rec.append({'ak': a['ak'] / n, 'ap': a['ap'] / n,
                'sk': a['sk'] / a['w'], 'sp': a['sp'] / a['w']})
r_kill = np.corrcoef([r['ak'] for r in rec], [r['sk'] for r in rec])[0, 1]
r_place = np.corrcoef([r['ap'] for r in rec], [r['sp'] for r in rec])[0, 1]
print(f"\n校验({len(rec)}队): 实际后期击杀 vs 点位击杀价值 r={r_kill:+.3f}  "
      f"实际排名分 vs 点位排名价值 r={r_place:+.3f}")

# 存数组
np.save("data/late_V_kill.npy", V_kill)
np.save("data/late_V_place.npy", V_place)
np.save("data/late_V_total.npy", V_total)
np.save("data/late_visits.npy", visits)
print("\n已存 data/late_V_{kill,place,total}.npy, late_visits.npy")