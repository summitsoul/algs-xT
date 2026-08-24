#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phi_split.py — 把 φ(xT) 按奖励通道分解成 击杀价值 φ_kill 和 排名价值 φ_place

Bellman 算子是线性的, 所以价值函数可以按奖励来源拆开:

    V(s)        = k(s) + γ[(1−h)·E[V']      + h·pl]
    V_kill(s)   = k(s) + γ[(1−h)·E[V_kill']          ]   # 只留击杀即时奖励
    V_place(s)  =         γ[(1−h)·E[V_place'] + h·pl ]   # 只留排名终态奖励
    严格成立: V = V_kill + V_place

其中:
    k(s)   = 击杀期望分 = k_sum[s] / visits[s]   (状态 s 下平均每 5 秒时刻击杀数)
    h(s)   = 淘汰概率   = deaths[s] / visits[s]
    pl(s)  = 排名分     = pl_sum[s] / deaths[s]  (该状态死时平均拿到的名次分)

输出 data/phi.json (phi / phi_kill / phi_place 三张表 + E_final)
"""
import json
from collections import defaultdict

GAMMA = 0.99
LONG = "data/long_table.jsonl"
REL_NAMES = ['圈心', '圈内', '圈内边缘', '圈外贴边', '圈外', '远圈外']

rows = [json.loads(l) for l in open(LONG, encoding='utf-8')]

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


def solve(imm, term):
    """imm: 即时奖励(击杀分), term: 终态奖励(排名分·h)"""
    V = {s: 0.0 for s in states}
    for _ in range(5000):
        Vn, delta = {}, 0.0
        for s in states:
            fut = sum(P[s].get(s2, 0) * V[s2] for s2 in P[s]) if P[s] else 0.0
            Vn[s] = imm(s) + GAMMA * ((1 - h[s]) * fut + term(s))
            delta = max(delta, abs(Vn[s] - V[s]))
        V = Vn
        if delta < 1e-6:
            break
    return V


phi_total = solve(lambda s: k[s], lambda s: h[s] * pl[s])
phi_kill = solve(lambda s: k[s], lambda s: 0.0)
phi_place = solve(lambda s: 0.0, lambda s: h[s] * pl[s])

maxdiff = max(abs(phi_total[s] - (phi_kill[s] + phi_place[s])) for s in states)
print(f"线性性校验: max|φ_total − (φ_kill+φ_place)| = {maxdiff:.2e} (应为 0)")
print(f"\n{'圈':>4} {'档位':>8} {'击杀φ':>8} {'排名φ':>8} {'合计φ':>8}   (每5秒桶的期望未来分)")
for ph in sorted({s[0] for s in states}):
    for rb in sorted({s[1] for s in states if s[0] == ph}):
        s = (ph, rb)
        print(f"{ph:>4} {REL_NAMES[rb]:>8} {phi_kill[s]:>8.3f} "
              f"{phi_place[s]:>8.3f} {phi_total[s]:>8.3f}")

# E_final (不折现期望终分, 供 β_p 回归复用)
tg = defaultdict(list)
final = defaultdict(float)
for r in rows:
    tg[(r['game'], r['team'])].append(r)
    final[(r['game'], r['team'])] += r['kills'] + r['placement_pts']
ef_sum = defaultdict(float)
for r in rows:
    ef_sum[(r['phase'], r['rel_bin'])] += final[(r['game'], r['team'])]
E_final = {s: ef_sum[s] / visits[s] for s in states}

json.dump({'phi': {f"{a},{b}": v for (a, b), v in phi_total.items()},
           'phi_kill': {f"{a},{b}": v for (a, b), v in phi_kill.items()},
           'phi_place': {f"{a},{b}": v for (a, b), v in phi_place.items()},
           'E_final': {f"{a},{b}": v for (a, b), v in E_final.items()}},
          open("data/phi.json", "w", encoding='utf-8'))
print("\n已写 data/phi.json (phi / phi_kill / phi_place)")