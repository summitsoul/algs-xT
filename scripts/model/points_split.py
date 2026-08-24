#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
points_split.py — 点位 xT 拆成 击杀价值 / 排名价值

定稿模型: xT(点位p, 第m圈) = β_p + φ_m(rel_bin), 毒里清零。
这里按奖励通道线性分解:
  xT_kill(p,圈)  = β_p_kill  + φ_kill(rel_bin)
  xT_place(p,圈) = β_p_place + φ_place(rel_bin)

φ_kill/φ_place: Bellman 线性分解 (击杀即时奖励 vs 淘汰时排名奖励)。
β_p_kill/β_p_place: ridge(λ=3) + 队伍固定效应, 分别解释
  "击杀分 − 该队圈位期望击杀" 和 "排名分 − 该队圈位期望排名"。
"""
import json, math
from collections import defaultdict
import numpy as np

GAMMA = 0.99
REL_BINS = [0.3, 0.7, 1.0, 1.15, 1.6]
LONG = "data/long_table.jsonl"
POS = "data/positions.csv"
LAM = 3.0
MIN_COV = 3


def rel_bin(rel):
    for i, th in enumerate(REL_BINS):
        if rel < th:
            return i
    return len(REL_BINS)


rows = [json.loads(l) for l in open(LONG, encoding='utf-8')]

pos = []
for line in open(POS, encoding='utf-8'):
    if line.startswith('id'):
        continue
    p = line.strip().split(',')
    pos.append((float(p[1]), float(p[2])))
pos_arr = np.array(pos)
NP = len(pos)

# ---- 1) φ 分解: 击杀 / 排名 (Bellman 线性) ----
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


phi_kill = solve(lambda s: k[s], lambda s: 0.0)
phi_place = solve(lambda s: 0.0, lambda s: h[s] * pl[s])

# ---- 2) E_final 分解 (圈位期望, 供 β_p 回归控制) ----
tg = defaultdict(list)
final_k = defaultdict(float)
final_p = defaultdict(float)
for r in rows:
    tg[(r['game'], r['team'])].append(r)
    final_k[(r['game'], r['team'])] += r['kills']
    final_p[(r['game'], r['team'])] += r['placement_pts']
efk_sum = defaultdict(float)
efp_sum = defaultdict(float)
for r in rows:
    s = (r['phase'], r['rel_bin'])
    efk_sum[s] += final_k[(r['game'], r['team'])]
    efp_sum[s] += final_p[(r['game'], r['team'])]
E_final_kill = {s: efk_sum[s] / visits[s] for s in states}
E_final_place = {s: efp_sum[s] / visits[s] for s in states}


def extract_stays(rws, R=150, T=6, TMIN=120):
    stays = []
    i, n = 0, len(rws)
    while i < n:
        j, pts = i, []
        while j < n:
            pts.append((rws[j]['ax'], rws[j]['ay']))
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            if all(math.hypot(p[0] - cx, p[1] - cy) <= R for p in pts):
                j += 1
            else:
                pts.pop()
                break
        if len(pts) >= T and rws[i]['t'] >= TMIN:
            stays.append((sum(p[0] for p in pts) / len(pts),
                          sum(p[1] for p in pts) / len(pts), len(pts)))
        i = max(i + 1, j)
    return stays


units = []
for (gid, tid), rws in tg.items():
    rws = sorted(rws, key=lambda r: r['t'])
    rk = sum(E_final_kill[(r['phase'], r['rel_bin'])] for r in rws) / len(rws)
    rp = sum(E_final_place[(r['phase'], r['rel_bin'])] for r in rws) / len(rws)
    stays = extract_stays(rws)
    occ = defaultdict(float)
    tot = sum(s[2] for s in stays)
    if tot > 0:
        for (cx, cy, nb) in stays:
            d = np.hypot(pos_arr[:, 0] - cx, pos_arr[:, 1] - cy)
            occ[int(np.argmin(d))] += nb / tot
    units.append({'name': rws[0]['team_name'], 'rk': rk, 'rp': rp,
                  'fk': final_k[(gid, tid)], 'fp': final_p[(gid, tid)],
                  'occ': dict(occ)})

team_names = sorted({u['name'] for u in units})
team_idx = {t: i for i, t in enumerate(team_names)}
N, Tdim = len(units), len(team_names)
X = np.zeros((N, NP + Tdim - 1))
yk = np.zeros(N)
yp = np.zeros(N)
for i, u in enumerate(units):
    yk[i] = u['fk'] - u['rk']
    yp[i] = u['fp'] - u['rp']
    for p, f in u['occ'].items():
        X[i, p] = f
    ti = team_idx[u['name']]
    if ti > 0:
        X[i, NP + ti - 1] = 1.0

A = X.T @ X
Areg = A.copy()
Areg[:NP, :NP] += LAM * np.eye(NP)
beta_kill = np.linalg.solve(Areg, X.T @ yk)[:NP]
beta_place = np.linalg.solve(Areg, X.T @ yp)[:NP]

cov = np.zeros(NP, int)
for u in units:
    for p in u['occ']:
        cov[p] += 1

print(f"点位 {NP} | team-game {N} | teams {Tdim} | λ={LAM}")
print(f"β_p_kill  范围: {beta_kill.min():+.2f} ~ {beta_kill.max():+.2f}")
print(f"β_p_place 范围: {beta_place.min():+.2f} ~ {beta_place.max():+.2f}")

ok = cov >= MIN_COV
print(f"\n== 击杀价值最高 (好打点, 覆盖>={MIN_COV}) ==")
for i in np.argsort(-beta_kill):
    if ok[i]:
        print(f"  ({pos[i][0]:8.0f},{pos[i][1]:8.0f}) 击杀β={beta_kill[i]:+.2f} "
              f"排名β={beta_place[i]:+.2f} 覆盖{cov[i]}")
        if sum(ok) and np.argsort(-beta_kill).tolist().index(i) >= 9:
            pass

print(f"\n== 排名价值最高 (好活点, 覆盖>={MIN_COV}) ==")
shown = 0
for i in np.argsort(-beta_place):
    if ok[i] and shown < 10:
        print(f"  ({pos[i][0]:8.0f},{pos[i][1]:8.0f}) 排名β={beta_place[i]:+.2f} "
              f"击杀β={beta_kill[i]:+.2f} 覆盖{cov[i]}")
        shown += 1

# 击杀 vs 排名 的分离度
print(f"\nβ_kill 与 β_place 相关性: "
      f"{np.corrcoef(beta_kill[ok], beta_place[ok])[0,1]:+.3f}")

np.save("data/beta_kill.npy", beta_kill)
np.save("data/beta_place.npy", beta_place)
json.dump({'phi': {f"{a},{b}": v for (a, b), v in
                   {s: phi_kill[s] + phi_place[s] for s in states}.items()},
           'phi_kill': {f"{a},{b}": v for (a, b), v in phi_kill.items()},
           'phi_place': {f"{a},{b}": v for (a, b), v in phi_place.items()},
           'E_final_kill': {f"{a},{b}": v for (a, b), v in E_final_kill.items()},
           'E_final_place': {f"{a},{b}": v for (a, b), v in E_final_place.items()}},
          open("data/phi.json", "w", encoding='utf-8'))
print("\n已写 data/beta_kill.npy, data/beta_place.npy, data/phi.json (含 φ_kill/φ_place)")