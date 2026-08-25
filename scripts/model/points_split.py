#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
points_split.py — 点位 xT 拆成 击杀价值 / 排名价值 (定稿)

定稿模型:
  xT(p, 圈m) = xT_kill(p, m) + xT_place(p, m)

  xT_kill(p, m) = 该点该圈每秒击杀数 (斜率), 沿轨迹积分得累计击杀。
                 无回归、无递推——击杀是即时奖励, 只跟点位地形/掩体 +
                 当前圈阶段有关(很多点要后期才打得起来), 跟转去哪无关。
                 一局内 xT_kill 单调递增, 换点只改斜率、之前累计不清零。

  xT_place(p, m) = β_p_place + φ_place(rel_bin)
                 φ_place = 圈相对排名价值, 非折现(γ=1.0) Bellman 线性解:
                           期望最终排名分(不加时间折扣)。
                 β_p_place = 点位排名偏离 (ridge λ=3 + 队伍固定效应), 解释
                            "排名分 − 该队圈位期望排名(用 φ_place 自身)"。
                 再拆 β_p_place = β_terrain(掩体/地形) + β_rotation(转点/连通,
                 粗块转移前瞻递归)——纯拆解, 合起来不变, 见第 5 步。

  毒里软处理在下游(igl/viz)做: 不再硬记 0, 按"圈外贴边"档给一个小正数。
"""
import json, math
from collections import defaultdict
import numpy as np

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

# ---- 1) φ_place: 圈相对排名价值 (非折现 Bellman, 线性求解) ----
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


def solve_place():
    """非折现(γ=1.0) Bellman: V[s] = (1-h[s])·Σ P[s][s']·V[s'] + h[s]·pl[s]
    即 (I − M)V = h·pl, M[s][s'] = (1-h[s])·P[s][s']。线性求解, 避免值迭代收敛问题。"""
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
    return {s: float(V[idx[s]]) for s in states}


phi_place = solve_place()

# ---- 2) E_final 分解 (圈位期望, 供 igl_two_layer 的 L2 兑现复用) ----
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
                          sum(p[1] for p in pts) / len(pts), len(pts),
                          rws[i]['phase']))
        i = max(i + 1, j)
    return stays


# ---- 3) 点位停留占比 + 排名偏离回归 (β_p_place) ----
# 目标 = 排名分 − 圈位期望排名(用 φ_place 自身, 保证 β 的零点跟 φ 同口径)
units = []
for (gid, tid), rws in tg.items():
    rws = sorted(rws, key=lambda r: r['t'])
    rp = sum(phi_place[(r['phase'], r['rel_bin'])] for r in rws) / len(rws)
    stays = extract_stays(rws)
    occ = defaultdict(float)
    tot = sum(s[2] for s in stays)
    if tot > 0:
        for (cx, cy, nb, ph) in stays:
            d = np.hypot(pos_arr[:, 0] - cx, pos_arr[:, 1] - cy)
            occ[int(np.argmin(d))] += nb / tot
    units.append({'name': rws[0]['team_name'], 'rp': rp,
                  'fp': final_p[(gid, tid)], 'occ': dict(occ)})

team_names = sorted({u['name'] for u in units})
team_idx = {t: i for i, t in enumerate(team_names)}
N, Tdim = len(units), len(team_names)
X = np.zeros((N, NP + Tdim - 1))
yp = np.zeros(N)
for i, u in enumerate(units):
    yp[i] = u['fp'] - u['rp']
    for p, f in u['occ'].items():
        X[i, p] = f
    ti = team_idx[u['name']]
    if ti > 0:
        X[i, NP + ti - 1] = 1.0

A = X.T @ X
Areg = A.copy()
Areg[:NP, :NP] += LAM * np.eye(NP)
beta_place = np.linalg.solve(Areg, X.T @ yp)[:NP]

# ---- 4) 击杀价值: 该点该圈每秒击杀数 (斜率), 沿轨迹积分得累计击杀 (无回归无递推) ----
# kill_slope[p, m] = 第 m 圈在点位 p 的历史击杀数 / 被站总秒数 (击杀/秒)。
# 下游把它沿轨迹积分: xT_kill(t) = Σ slope·Δt, 单调递增, 换点只改斜率不清零。
NPH = max(r['phase'] for r in rows) + 1
BUCKET = 5  # 采样步长(秒)
try:
    from scipy.spatial import cKDTree
    j_all = cKDTree(pos_arr).query(np.c_[[r['ax'] for r in rows],
                                         [r['ay'] for r in rows]])[1]
except Exception:
    j_all = np.array([int(np.argmin(np.hypot(pos_arr[:, 0] - r['ax'],
                                             pos_arr[:, 1] - r['ay'])))
                      for r in rows])

kills_pm = np.zeros((NP, NPH))
rows_pm = np.zeros((NP, NPH))
for idx, r in enumerate(rows):
    p = int(j_all[idx])
    m = int(r['phase'])
    kills_pm[p, m] += r['kills']
    rows_pm[p, m] += 1
time_pm = rows_pm * BUCKET
kill_slope = kills_pm / np.maximum(1.0, time_pm)   # 击杀/秒

cov = np.zeros(NP, int)
for u in units:
    for p in u['occ']:
        cov[p] += 1

# ---- 5) β 拆解: 掩体/地形价值 + 转点/连通性价值 (折中) ----
# β_full(=beta_place) 隐式含"转点/连通"价值, 这里显式拆成两子项:
#   beta_rotation[p] = 块级前瞻价值 (粗块转移 P(b'|b) 递归), "从这块能转到多好的地方"
#   beta_terrain[p]  = 点级残差 (掩体/高地等点位本身价值)
# 纯拆解: beta_terrain + beta_rotation == beta_place, 不改模型输出。
ROT_GAMMA = 0.8    # 转点前瞻折扣
BLOCK = 8000.0     # 粗块边长 (世界单位)
LATE_PHASE = 2     # 圈3+(stage 2+): 点位被踩满后转点才有连通性意义, 开局随便转不算

bx0, by0 = float(pos_arr[:, 0].min()), float(pos_arr[:, 1].min())
blk_index = {}
for i in range(NP):
    b = (int((pos_arr[i, 0] - bx0) // BLOCK), int((pos_arr[i, 1] - by0) // BLOCK))
    if b not in blk_index:
        blk_index[b] = len(blk_index)
p2b = np.array([blk_index[(int((pos_arr[i, 0] - bx0) // BLOCK),
                           int((pos_arr[i, 1] - by0) // BLOCK))] for i in range(NP)])
NB = len(blk_index)

# 块级转移: 每 (game,team) 的 stay 序列 -> 块序列 -> 相邻块转移
# 只取圈3+ 的 stay: 开局(圈1/2)大家随便转点(搜刮/乱动), 无连通性决策意义;
# 圈3+ 好点位被踩满、转点才难, "从这块能转到哪"才体现连通性价值。
trans_b = np.zeros((NB, NB))
stay_b = np.zeros(NB)
n_stays_all = 0
n_stays_late = 0
for (gid, tid), rws in tg.items():
    rws = sorted(rws, key=lambda r: r['t'])
    seq = []
    for (cx, cy, nb, ph) in extract_stays(rws):
        n_stays_all += 1
        if ph < LATE_PHASE:
            continue
        n_stays_late += 1
        d = np.hypot(pos_arr[:, 0] - cx, pos_arr[:, 1] - cy)
        seq.append(p2b[int(np.argmin(d))])
    for a, b in zip(seq, seq[1:]):
        trans_b[a, b] += 1
    for b in seq:
        stay_b[b] += 1
P = trans_b / np.maximum(1.0, stay_b[:, None])   # 子随机: 死亡/结束不转移

# 块"即时"价值 = 块内 β_full 的覆盖加权均值
beta_block = np.zeros(NB)
blk_cov = np.zeros(NB)
for i in range(NP):
    beta_block[p2b[i]] += beta_place[i] * cov[i]
    blk_cov[p2b[i]] += cov[i]
beta_block /= np.maximum(1.0, blk_cov)

# 前瞻递归: R = (I - γ P)^{-1} beta_block; 转点项 = R - beta_block
R = np.linalg.solve(np.eye(NB) - ROT_GAMMA * P, beta_block)
beta_rotation = (R - beta_block)[p2b]         # 点位转点价值 (块级)
beta_terrain = beta_place - beta_rotation     # 点位掩体/地形价值 (残差)

print(f"\nβ 拆解: 粗块 {NB} 个 (边长 {BLOCK:.0f}) | 圈3+ stay 段 {n_stays_late}/{n_stays_all} "
      f"| β_rotation {beta_rotation.min():+.2f}~{beta_rotation.max():+.2f} "
      f"| β_terrain {beta_terrain.min():+.2f}~{beta_terrain.max():+.2f}")
print(f"  校验 β_terrain+β_rotation==β_place: "
      f"{np.abs((beta_terrain + beta_rotation) - beta_place).max():.2e} | "
      f"corr(terrain,rotation)={np.corrcoef(beta_terrain, beta_rotation)[0,1]:+.3f}")

print(f"点位 {NP} | team-game {N} | teams {Tdim} | λ={LAM} | γ=1.0 (非折现)")
print(f"β_p_place  范围: {beta_place.min():+.2f} ~ {beta_place.max():+.2f}")
print(f"kill_slope 范围: {kill_slope.min():.4f} ~ {kill_slope.max():.4f} 击杀/秒 "
      f"(最热 {kill_slope.max()*60:.1f} 头/分钟)")
print(f"φ_place   范围: {min(phi_place.values()):+.2f} ~ {max(phi_place.values()):+.2f}")

ok = cov >= MIN_COV
print(f"\n== 击杀斜率最高 (好打点, 覆盖>={MIN_COV}) ==")
shown = 0
for i in np.argsort(-kill_slope.max(1)):
    if ok[i]:
        bm = int(np.argmax(kill_slope[i]))
        print(f"  ({pos[i][0]:8.0f},{pos[i][1]:8.0f}) "
              f"击杀={kill_slope[i].max():.4f}/秒(圈{bm + 1}) "
              f"排名β={beta_place[i]:+.2f} 覆盖{cov[i]}")
        shown += 1
        if shown >= 10:
            break

print(f"\n== 排名价值最高 (好活点, 覆盖>={MIN_COV}) ==")
shown = 0
for i in np.argsort(-beta_place):
    if ok[i] and shown < 10:
        print(f"  ({pos[i][0]:8.0f},{pos[i][1]:8.0f}) 排名β={beta_place[i]:+.2f} "
              f"击杀={kill_slope[i].max():.4f}/秒 覆盖{cov[i]}")
        shown += 1

# 击杀 vs 排名 的分离度
ks_max = kill_slope.max(1)
print(f"\nkill_slope(max over 圈) 与 β_place 相关性: "
      f"{np.corrcoef(ks_max[ok], beta_place[ok])[0,1]:+.3f}")

np.save("data/beta_kill.npy", kill_slope)
np.save("data/beta_place.npy", beta_place)
np.save("data/beta_terrain.npy", beta_terrain)
np.save("data/beta_rotation.npy", beta_rotation)
np.save("data/points_pos.npy", pos_arr)
json.dump({'phi_place': {f"{a},{b}": v for (a, b), v in phi_place.items()},
           'E_final_kill': {f"{a},{b}": v for (a, b), v in E_final_kill.items()},
           'E_final_place': {f"{a},{b}": v for (a, b), v in E_final_place.items()}},
          open("data/phi.json", "w", encoding='utf-8'))
print("\n已写 data/beta_kill.npy(=击杀斜率, 击杀/秒, 2D), data/beta_place.npy(=β_full), "
      "data/beta_terrain.npy + data/beta_rotation.npy (β 拆解), "
      "data/points_pos.npy, data/phi.json (φ_place + E_final)")