#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_xT.py — 读长表, 联合建模 (22 场 Storm Point)

两级结构:
  Stage 1  环形 MDP: 状态 s = (第几圈 phase, 圈相对 rel_bin)
           Bellman: V(s) = k(s) + γ[(1-h)E[V(s')] + h·pl(s)]
           -> 得到 xT 值 V(s) (γ 折现), 可解释分量 k/h/pl
  Stage 2  地形残差: 绝对位置(踩的位置, 粗网格)的价值
           g(grid) = mean( 最终分 - E_final[phase,rel_bin] )
           E_final[state] = 该状态下的期望最终分 (不折现, 用于和真实终分对比)

验证: V 单调性 sanity + 回测相关性(期望 vs 实际终分)
IGL: 每队 residual = 最终分 - (环形期望 + 地形期望), 正值=超常发挥
"""
import json, math
from collections import defaultdict

GRID = 16
GAMMA = 0.99
REL_NAMES = ['圈心', '圈内', '圈内边缘', '圈外贴边', '圈外', '远圈外']
LONG = "data/long_table.jsonl"
META = "data/long_table.meta.json"

rows = [json.loads(l) for l in open(LONG, encoding='utf-8')]
meta = json.load(open(META))
bnd = meta['bounds']
xmin, xmax, ymin, ymax = bnd['xmin'], bnd['xmax'], bnd['ymin'], bnd['ymax']
N = len(rows)


def grid(ax, ay):
    gx = int((ax - xmin) / (xmax - xmin) * GRID)
    gy = int((ay - ymin) / (ymax - ymin) * GRID)
    return min(max(gx, 0), GRID - 1), min(max(gy, 0), GRID - 1)


# ---- 每队每场最终分 (拆击杀/排名) ----
final = defaultdict(float)
kp = defaultdict(float)   # 击杀分
pp = defaultdict(float)   # 排名分
for r in rows:
    final[(r['game'], r['team'])] += r['kills'] + r['placement_pts']
    kp[(r['game'], r['team'])] += r['kills']
    pp[(r['game'], r['team'])] += r['placement_pts']

# ---- Stage 1: MDP (phase, rel_bin) ----
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
it = 0
for it in range(5000):
    Vn, delta = {}, 0.0
    for s in states:
        fut = sum(P[s].get(s2, 0) * V[s2] for s2 in P[s]) if P[s] else 0.0
        Vn[s] = k[s] + GAMMA * ((1 - h[s]) * fut + h[s] * pl[s])
        delta = max(delta, abs(Vn[s] - V[s]))
    V = Vn
    if delta < 1e-6:
        break

# E_final (不折现期望最终分)
ef_sum = defaultdict(float)
for r in rows:
    ef_sum[(r['phase'], r['rel_bin'])] += final[(r['game'], r['team'])]
E_final = {s: ef_sum[s] / visits[s] for s in states}

# ---- Stage 2: 地形残差 g(grid) ----
g_sum = defaultdict(float)
g_cnt = defaultdict(int)
for r in rows:
    s = (r['phase'], r['rel_bin'])
    resid = final[(r['game'], r['team'])] - E_final[s]
    gx, gy = grid(r['ax'], r['ay'])
    g_sum[(gx, gy)] += resid
    g_cnt[(gx, gy)] += 1
g = {(gx, gy): g_sum[(gx, gy)] / g_cnt[(gx, gy)] for (gx, gy) in g_cnt}

# ---- 输出 1: V 表 + sanity ----
print("=" * 78)
print(f"Stage 1 环形 xT 值 V(s) | 状态数 {len(states)} | 迭代 {it} 次 (delta={delta:.2e}) | γ={GAMMA}")
print(f"表头: 样本/击杀k/淘汰率h/终态pl/V(xT)   [不折现期望终分 E_final]")
print("=" * 78)
for ph in range(6):
    line = f"圈{ph+1}: "
    for rb in range(len(REL_NAMES)):
        s = (ph, rb)
        if s in visits:
            line += (f"{REL_NAMES[rb]}[n={visits[s]} k={k[s]:.2f} h={h[s]:.2f} "
                     f"pl={pl[s]:.1f} V={V[s]:.2f}/E={E_final[s]:.1f}]  ")
    print(line)

print("\n=== sanity: 圈内(rel<1) vs 圈外贴边(1.0-1.15) vs 远圈外(>1.6) 的分化 ===")
for ph in range(6):
    vals = [V.get((ph, rb), float('nan')) for rb in range(len(REL_NAMES))]
    print(f"圈{ph+1}: " + " | ".join(f"{REL_NAMES[i]}={v:.2f}" for i, v in enumerate(vals)))

# ---- 输出 2: 地形残差 ----
print("\n" + "=" * 78)
print(f"Stage 2 地形残差 g(踩的位置, {GRID}x{GRID} 网格): 值 = 该位置额外拿分(超出环形期望)")
print("=" * 78)
gcells = [(gv, (gx, gy), g_cnt[(gx, gy)]) for (gx, gy), gv in g.items() if g_cnt[(gx, gy)] >= 30]
gcells.sort(reverse=True)
print("  最值钱的位置 (top 10):")
for gv, (gx, gy), c in gcells[:10]:
    print(f"    ({gx:2d},{gy:2d}) n={c:4d} 地形残差={gv:+.2f}")
print("  最废的位置 (bottom 8):")
for gv, (gx, gy), c in gcells[-8:]:
    print(f"    ({gx:2d},{gy:2d}) n={c:4d} 地形残差={gv:+.2f}")

# 保存地形 CSV
with open("data/terrain.csv", "w") as f:
    f.write("gx,gy,value,count\n")
    for (gx, gy), gv in g.items():
        f.write(f"{gx},{gy},{gv:.4f},{g_cnt[(gx, gy)]}\n")

# ---- 输出 3: 回测相关性 ----
print("\n" + "=" * 78)
print("回测: 模型期望分 vs 真实最终分 的相关性 (每队每场一个点)")
print("=" * 78)
teams_grp = defaultdict(list)  # (game,team) -> rows
for r in rows:
    teams_grp[(r['game'], r['team'])].append(r)

obs = []  # (ring_exp, terr_exp, total_exp, actual)
for (gid, tid), rws in teams_grp.items():
    ring_exp = sum(E_final[(r['phase'], r['rel_bin'])] for r in rws) / len(rws)
    terr_exp = sum(g[grid(r['ax'], r['ay'])] for r in rws) / len(rws)
    obs.append((ring_exp, terr_exp, ring_exp + terr_exp, final[(gid, tid)]))


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return cov / math.sqrt(va * vb) if va and vb else 0.0


ring_a, terr_a, tot_a, act = zip(*obs)
print(f"  环形期望 vs 终分:  r = {pearson(ring_a, act):.3f}")
print(f"  地形期望 vs 终分:  r = {pearson(terr_a, act):.3f}")
print(f"  总期望   vs 终分:  r = {pearson(tot_a, act):.3f}")

# ---- E_kill / E_placement (环形期望拆成击杀/排名两条通道) ----
ek_sum = defaultdict(float)
ep_sum = defaultdict(float)
for r in rows:
    s = (r['phase'], r['rel_bin'])
    ek_sum[s] += kp[(r['game'], r['team'])]
    ep_sum[s] += pp[(r['game'], r['team'])]
E_kill = {s: ek_sum[s] / visits[s] for s in states}
E_pl = {s: ep_sum[s] / visits[s] for s in states}

# ---- 输出 4: IGL 评分 (拆击杀/排名, 过滤单场噪声) ----
MIN_GAMES = 3
print("\n" + "=" * 78)
print(f"IGL 评分 (环形模型; 只显示 >= {MIN_GAMES} 场的队, 单场噪声太大)")
print("  击杀残差 = 实际击杀分 - 位置期望击杀分   [>0 = 能打架]")
print("  排名残差 = 实际排名分 - 位置期望排名分   [>0 = 会占位/活得久 = IGL]")
print("=" * 78)
by_name = defaultdict(lambda: {'games': set(), 'kill': 0.0, 'pl': 0.0})
for (gid, tid), rws in teams_grp.items():
    nm = rws[0]['team_name']
    ke = sum(E_kill[(r['phase'], r['rel_bin'])] for r in rws) / len(rws)
    pe = sum(E_pl[(r['phase'], r['rel_bin'])] for r in rws) / len(rws)
    d = by_name[nm]
    d['games'].add(gid)
    d['kill'] += kp[(gid, tid)] - ke
    d['pl'] += pp[(gid, tid)] - pe

ranking = [(nm, len(d['games']), d['kill'] / len(d['games']), d['pl'] / len(d['games']))
           for nm, d in by_name.items()]
multi = [x for x in ranking if x[1] >= MIN_GAMES]

print(f"{'队伍':<22}{'场':>3}{'击杀残差':>9}{'排名残差':>9}{'总残差':>9}")
multi.sort(key=lambda x: -(x[2] + x[3]))
for nm, ng, kr, pr in multi:
    print(f"{nm:<22}{ng:>3}{kr:>+9.2f}{pr:>+9.2f}{kr+pr:>+9.2f}")

print(f"\n按'排名残差'排 (定位/存活 = IGL 核心能力):")
multi.sort(key=lambda x: -x[3])
for nm, ng, kr, pr in multi[:12]:
    print(f"  {nm:<22}{ng:>3}  击杀{kr:+.2f}  排名{pr:+.2f}")

# 保存 IGL 排行 (含单场队)
with open("data/igl_ranking.csv", "w") as f:
    f.write("team,games,kill_residual,placement_residual,total_residual\n")
    for nm, ng, kr, pr in ranking:
        f.write(f"{nm},{ng},{kr:.4f},{pr:.4f},{kr + pr:.4f}\n")

print("\n已保存: data/terrain.csv, data/igl_ranking.csv, data/v_ring.json")
json.dump({'V': {f"{a},{b}": v for (a, b), v in V.items()},
           'E_final': {f"{a},{b}": v for (a, b), v in E_final.items()}},
          open("data/v_ring.json", "w"))
