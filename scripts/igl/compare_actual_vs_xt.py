#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_actual_vs_xt.py — 实际(击杀分/排名分) vs 期望(xT) 对比

每队两套数字并排:
  实际 = 该队真实拿到的击杀分 / 排名分 (场均)
  期望 = 该队轨迹圈位的 φ_kill / φ_place 均值 (xT 预测的击杀/排名价值)

目的: 看 xT 的"期望击杀"和"期望排名"能不能对得上"实际"。
"""
import json
from collections import defaultdict
import numpy as np

LONG = "data/long_table.jsonl"
PHI_JSON = "data/phi.json"
OUT_CSV = "output/actual_vs_xt.csv"
TMIN = 120
MIN_GAMES = 3


def load_phi():
    j = json.load(open(PHI_JSON, encoding='utf-8'))
    def tbl(name):
        return {(int(k.split(',')[0]), int(k.split(',')[1])): v
                for k, v in j[name].items()}
    return tbl('phi_kill'), tbl('phi_place')


def main():
    pk, pp = load_phi()

    agg = defaultdict(lambda: {'ak': 0.0, 'ap': 0.0, 'ek': 0.0, 'ep': 0.0,
                               'w': 0, 'games': set()})
    for line in open(LONG, encoding='utf-8'):
        r = json.loads(line)
        if r['region'] != 'apac-s':
            continue
        nm = r['team_name'].lower()
        a = agg[nm]
        a['ak'] += r['kills']          # 实际击杀分 (全时段)
        a['ap'] += r['placement_pts']  # 实际排名分 (死亡桶)
        a['games'].add(r['game'])
        if r['t'] >= TMIN:
            if r['rel'] <= 1.0:
                s = (r['phase'], r['rel_bin'])
                a['ek'] += pk.get(s, 0.0)
                a['ep'] += pp.get(s, 0.0)
            a['w'] += 1

    rec = []
    for nm, a in agg.items():
        if not a['games']:
            continue
        n = len(a['games'])
        rec.append({'team': nm, 'n': n,
                    'ak': a['ak'] / n, 'ap': a['ap'] / n,
                    'ek': a['ek'] / max(a['w'], 1), 'ep': a['ep'] / max(a['w'], 1)})
    rec.sort(key=lambda x: -(x['ak'] + x['ap']))

    sub = [r for r in rec if r['n'] >= MIN_GAMES]
    r_kill = np.corrcoef([r['ak'] for r in sub], [r['ek'] for r in sub])[0, 1]
    r_place = np.corrcoef([r['ap'] for r in sub], [r['ep'] for r in sub])[0, 1]

    with open(OUT_CSV, 'w', encoding='utf-8') as f:
        f.write("team,games,actual_kill_pg,xT_kill,actual_place_pg,xT_place,"
                "actual_total_pg,xT_total\n")
        for r in rec:
            f.write(f"{r['team']},{r['n']},{r['ak']:.2f},{r['ek']:.3f},"
                    f"{r['ap']:.2f},{r['ep']:.3f},{r['ak']+r['ap']:.2f},"
                    f"{r['ek']+r['ep']:.3f}\n")

    print(f"亚太南 {len(rec)} 支 | 实际 vs 期望 (xT)")
    print(f"相关性: 实际击杀 vs 击杀xT  r={r_kill:+.3f}   "
          f"实际排名分 vs 排名xT  r={r_place:+.3f}\n")
    print(f"{'队伍':<20} {'实际击杀':>7} {'击杀xT':>7} | {'实际排名':>7} {'排名xT':>7} | "
          f"{'实际总':>6} {'xT总':>6}")
    for r in rec:
        flag = "  *" if r['n'] < MIN_GAMES else ""
        print(f"{r['team']:<20} {r['ak']:>7.2f} {r['ek']:>7.3f} | "
              f"{r['ap']:>7.2f} {r['ep']:>7.3f} | "
              f"{r['ak']+r['ap']:>6.2f} {r['ek']+r['ep']:>6.3f}{flag}")
    print("  * = 场次 < 3")
    print(f"\n已写 {OUT_CSV}")


if __name__ == '__main__':
    main()