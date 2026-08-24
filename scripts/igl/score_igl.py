#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_igl.py — IGL 评分 (亚太南), xT 分解版: 击杀xT / 排名xT / 合计xT

xT 按奖励通道线性分解 (见 scripts/phi_split.py):
    V(s) = V_kill(s) + V_place(s)
    击杀xT  = 该队轨迹上"圈位击杀价值 φ_kill"的均值 (期望未来击杀分)
    排名xT  = 该队轨迹上"圈位排名价值 φ_place"的均值 (期望未来排名分)
    合计xT  = 击杀xT + 排名xT (期望未来总分)

口径同前: 搜完物资(t>=120s)后全轨迹, 毒里(rel>1)清零。
"""
import json, glob, os
from collections import defaultdict
import numpy as np

LONG = "data/long_table.jsonl"
PHI_JSON = "data/phi.json"
OUT_CSV = "output/igl_apac_s.csv"
OUT_PNG = "output/igl_apac_s.png"
TMIN = 120
MIN_GAMES = 3


def load_phi():
    j = json.load(open(PHI_JSON, encoding='utf-8'))
    def tbl(name):
        return {(int(k.split(',')[0]), int(k.split(',')[1])): v
                for k, v in j[name].items()}
    return tbl('phi_kill'), tbl('phi_place'), tbl('phi')


def main():
    pk, pp, pt = load_phi()

    agg = defaultdict(lambda: {'k': 0.0, 'p': 0.0, 't': 0.0, 'w': 0, 'games': set()})
    for line in open(LONG, encoding='utf-8'):
        r = json.loads(line)
        if r['region'] != 'apac-s' or r['t'] < TMIN:
            continue
        nm = r['team_name'].lower()
        a = agg[nm]
        if r['rel'] <= 1.0:
            s = (r['phase'], r['rel_bin'])
            a['k'] += pk.get(s, 0.0)
            a['p'] += pp.get(s, 0.0)
            a['t'] += pt.get(s, 0.0)
        a['w'] += 1
        a['games'].add(r['game'])

    rec = []
    for nm, a in agg.items():
        if a['w'] == 0:
            continue
        rec.append({'team': nm, 'n_games': len(a['games']),
                    'kill': a['k'] / a['w'], 'place': a['p'] / a['w'],
                    'total': a['t'] / a['w']})
    rec.sort(key=lambda x: -x['total'])

    with open(OUT_CSV, 'w', encoding='utf-8') as f:
        f.write("rank,team,games,kill_xT,place_xT,total_xT\n")
        for i, r in enumerate(rec):
            f.write(f"{i+1},{r['team']},{r['n_games']},"
                    f"{r['kill']:.3f},{r['place']:.3f},{r['total']:.3f}\n")

    print(f"亚太南队伍 {len(rec)} 支 | xT 分解 (圈位价值均值, 搜物资后全轨迹):")
    print(f"{'#':>2} {'队伍':<20} {'场':>2} {'击杀xT':>8} {'排名xT':>8} {'合计xT':>8}")
    for i, r in enumerate(rec):
        flag = "  *" if r['n_games'] < MIN_GAMES else ""
        print(f"{i+1:>2} {r['team']:<20} {r['n_games']:>2} {r['kill']:>8.3f} "
              f"{r['place']:>8.3f} {r['total']:>8.3f}{flag}")
    print("  * = 场次 < 3, 样本少, 排名噪声大")

    # ---- 堆叠条形图: 排名xT(蓝) + 击杀xT(红) ----
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

    top = [r for r in rec if r['n_games'] >= MIN_GAMES]
    names = [r['team'] for r in top]
    place = [r['place'] for r in top]
    kill = [r['kill'] for r in top]
    totals = [r['total'] for r in top]

    fig, ax = plt.subplots(figsize=(8, 0.30 * len(top) + 2.4))
    y = np.arange(len(top))
    c_place, c_kill = '#4c78a8', '#e45756'
    ax.barh(y, place, color=c_place, edgecolor='#222', height=0.7,
            label='排名xT (期望排名分)')
    ax.barh(y, kill, left=place, color=c_kill, edgecolor='#222', height=0.7,
            label='击杀xT (期望击杀分)')
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('xT = 击杀xT + 排名xT (圈位期望未来分, 搜物资后全轨迹)', fontsize=10)
    ax.set_title('亚太南队伍 xT 分解 (击杀 / 排名 分开 + 合计)', fontsize=12)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    for yi, t in zip(y, totals):
        ax.text(t + 0.02, yi, f'{t:.2f}', va='center', ha='left', fontsize=8)
    ax.set_xlim(0, max(totals) * 1.12)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130)
    print(f"\n已写 {OUT_CSV} | {OUT_PNG} (主榜 {len(top)} 支, 场次>={MIN_GAMES})")


if __name__ == '__main__':
    main()