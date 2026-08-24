#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_actual_vs_xt.py — 实际 vs 期望(xT) 散点图 (击杀 / 排名 各一张)

目的: 直观看出 "排名 xT 能对上实际, 击杀 xT 对不上"。
  左: 实际击杀分/场 vs 击杀xT  (散 = 弱相关)
  右: 实际排名分/场 vs 排名xT  (贴线 = 强相关)
标出 Wolves / VK / GenG 三支焦点队伍。
"""
import json
from collections import defaultdict
import numpy as np

LONG = "data/long_table.jsonl"
PHI_JSON = "data/phi.json"
OUT_PNG = "output/actual_vs_xt.png"
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
        a['ak'] += r['kills']
        a['ap'] += r['placement_pts']
        a['games'].add(r['game'])
        if r['t'] >= TMIN:
            if r['rel'] <= 1.0:
                s = (r['phase'], r['rel_bin'])
                a['ek'] += pk.get(s, 0.0)
                a['ep'] += pp.get(s, 0.0)
            a['w'] += 1

    rec = []
    for nm, a in agg.items():
        if len(a['games']) < MIN_GAMES:
            continue
        n = len(a['games'])
        rec.append({'team': nm, 'ak': a['ak'] / n, 'ap': a['ap'] / n,
                    'ek': a['ek'] / a['w'], 'ep': a['ep'] / a['w']})

    teams = {r['team']: r for r in rec}

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

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    def panel(ax, x, y, xlab, ylab, title, highlights):
        xs = [r[x] for r in rec]
        ys = [r[y] for r in rec]
        r_val = np.corrcoef(xs, ys)[0, 1]
        ax.scatter(xs, ys, s=34, c='#b3bcc7', edgecolors='#5b6570',
                   linewidths=0.5, zorder=2)
        # 拟合线
        m, b = np.polyfit(xs, ys, 1)
        xl = np.linspace(min(xs), max(xs), 50)
        ax.plot(xl, m * xl + b, color='#444', lw=1.2, zorder=3,
                label=f'拟合线')
        for nm, color, label in highlights:
            t = teams[nm]
            ax.scatter([t[x]], [t[y]], s=110, c=color, edgecolors='#222',
                       linewidths=1.2, zorder=4)
            ax.annotate(label, (t[x], t[y]), textcoords='offset points',
                        xytext=(8, 6), fontsize=10, fontweight='bold',
                        color=color, zorder=5)
        ax.set_xlabel(xlab, fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(f'{title}\n(r = {r_val:+.3f})', fontsize=11)
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)

    hl_kill = [('wolves esports', '#e45756', 'Wolves'),
               ('vk gaming', '#4c78a8', 'VK'),
               ('geng esports', '#f58518', 'GenG')]
    panel(axes[0], 'ak', 'ek', '实际击杀分 / 场', '击杀xT（期望）',
          '击杀：期望 vs 实际', hl_kill)
    panel(axes[1], 'ap', 'ep', '实际排名分 / 场', '排名xT（期望）',
          '排名：期望 vs 实际', hl_kill)

    fig.suptitle('xT 模型：排名能预测、击杀预测不了', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130, bbox_inches='tight')
    print(f"已写 {OUT_PNG}")


if __name__ == '__main__':
    main()