#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_rotation_spike.py — 全量 59 场里找「转点 xT_place 跳变」最大的一次

用途: 给 xT 介绍视频找 hook 素材——一段「队伍转点、xT 突然跳升」的片段。

口径 (与 team_xT_trajectory.py / points_split.py 完全一致):
  xT_place(row) = max(0,  w_zone * w_stage * β_p[j]  +  φ_place[(phase, rel_bin)] )
    其中 j = 锚点(ax,ay) 最近点位; w_zone / w_stage / phase / rel_bin 已存在 long_table 行里。

「转点」= 连续两个 5s 桶之间最近点位 j 发生变化 (队伍锚点跨到了另一个点位簇)。
输出:
  1) 转点跳变 Top 20 (按 xT_place 前后差排序), 附 β_p 项 / φ 项 分解与前后点位坐标、β 值。
  2) 不做「转点」过滤的全量最大跳变 Top 10 (对照: 圈收缩/进圈造成的自然抬升)。
"""
import json
import numpy as np

REL_BINS = [0.3, 0.7, 1.0, 1.15, 1.6]
LONG = "data/long_table.jsonl"
POS = "data/points_pos.npy"
PHI = "data/phi.json"


def game_label(name):
    import re
    b = name.replace(".json", "")
    m = re.match(r"sp_(.+?)_d(.*?)_(g\d+)_", b)
    if not m:
        return b
    region = {"apac-s": "亚太南", "apac-n": "亚太北", "global": "世界赛",
              "na": "北美", "emea": "欧洲"}.get(m.group(1), m.group(1))
    day = m.group(2).strip("_")
    day_s = f" Day{day}" if day else ""
    return f"{region}{day_s} {m.group(3).replace('g', 'Game#')}"


def mmss(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def main():
    bp = np.load("data/beta_place.npy")
    pos = np.load(POS)
    NP = len(bp)
    phi = json.load(open(PHI, encoding='utf-8'))
    pp = {(int(a), int(b)): v for a, b, v in
          (k.split(',') + [v] for k, v in phi['phi_place'].items())}

    from scipy.spatial import cKDTree
    tree = cKDTree(pos)

    # 逐行: 点位 j + xT_place + β_p项 / φ项
    rows = []
    with open(LONG, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            j = int(tree.query([[r['ax'], r['ay']]])[1][0])
            beta_term = r['w_zone'] * r['w_stage'] * bp[j]
            phi_term = pp[(r['phase'], r['rel_bin'])]
            xp = max(0.0, beta_term + phi_term)
            rows.append(dict(r, j=j, beta_term=beta_term, phi_term=phi_term, xp=xp))

    # 按 (game, team, t) 排序, 扫连续桶
    rows.sort(key=lambda r: (r['game'], r['team'], r['t']))
    n = len(rows)

    rot_jumps = []   # 转点(点位变化)跳变
    all_jumps = []   # 所有连续桶跳变(对照)

    for i in range(n - 1):
        a, b = rows[i], rows[i + 1]
        if (a['game'], a['team']) != (b['game'], b['team']):
            continue
        dj = b['xp'] - a['xp']
        if dj <= 0:
            continue
        rec = {
            'game': a['game'], 'label': game_label(a['game']),
            'team': a['team'], 'team_name': a['team_name'],
            't': b['t'], 't_str': mmss(b['t']),
            'phase': b['phase'],  # 圈 stage (0=圈1)
            'jump': dj,
            'dbeta': b['beta_term'] - a['beta_term'],
            'dphi': b['phi_term'] - a['phi_term'],
            'xp_prev': a['xp'], 'xp_next': b['xp'],
            'j_prev': a['j'], 'j_next': b['j'],
            'beta_prev': bp[a['j']], 'beta_next': bp[b['j']],
            'pos_prev': tuple(pos[a['j']]), 'pos_next': tuple(pos[b['j']]),
            'wz_prev': a['w_zone'], 'wz_next': b['w_zone'],
            'ws_prev': a['w_stage'], 'ws_next': b['w_stage'],
            'rel_prev': a['rel'], 'rel_next': b['rel'],
            'rb_prev': a['rel_bin'], 'rb_next': b['rel_bin'],
        }
        all_jumps.append(rec)
        if a['j'] != b['j']:
            rot_jumps.append(rec)

    rot_jumps.sort(key=lambda r: -r['jump'])
    all_jumps.sort(key=lambda r: -r['jump'])

    print(f"全量 {len(rows)} 行 | 点位 {NP} | 正跳变 {len(all_jumps)} 次 | 转点正跳变 {len(rot_jumps)} 次\n")

    def dump(rec):
        print(f"  [{rec['label']}] {rec['team_name']:<12} t={rec['t_str']} 圈{rec['phase']+1}  "
              f"跳变 {rec['jump']:+.2f}")
        print(f"      xT_place {rec['xp_prev']:.2f} → {rec['xp_next']:.2f}   "
              f"(β_p项 {rec['dbeta']:+.2f} + φ项 {rec['dphi']:+.2f})")
        print(f"      点#{rec['j_prev']}({rec['pos_prev'][0]:.0f},{rec['pos_prev'][1]:.0f}) β={rec['beta_prev']:+.2f} "
              f"→ 点#{rec['j_next']}({rec['pos_next'][0]:.0f},{rec['pos_next'][1]:.0f}) β={rec['beta_next']:+.2f}")
        print(f"      w_zone {rec['wz_prev']:.2f}→{rec['wz_next']:.2f} | w_stage {rec['ws_prev']:.2f}→{rec['ws_next']:.2f} "
              f"| rel {rec['rel_prev']:.2f}→{rec['rel_next']:.2f} (档{rec['rb_prev']}→{rec['rb_next']})")

    print("== ① 转点 xT_place 跳变 Top 20 ==")
    for rec in rot_jumps[:20]:
        dump(rec)

    print("\n== ② 所有连续桶正跳变 Top 10 (对照: 含进圈/圈收缩抬升) ==")
    for rec in all_jumps[:10]:
        dump(rec)


if __name__ == '__main__':
    main()
