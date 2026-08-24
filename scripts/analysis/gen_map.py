#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成单场 ALGS 地图可视化 (自包含 HTML + 内联 SVG)。
叠加: POI 多边形 + 圈型(ring) + 队伍落点 + 队伍行动轨迹。
坐标系: 地图像素空间 (0~16384), 由 analyze_game 的 world->pix 转换。
"""
import json, math, html, sys
from analyze_game import load, analyze, world_to_pix, get_ratio

W, H = 16384, 16384

def load_map_json(path):
    d = json.load(open(path, encoding="utf-8"))
    return d

def main(path, out_path):
    d = load_map_json(path)
    r = analyze(d)
    ratio = r["ratio"]

    # 队伍颜色 (来自 summary)
    team_color = {}
    for t in d["summary"]["teams"]:
        team_color[t["teamId"]] = t.get("color", "#888888")
    # pathing 里也有 color
    for t in d["pathing"]["teams"]:
        team_color.setdefault(t["teamId"], t.get("color", "#888888"))

    # 圈 (世界 -> 像素)
    rings_svg = []
    ring_colors = ["#7f7f7f", "#b58900", "#d33682", "#6c71c4", "#dc322f", "#2aa198"]
    for i, ring in enumerate(r["rings"]):
        cx, cy = world_to_pix(ring["center"][0], ring["center"][1], ratio)
        cr = ring["radius"] / ratio * 4
        c = ring_colors[i % len(ring_colors)]
        rings_svg.append((i, cx, cy, cr, c))

    # POI 多边形 + 名字
    pois_svg = []
    for p in r["pois"]:
        pts = [world_to_pix(x, y, ratio) for (x, y) in p["world"]]
        pts_str = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        cx, cy = world_to_pix(p["centroid"][0], p["centroid"][1], ratio)
        pois_svg.append((p["name"], pts_str, cx, cy))

    # 队伍轨迹 (每名玩家一条折线) + 落点
    traj_svg = []   # (teamId, teamName, color, points_pix)
    landing_svg = []
    for t in d["pathing"]["teams"]:
        color = team_color.get(t["teamId"], "#888888")
        name = t["teamName"]
        # 队伍落点 (各玩家首点均值)
        firsts = []
        for pl in t["players"]:
            ps = pl.get("points") or []
            if ps:
                p0 = min(ps, key=lambda p: p["tsGame"])
                firsts.append((p0["x"], p0["y"]))
            # 玩家轨迹
            if ps:
                pts_pix = [world_to_pix(p["x"], p["y"], ratio) for p in ps]
                traj_svg.append((t["teamId"], name, color, pts_pix))
        if firsts:
            lx = sum(p[0] for p in firsts) / len(firsts)
            ly = sum(p[1] for p in firsts) / len(firsts)
            lpx, lpy = world_to_pix(lx, ly, ratio)
            landing_svg.append((t["teamId"], name, color, lpx, lpy))

    # 组装 SVG
    parts = []
    # 背景
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="var(--map-bg)"/>')

    # POI 多边形
    for (name, pts_str, cx, cy) in pois_svg:
        parts.append(f'<polygon points="{pts_str}" fill="var(--poi-fill)" stroke="var(--poi-stroke)" stroke-width="2"/>')
        parts.append(f'<text x="{cx:.1f}" y="{cy:.1f}" class="poi-label">{html.escape(name)}</text>')

    # 圈
    for (i, cx, cy, cr, c) in rings_svg:
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{cr:.1f}" fill="none" stroke="{c}" stroke-width="2" stroke-dasharray="8 6" opacity="0.85"/>')
        parts.append(f'<text x="{cx:.1f}" y="{cy - cr - 6:.1f}" class="ring-label" fill="{c}">圈{i+1}</text>')

    # 轨迹 (先画, 在圈下面? 不, 轨迹要在最上层才看得见; 但圈要能透出轨迹 -> 轨迹在上, 圈在轨迹之下)
    # 这里把轨迹放最后画, 但用半透明
    for (tid, name, color, pts) in traj_svg:
        pts_str = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        parts.append(f'<polyline class="traj" data-team="{tid}" points="{pts_str}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round" opacity="0.55"/>')

    # 落点
    for (tid, name, color, lpx, lpy) in landing_svg:
        parts.append(f'<circle class="landing" data-team="{tid}" cx="{lpx:.1f}" cy="{lpy:.1f}" r="8" fill="{color}" stroke="#fff" stroke-width="1.5"/>')

    svg = "\n".join(parts)
    inner = f'<svg viewBox="0 0 {W} {H}" class="map" id="map" preserveAspectRatio="xMidYMid meet">{svg}</svg>'

    # 图例
    legend_items = []
    for t in sorted(d["pathing"]["teams"], key=lambda x: x["teamId"]):
        c = team_color.get(t["teamId"], "#888888")
        legend_items.append(f'<span class="leg" data-team="{t["teamId"]}"><i style="background:{c}"></i>{html.escape(t["teamName"])}</span>')
    legend = '<div class="legend">' + "\n".join(legend_items) + "</div>"

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(d['summary']['headerDisplayName'])} — 地图轨迹</title>
<style>
:root{{
  --bg:#0f1115; --panel:#171a21; --ink:#e6e8ec; --muted:#9aa0a8;
  --map-bg:#14161c; --poi-fill:#1d2129; --poi-stroke:#3a4150;
}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;}}
header{{padding:16px 20px;border-bottom:1px solid #262b36;display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap;}}
header h1{{margin:0;font-size:16px;font-weight:600;}}
header .meta{{color:var(--muted);font-size:12px;}}
.wrap{{display:flex;flex-direction:column;}}
.map{{width:100%;max-width:1000px;display:block;margin:0 auto;}}
.map polygon{{transition:fill .15s;}}
.map text{{font-size:22px;fill:var(--muted);text-anchor:middle;pointer-events:none;}}
.map .ring-label{{font-size:26px;font-weight:700;}}
.legend{{padding:14px 20px;display:flex;flex-wrap:wrap;gap:6px 16px;justify-content:center;}}
.legend .leg{{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--ink);cursor:pointer;padding:2px 6px;border-radius:6px;user-select:none;}}
.legend .leg i{{width:10px;height:10px;border-radius:50%;display:inline-block;}}
.legend .leg.off{{opacity:.25;}}
.tip{{position:fixed;pointer-events:none;background:#000c;color:#fff;padding:4px 8px;border-radius:6px;font-size:12px;display:none;z-index:10;}}
</style>
</head>
<body>
<header>
  <h1>{html.escape(d['summary']['headerDisplayName'])}</h1>
  <div class="meta">圈型 + POI + 队伍行动轨迹 &nbsp;|&nbsp; 点击图例可高亮/隐藏队伍</div>
</header>
<div class="wrap">
{inner}
{legend}
</div>
<div class="tip" id="tip"></div>
<script>
const tip=document.getElementById('tip');
// 高亮单个队伍: 该队轨迹加粗, 其余淡出
document.querySelectorAll('.legend .leg').forEach(el=>{{
  el.addEventListener('click',()=>{{
    const on=el.classList.toggle('off');
    const tid=el.dataset.team;
    document.querySelectorAll(`.traj[data-team="${{tid}}"], .landing[data-team="${{tid}}"]`).forEach(s=>{{
      s.style.opacity=on?'0.06':'0.9';
      s.style.strokeWidth=on?'1.5':'3';
    }});
  }});
}});
// 悬停显示队伍名
document.querySelectorAll('.traj').forEach(s=>{{
  s.addEventListener('mousemove',e=>{{
    const t=[...document.querySelectorAll('.legend .leg')].find(x=>x.dataset.team===s.dataset.team);
    tip.textContent=t?t.textContent.trim():'';
    tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';
  }});
  s.addEventListener('mouseleave',()=>{{tip.style.display='none';}});
}});
</script>
</body>
</html>"""
    open(out_path, "w", encoding="utf-8").write(page)
    print(f"生成 {out_path}  ({len(d['pathing']['teams'])} 队, {sum(len(t['players']) for t in d['pathing']['teams'])} 名玩家, {len(r['rings'])} 圈, {len(r['pois'])} POI)")

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "/mnt/j/rank/replay2_99c6fb4194f5d70ed78b03f481fe508f.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "/mnt/j/rank/output/map_game.html"
    main(src, out)
