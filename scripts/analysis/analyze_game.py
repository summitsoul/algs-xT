#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析单场 ALGS replay 数据 (apexlegendsstatus.com 抓取的 replay2_*.json)。
核心: 队伍落点 POI 判定 + 圈型(ring) + 落点到圈中心距离(圈运) + 轨迹统计。

坐标系说明 (关键):
  - pathing / rings / deathboxes / heatmap / events.posActor 都是「Apex 世界坐标」(x,y 可为负, 跨度大)
  - pois.payload.polygons[].coordinates 是「地图像素坐标」(0~16384)
  - 世界 -> 像素: x_pix = round(x_world/ratio*4 + 8192);  y_pix = round(-y_world/ratio*4 + 8192)
  - 像素 -> 世界: x_world = (x_pix - 8192)*ratio/4;         y_world = -(y_pix - 8192)*ratio/4
  - ratio: Storm Point(tropic)=24.93, Broken Moon=21.05, Olympus=22.01, World's Edge(desertlands)=21.97, KC(canyon)=20, E-District=21
"""
import json, sys, math
from collections import defaultdict

RATIOS = {"tropic": 24.93, "storm": 24.93, "moon": 21.05, "divided": 21.05,
          "olympus": 22.01, "desertlands": 21.97, "edge": 21.97, "canyon": 20.0}
FALLBACK_RATIO = 21.0

def get_ratio(mapname):
    m = (mapname or "").lower()
    for k, v in RATIOS.items():
        if k in m:
            return v
    return FALLBACK_RATIO

def world_to_pix(x, y, ratio):
    return (x / ratio * 4 + 8192, -y / ratio * 4 + 8192)

def pix_to_world(px, py, ratio):
    return ((px - 8192) * ratio / 4, -(py - 8192) * ratio / 4)

def point_in_polygon(pt, poly):
    """ray-casting. pt=(x,y), poly=[(x,y),...] 世界坐标."""
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def poly_centroid(poly):
    x = sum(p[0] for p in poly) / len(poly)
    y = sum(p[1] for p in poly) / len(poly)
    return (x, y)

def load(path):
    d = json.load(open(path, encoding="utf-8"))
    return d

def analyze(d):
    summary = d["summary"]
    mapname = summary.get("map", "")
    ratio = get_ratio(mapname)

    # --- POI: 像素 -> 世界, 保留名字 + 世界多边形 + 质心 ---
    pois = []
    for poly in (d["pois"]["payload"].get("polygons") or []):
        name = poly["name"]
        coords_pix = [(c["x"], c["y"]) for c in poly["coordinates"]]
        coords_world = [pix_to_world(px, py, ratio) for (px, py) in coords_pix]
        pois.append({"name": name, "world": coords_world, "centroid": poly_centroid(coords_world)})

    # --- 圈中心 (finishedClosing 即该圈最终中心) ---
    rings = []
    for r in summary.get("ringPhases") or []:
        if r.get("type") == "finishedClosing":
            rings.append({"stage": r["stage"],
                          "center": (r["center"]["x"], r["center"]["y"]),
                          "radius": r.get("currentRadius")})

    # --- 每队落点位置 + 落点 POI ---
    def team_landing_pos(team):
        # 取每名玩家第一个点(tsGame 最小)的平均
        pts = []
        for pl in team.get("players", []):
            ps = pl.get("points") or []
            if ps:
                p0 = min(ps, key=lambda p: p["tsGame"])
                pts.append((p0["x"], p0["y"]))
        if not pts:
            return None
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    def assign_poi(pos):
        if pos is None:
            return None
        # 先找包含该点的 POI
        for p in pois:
            if point_in_polygon(pos, p["world"]):
                return p
        # 否则找质心最近的
        best = min(pois, key=lambda p: (p["centroid"][0]-pos[0])**2 + (p["centroid"][1]-pos[1])**2)
        return best

    def team_travel_dist(team):
        d = 0.0
        for pl in team.get("players", []):
            ps = pl.get("points") or []
            for a, b in zip(ps, ps[1:]):
                d += math.hypot(b["x"]-a["x"], b["y"]-a["y"])
        return d

    teams = []
    for t in d["pathing"]["teams"]:
        pos = team_landing_pos(t)
        poi = assign_poi(pos)
        # 名次/击杀从 summary 里取
        st = next((x for x in summary["teams"] if x["teamId"] == t["teamId"]), None)
        placement = st["placement"] if st else None
        kills = (st["stats"]["kills"] if st else 0)
        teams.append({
            "teamId": t["teamId"], "teamName": t["teamName"],
            "landing": pos, "poi": poi["name"] if poi else None,
            "placement": placement, "kills": kills,
            "travel": team_travel_dist(t),
            "ring_dists": [math.hypot(pos[0]-r["center"][0], pos[1]-r["center"][1]) for r in rings] if pos else [],
        })

    return {"map": mapname, "ratio": ratio, "pois": pois, "rings": rings, "teams": teams}

def fmt_table(result):
    rings = result["rings"]
    hdr = f"{'名次':>3} {'队伍':<16} {'落点POI':<16} {'击杀':>3} {'行程':>7} |"
    for r in rings:
        hdr += f" 圈{r['stage']+1}距"
    print(hdr)
    print("-" * len(hdr))
    for t in sorted(result["teams"], key=lambda x: (x["placement"] or 99)):
        rd = t["ring_dists"]
        rds = " ".join(f"{d/1000:5.1f}k" for d in rd) if rd else "-"
        print(f"{t['placement']:>3} {t['teamName']:<16} {t['poi'] or '?':<16} {t['kills']:>3} {t['travel']/1000:6.1f}k | {rds}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/j/rank/replay2_99c6fb4194f5d70ed78b03f481fe508f.json"
    d = load(path)
    r = analyze(d)
    print(f"地图: {r['map']}  ratio={r['ratio']}")
    print(f"圈数: {len(r['rings'])}  圈中心(世界):")
    for ring in r["rings"]:
        print(f"  圈{ring['stage']+1}: 中心({ring['center'][0]:.0f},{ring['center'][1]:.0f}) 半径{ring['radius']:.0f}")
    print(f"\nPOI 数量: {len(r['pois'])}")
    print()
    fmt_table(r)
