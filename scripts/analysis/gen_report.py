#!/usr/bin/env python3
"""生成自包含的 HTML 分析报告（内联 SVG 图表 + 深色模式 + hover 提示）。"""
import json
import glob
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output"

# ---- palette (dataviz 参考实例) ----
C = {
    "blue": "#2a78d6", "blue_dark": "#1c5cab", "blue_light": "#86b6ef",
    "orange": "#eb6834", "aqua": "#1baf7a",
    "surface": "#fcfcfb", "ink": "#0b0b0b", "sec": "#52514e",
    "muted": "#898781", "grid": "#e1e0d9", "baseline": "#c3c2b7",
}


# ============ 数据计算 ============

def spearman(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0, n
    def ranks(v):
        o = sorted(range(n), key=lambda i: v[i])
        r = [0] * n
        for p, i in enumerate(o):
            r[i] = p + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n))
    vy = sum((ry[i] - my) ** 2 for i in range(n))
    return cov / (vx * vy) ** 0.5, n


def compute():
    # 每队 Round1 组内 rank -> points/placement/kills
    rows = []  # (split, rank, points, placement, kills, position, map_pairs)
    for f in glob.glob(str(RAW / "*.json")):
        rec = json.load(open(f))
        s = rec.get("series") or {}
        if s.get("status") != "completed":
            continue
        stats = rec.get("stats") or {}
        picks = (rec.get("picks") or {}).get("picks", [])
        if not picks:
            continue
        slug = rec["slug"]
        is_rf = s.get("name") == "Regional Finals"
        grp = {t["teamId"]: t.get("group") for t in s.get("teams", [])}
        st = {t["id"]: t for t in stats.get("teams", [])}
        split = "split-1" if slug.startswith("split-1") else "split-2"
        if is_rf:
            continue
        picks = sorted(picks, key=lambda x: x["pickNumber"])
        r1 = [p for p in picks if p["pickNumber"] <= 20]
        g = defaultdict(list)
        for p in r1:
            g[grp.get(p["team"]["id"])].append(p)
        r1rank = {}
        for gg, pp in g.items():
            for i, p in enumerate(sorted(pp, key=lambda x: x["pickNumber"]), 1):
                r1rank[p["team"]["id"]] = i
        for tid, rk in r1rank.items():
            t = st.get(tid)
            if not t or t.get("points") is None:
                continue
            rows.append({
                "split": split, "rank": rk,
                "points": t.get("points", 0),
                "placement": t.get("placementPoints", 0),
                "kills": t.get("kills", 0),
                "position": t.get("position"),
            })

    # 分桶
    def bucket(rk):
        return "1-3" if rk <= 3 else ("4-7" if rk <= 7 else "8-10")
    buckets = {}
    for split in ["split-1", "split-2"]:
        sub = [r for r in rows if r["split"] == split]
        for b in ["1-3", "4-7", "8-10"]:
            bb = [r for r in sub if bucket(r["rank"]) == b]
            buckets[(split, b)] = {
                "points": np.mean([r["points"] for r in bb]) if bb else 0,
                "placement": np.mean([r["placement"] for r in bb]) if bb else 0,
                "kills": np.mean([r["kills"] for r in bb]) if bb else 0,
                "n": len(bb),
            }

    # 每档 rank 1-10 平均分（split-1）
    per_rank = {}
    for split in ["split-1", "split-2"]:
        sub = [r for r in rows if r["split"] == split]
        pr = {}
        for rk in range(1, 11):
            bb = [r for r in sub if r["rank"] == rk]
            pr[rk] = np.mean([r["points"] for r in bb]) if bb else 0
        per_rank[split] = pr

    # 随机性验证
    sub = [r for r in rows if r["position"] is not None]
    rand_rho, _ = spearman([r["rank"] for r in sub], [r["position"] for r in sub])

    # split 相关
    corr = {}
    for split in ["split-1", "split-2"]:
        sub = [r for r in rows if r["split"] == split]
        corr[split], _ = spearman([r["rank"] for r in sub], [r["points"] for r in sub])

    # 统计检验
    def welch(a, b):
        import math
        ea, eb = np.array(a), np.array(b)
        diff = ea.mean() - eb.mean()
        se = math.sqrt(ea.var() / len(ea) + eb.var() / len(eb))
        t = diff / se
        df = (ea.var() / len(ea) + eb.var() / len(eb)) ** 2 / (
            (ea.var() / len(ea)) ** 2 / (len(ea) - 1) + (eb.var() / len(eb)) ** 2 / (len(eb) - 1))
        from scipy import stats as sp
        p = sp.t.sf(abs(t), df) * 2
        pooled = math.sqrt((ea.var() * (len(ea) - 1) + eb.var() * (len(eb) - 1)) / (len(ea) + len(eb) - 2))
        return diff, t, p, pooled and diff / pooled
    tests = {}
    for split in ["split-1", "split-2"]:
        sub = [r for r in rows if r["split"] == split]
        e = [r["points"] for r in sub if bucket(r["rank"]) == "1-3"]
        l = [r["points"] for r in sub if bucket(r["rank"]) == "8-10"]
        tests[split] = welch(e, l)

    # 分地图相关（每地图全局 rank vs points）
    map_rows = []
    for f in glob.glob(str(RAW / "*.json")):
        rec = json.load(open(f))
        s = rec.get("series") or {}
        if s.get("status") != "completed" or s.get("name") == "Regional Finals":
            continue
        stats = rec.get("stats") or {}
        picks = (rec.get("picks") or {}).get("picks", [])
        pts = {t["id"]: t.get("points") for t in stats.get("teams", [])}
        by_map = defaultdict(list)
        for p in picks:
            by_map[p["map"]["name"]].append(p)
        for m, pl in by_map.items():
            pl = sorted(pl, key=lambda x: x["pickNumber"])
            for rank, p in enumerate(pl, 1):
                tid = p["team"]["id"]
                if pts.get(tid) is not None:
                    map_rows.append((rec["slug"], m, rank, pts[tid]))
    map_corr = {}
    for m in ["Storm Point", "E-District", "Olympus"]:
        sub = [(r, p) for slug, mm, r, p in map_rows if mm == m and slug.startswith("split-1")]
        if sub:
            map_corr[m], _ = spearman([x[0] for x in sub], [x[1] for x in sub])
        else:
            map_corr[m] = 0.0

    return {
        "buckets": buckets, "per_rank": per_rank, "rand_rho": rand_rho,
        "corr": corr, "tests": tests, "map_corr": map_corr,
    }


def pos_range():
    """每个名次(1-20)的总分 最低/平均/最高（Split 1 常规赛）。"""
    agg = defaultdict(list)
    for f in glob.glob(str(RAW / "*.json")):
        rec = json.load(open(f))
        s = rec.get("series") or {}
        if s.get("status") != "completed" or s.get("name") == "Regional Finals":
            continue
        if not rec["slug"].startswith("split-1"):
            continue
        for t in (rec.get("stats") or {}).get("teams", []):
            p = t.get("position")
            if p is None:
                continue
            agg[p].append(t.get("points") or 0)
    return {p: (min(v), float(np.median(v)), float(np.mean(v)), max(v)) for p, v in sorted(agg.items())}


# ============ SVG 图表 ============

def bar_chart_grouped(D):
    """分组柱状图：3 桶 × (Split1, Split2) 总分。"""
    W, H = 660, 360
    left, right, top, bottom = 60, 20, 30, 50
    plot_w = W - left - right
    plot_h = H - top - bottom
    ymax = 50
    labels = ["先选 1-3", "中段 4-7", "后选 8-10"]
    n = len(labels)
    group_w = plot_w / n
    bar_w = group_w * 0.32

    def y(v):
        return top + plot_h - (v / ymax) * plot_h

    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="选点顺序与平均总分">']
    # grid + y ticks
    for gv in [0, 10, 20, 30, 40, 50]:
        svg.append(f'<line x1="{left}" y1="{y(gv)}" x2="{W-right}" y2="{y(gv)}" class="grid"/>')
        svg.append(f'<text x="{left-8}" y="{y(gv)+4}" class="tick" text-anchor="end">{gv}</text>')
    for i, lab in enumerate(labels):
        cx = left + group_w * i + group_w / 2
        s1 = D["buckets"][("split-1", ["1-3", "4-7", "8-10"][i])]["points"]
        s2 = D["buckets"][("split-2", ["1-3", "4-7", "8-10"][i])]["points"]
        # split1
        x1 = cx - bar_w - 2
        svg.append(f'<rect x="{x1:.1f}" y="{y(s1)}" width="{bar_w}" height="{top+plot_h-y(s1)}" class="bar" data-v="{s1:.1f}"><title>Split 1：{s1:.1f} 分</title></rect>')
        svg.append(f'<text x="{x1+bar_w/2:.1f}" y="{y(s1)-6}" class="vallabel" text-anchor="middle">{s1:.1f}</text>')
        # split2
        x2 = cx + 2
        svg.append(f'<rect x="{x2:.1f}" y="{y(s2)}" width="{bar_w}" height="{top+plot_h-y(s2)}" class="bar bar2" data-v="{s2:.1f}"><title>Split 2：{s2:.1f} 分</title></rect>')
        svg.append(f'<text x="{x2+bar_w/2:.1f}" y="{y(s2)-6}" class="vallabel" text-anchor="middle">{s2:.1f}</text>')
        svg.append(f'<text x="{cx}" y="{H-bottom+20}" class="xlabel" text-anchor="middle">{lab}</text>')
    svg.append(f'<text x="{left}" y="{H-14}" class="axis-title">选点顺序（Round 1 组内）</text>')
    svg.append("</svg>")
    return "".join(svg)


def bar_chart_stacked(D):
    """堆叠柱状图：Split1 排名分 + 击杀。"""
    W, H = 660, 360
    left, right, top, bottom = 60, 20, 30, 50
    plot_w = W - left - right
    plot_h = H - top - bottom
    ymax = 45
    labels = ["先选 1-3", "中段 4-7", "后选 8-10"]
    n = len(labels)
    group_w = plot_w / n
    bar_w = group_w * 0.5

    def y(v):
        return top + plot_h - (v / ymax) * plot_h

    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Split 1 得分构成">']
    for gv in [0, 10, 20, 30, 40]:
        svg.append(f'<line x1="{left}" y1="{y(gv)}" x2="{W-right}" y2="{y(gv)}" class="grid"/>')
        svg.append(f'<text x="{left-8}" y="{y(gv)+4}" class="tick" text-anchor="end">{gv}</text>')
    for i, lab in enumerate(labels):
        b = D["buckets"][("split-1", ["1-3", "4-7", "8-10"][i])]
        pl, kl = b["placement"], b["kills"]
        cx = left + group_w * i + group_w / 2
        x1 = cx - bar_w / 2
        svg.append(f'<rect x="{x1}" y="{y(pl+kl)}" width="{bar_w}" height="{plot_h - (plot_h*(pl+kl)/ymax)}" class="bar-stack-a"><title>排名分 {pl:.1f} + 击杀 {kl:.1f} = {pl+kl:.1f}</title></rect>')
        svg.append(f'<rect x="{x1}" y="{y(pl)}" width="{bar_w}" height="{plot_h*kl/ymax}" class="bar-stack-b"><title>击杀 {kl:.1f}</title></rect>')
        svg.append(f'<text x="{cx}" y="{y(pl+kl)-6}" class="vallabel" text-anchor="middle">{pl+kl:.1f}</text>')
        svg.append(f'<text x="{cx}" y="{H-bottom+20}" class="xlabel" text-anchor="middle">{lab}</text>')
    svg.append(f'<text x="{left}" y="{H-14}" class="axis-title">Split 1 平均得分构成</text>')
    svg.append("</svg>")
    return "".join(svg)


def line_chart(D):
    """折线图：rank 1-10 平均分趋势（Split1 + Split2）。"""
    W, H = 660, 360
    left, right, top, bottom = 50, 20, 30, 50
    plot_w = W - left - right
    plot_h = H - top - bottom
    ymax = 50

    def x(rk):
        return left + (rk - 1) / 9 * plot_w
    def y(v):
        return top + plot_h - (v / ymax) * plot_h

    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="选点顺序趋势">']
    for gv in [0, 10, 20, 30, 40, 50]:
        svg.append(f'<line x1="{left}" y1="{y(gv)}" x2="{W-right}" y2="{y(gv)}" class="grid"/>')
        svg.append(f'<text x="{left-8}" y="{y(gv)+4}" class="tick" text-anchor="end">{gv}</text>')
    for rk in range(1, 11):
        svg.append(f'<text x="{x(rk)}" y="{H-bottom+18}" class="xlabel" text-anchor="middle">{rk}</text>')
    for split, cls, label in [("split-1", "line-a", "Split 1（40场）"), ("split-2", "line-b", "Split 2（4场）")]:
        pts = [(x(rk), y(D["per_rank"][split][rk])) for rk in range(1, 11)]
        d = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        svg.append(f'<polyline points="{d}" fill="none" class="{cls}"/>')
        for px, py in pts:
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" class="{cls}"/>')
    svg.append(f'<text x="{left}" y="{H-14}" class="axis-title">Round 1 组内选点顺序（1 = 最先选）</text>')
    svg.append("</svg>")
    return "".join(svg)


def heatmap(D):
    """分地图 × 指标 的相关性（简化为横向条形）。"""
    mc = D["map_corr"]
    W, H = 660, 240
    left, right, top, bottom = 140, 30, 20, 30
    plot_w = W - left - right
    # 中心 0 在中间
    def x(v):
        return left + plot_w / 2 + (v / 0.25) * (plot_w / 2)
    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="分地图相关性">']
    svg.append(f'<line x1="{x(0)}" y1="{top}" x2="{x(0)}" y2="{H-bottom}" class="baseline"/>')
    svg.append(f'<text x="{x(-0.2)}" y="{H-bottom+14}" class="tick" text-anchor="middle">← 先选分高</text>')
    svg.append(f'<text x="{x(0.2)}" y="{H-bottom+14}" class="tick" text-anchor="middle">后选分高 →</text>')
    row_h = (H - top - bottom) / len(mc)
    for i, (m, v) in enumerate(mc.items()):
        cy = top + row_h * i + row_h / 2
        svg.append(f'<text x="{left-10}" y="{cy+4}" class="xlabel" text-anchor="end">{m}</text>')
        x0 = x(0)
        xv = x(v)
        svg.append(f'<rect x="{min(x0,xv)}" y="{cy-row_h*0.3}" width="{abs(xv-x0)}" height="{row_h*0.6}" class="{ "bar" if v<0 else "bar2"}" rx="3"><title>{m}: ρ={v:+.3f}</title></rect>')
        svg.append(f'<text x="{xv + (8 if v>=0 else -8)}" y="{cy+4}" class="vallabel" text-anchor="{ "start" if v>=0 else "end" }">{v:+.2f}</text>')
    svg.append("</svg>")
    return "".join(svg)


def range_chart(P):
    """范围图：每个名次的总分区间(min-max) + 均值点。"""
    W, H = 680, 400
    left, right, top, bottom = 56, 20, 24, 56
    plot_w = W - left - right
    plot_h = H - top - bottom
    ymax = 110
    n = 20

    def x(p):
        return left + (p - 1) / (n - 1) * plot_w

    def y(v):
        return top + plot_h - (v / ymax) * plot_h

    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="各名次总分区间">']
    for gv in range(0, 111, 10):
        svg.append(f'<line x1="{left}" y1="{y(gv)}" x2="{W-right}" y2="{y(gv)}" class="grid"/>')
        svg.append(f'<text x="{left-8}" y="{y(gv)+4}" class="tick" text-anchor="end">{gv}</text>')
    for p in range(1, 21):
        mn, med, avg, mx = P[p]
        px = x(p)
        svg.append(f'<line x1="{px:.1f}" y1="{y(mx)}" x2="{px:.1f}" y2="{y(mn)}" class="range-line"/>')
        svg.append(f'<line x1="{px-3:.1f}" y1="{y(mx)}" x2="{px+3:.1f}" y2="{y(mx)}" class="range-line"/>')
        svg.append(f'<line x1="{px-3:.1f}" y1="{y(mn)}" x2="{px+3:.1f}" y2="{y(mn)}" class="range-line"/>')
        svg.append(f'<line x1="{px-5:.1f}" y1="{y(med)}" x2="{px+5:.1f}" y2="{y(med)}" class="median-tick"><title>第{p}名中位数 {med:.1f}</title></line>')
        svg.append(f'<circle cx="{px:.1f}" cy="{y(avg)}" r="3.5" class="mean-dot"><title>第{p}名：最低 {mn} · 中位数 {med:.1f} · 平均 {avg:.1f} · 最高 {mx}</title></circle>')
        svg.append(f'<text x="{px:.1f}" y="{H-bottom+18}" class="xlabel" text-anchor="middle">{p}</text>')
    svg.append(f'<text x="{left}" y="{H-12}" class="axis-title">比赛日最终名次（1 = 冠军）</text>')
    svg.append("</svg>")
    return "".join(svg)


# ============ HTML 组装 ============

def stat_tile(value, label, note, tone=""):
    return f'''
    <div class="tile {tone}">
      <div class="tile-value">{value}</div>
      <div class="tile-label">{label}</div>
      <div class="tile-note">{note}</div>
    </div>'''


def main():
    D = compute()
    t1 = D["tests"]["split-1"]
    t2 = D["tests"]["split-2"]
    P = pos_range()
    pos_rows = "".join(
        f'<tr><td class="num">{p}</td><td class="num">{P[p][0]}</td><td class="num">{P[p][1]:.1f}</td><td class="num">{P[p][2]:.1f}</td><td class="num">{P[p][3]}</td></tr>'
        for p in range(1, 21)
    )

    tiles = "".join([
        stat_tile(f"+{t1[0]:.1f}<span class='unit'>分</span>", "Split 1 先选优势", f"先选1-3 vs 后选8-10 · p={t1[2]:.3f} · d={t1[3]:.2f}"),
        stat_tile(f"+{t2[0]:.1f}<span class='unit'>分</span>", "Split 2 先选优势", "仅 4 场 · 样本不足 · 待更新", "warn"),
        stat_tile(f"{D['rand_rho']:+.2f}", "选点顺序随机性", "选点顺序 vs 队伍排名（≈0 为随机）"),
        stat_tile(f"{D['corr']['split-1']:+.3f}", "Split 1 秩相关 ρ", "选点顺序 vs 总分（弱负相关）"),
    ])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ALGS 选点顺序 × 得分 分析</title>
<style>
:root {{
  --surface: {C['surface']}; --ink: {C['ink']}; --sec: {C['sec']}; --muted: {C['muted']};
  --grid: {C['grid']}; --baseline: {C['baseline']};
  --blue: {C['blue']}; --blue-dark: {C['blue_dark']}; --blue-light: {C['blue_light']};
  --orange: {C['orange']}; --aqua: {C['aqua']};
  --page: #f9f9f7;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --surface: #1a1a19; --ink: #ffffff; --sec: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --baseline: #383835; --page: #0d0d0d;
    --blue: #3987e5; --blue-dark: #5598e7; --blue-light: #2a78d6; --orange: #d95926; }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--page); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", "PingFang SC", "Noto Sans SC", sans-serif;
  line-height: 1.6; -webkit-font-smoothing: antialiased;
}}
.wrap {{ max-width: 960px; margin: 0 auto; padding: 48px 24px 80px; }}
h1 {{ font-size: 30px; margin: 0 0 8px; letter-spacing: -0.5px; }}
.sub {{ color: var(--sec); font-size: 15px; margin: 0 0 40px; }}
h2 {{ font-size: 20px; margin: 48px 0 16px; letter-spacing: -0.3px; }}
h3 {{ font-size: 16px; margin: 24px 0 8px; }}
p, li {{ color: var(--sec); font-size: 15px; }}
strong {{ color: var(--ink); }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin: 28px 0; }}
.tile {{ background: var(--surface); border: 1px solid var(--grid); border-radius: 12px; padding: 20px 18px; }}
.tile.warn {{ border-left: 3px solid var(--orange); }}
.tile-value {{ font-size: 34px; font-weight: 700; letter-spacing: -1px; }}
.tile-value .unit {{ font-size: 15px; font-weight: 500; color: var(--sec); margin-left: 2px; }}
.tile-label {{ font-size: 14px; color: var(--ink); margin-top: 4px; }}
.tile-note {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
.chart {{ background: var(--surface); border: 1px solid var(--grid); border-radius: 12px; padding: 20px; margin: 20px 0; }}
.chart h3 {{ margin: 0 0 4px; }}
.chart .cap {{ color: var(--muted); font-size: 13px; margin: 0 0 12px; }}
svg {{ display: block; width: 100%; height: auto; }}
.grid {{ stroke: var(--grid); stroke-width: 1; }}
.baseline {{ stroke: var(--baseline); stroke-width: 1.2; }}
.tick {{ fill: var(--muted); font-size: 12px; }}
.xlabel {{ fill: var(--sec); font-size: 13px; }}
.axis-title {{ fill: var(--muted); font-size: 12px; }}
.vallabel {{ fill: var(--ink); font-size: 12px; font-weight: 600; }}
.bar {{ fill: var(--blue); }}
.bar2 {{ fill: var(--orange); }}
.bar-stack-a {{ fill: var(--blue-dark); }}
.bar-stack-b {{ fill: var(--blue-light); }}
.line-a {{ stroke: var(--blue); stroke-width: 2.5; fill: var(--blue); }}
.line-b {{ stroke: var(--orange); stroke-width: 2.5; fill: var(--orange); }}
.range-line {{ stroke: var(--blue-light); stroke-width: 1.5; }}
.mean-dot {{ fill: var(--blue); stroke: var(--surface); stroke-width: 2; }}
.median-tick {{ stroke: var(--blue-dark); stroke-width: 2.5; }}
.bar, .bar2, .bar-stack-a, .bar-stack-b {{ transition: opacity .15s; }}
.chart:hover .bar:hover {{ opacity: .82; }}
.legend {{ display: flex; gap: 18px; flex-wrap: wrap; font-size: 13px; color: var(--sec); margin: 8px 0 0; }}
.legend .sw {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }}
.sw.blue {{ background: var(--blue); }} .sw.blue-dark {{ background: var(--blue-dark); }}
.sw.blue-light {{ background: var(--blue-light); }} .sw.orange {{ background: var(--orange); }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin: 16px 0; }}
th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--grid); }}
th {{ color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .4px; }}
td.num {{ font-variant-numeric: tabular-nums; }}
.callout {{ background: var(--surface); border: 1px solid var(--grid); border-left: 3px solid var(--blue); border-radius: 8px; padding: 16px 18px; margin: 20px 0; }}
.callout.warn {{ border-left-color: var(--orange); }}
.callout p {{ margin: 0; }}
code {{ background: var(--grid); padding: 2px 6px; border-radius: 4px; font-size: 13px; }}
.small {{ font-size: 13px; color: var(--muted); }}
</style>
</head>
<body>
<div class="wrap">
  <h1>ALGS 选点顺序 × 比赛得分</h1>
  <p class="sub">Apex Legends Global Series · Year 6 Pro League · POI Draft 顺序与总分的关联分析</p>

  <div class="tiles">{tiles}</div>

  <div class="callout">
    <p><strong>一句话结论：</strong>常规赛采用「准随机」的蛇形选秀，先选确实带来约 <strong>+3.5 分</strong>的轻微优势（主要来自排名分），但效应量小（Cohen's d = 0.22）——说明平衡轮换的设计把「选点运气」压到了很低的水平。</p>
  </div>

  <h2>1 · 选点顺序与总分</h2>
  <div class="chart">
    <h3>平均总分：先选 vs 中段 vs 后选</h3>
    <p class="cap">Round 1 组内选点顺序分桶（1-3 先选 / 4-7 中段 / 8-10 后选）</p>
    {bar_chart_grouped(D)}
    <div class="legend"><span><span class="sw blue"></span>Split 1（36 日）</span><span><span class="sw orange"></span>Split 2（4 日，样本小）</span></div>
  </div>

  <h2>2 · 先选优势来自哪里</h2>
  <div class="chart">
    <h3>Split 1 得分构成：排名分 + 击杀</h3>
    <p class="cap">先选到好点位后，更容易存活到后期（排名分），击杀也略高</p>
    {bar_chart_stacked(D)}
    <div class="legend"><span><span class="sw blue-dark"></span>排名分</span><span><span class="sw blue-light"></span>击杀</span></div>
  </div>

  <h2>3 · 逐档趋势</h2>
  <div class="chart">
    <h3>选点顺序（1-10 档）→ 平均总分</h3>
    <p class="cap">Split 1 呈平缓下行；Split 2 波动大但样本仅 4 场</p>
    {line_chart(D)}
    <div class="legend"><span><span class="sw blue"></span>Split 1</span><span><span class="sw orange"></span>Split 2</span></div>
  </div>

  <h2>4 · 分地图相关性</h2>
  <div class="chart">
    <h3>各地图选点顺序 → 总分的 Spearman ρ（Split 1）</h3>
    <p class="cap">负值 = 先选分高；Storm Point 的先选优势最明显</p>
    {heatmap(D)}
  </div>

  <h2>5 · 数据明细</h2>
  <table>
    <thead><tr><th>分组</th><th>选点顺序</th><th class="num">平均总分</th><th class="num">排名分</th><th class="num">击杀</th><th class="num">样本(队次)</th></tr></thead>
    <tbody>
      <tr><td rowspan="3">Split 1</td><td>先选 1-3</td><td class="num">{D['buckets'][('split-1','1-3')]['points']:.1f}</td><td class="num">{D['buckets'][('split-1','1-3')]['placement']:.1f}</td><td class="num">{D['buckets'][('split-1','1-3')]['kills']:.1f}</td><td class="num">{D['buckets'][('split-1','1-3')]['n']}</td></tr>
      <tr><td>中段 4-7</td><td class="num">{D['buckets'][('split-1','4-7')]['points']:.1f}</td><td class="num">{D['buckets'][('split-1','4-7')]['placement']:.1f}</td><td class="num">{D['buckets'][('split-1','4-7')]['kills']:.1f}</td><td class="num">{D['buckets'][('split-1','4-7')]['n']}</td></tr>
      <tr><td>后选 8-10</td><td class="num">{D['buckets'][('split-1','8-10')]['points']:.1f}</td><td class="num">{D['buckets'][('split-1','8-10')]['placement']:.1f}</td><td class="num">{D['buckets'][('split-1','8-10')]['kills']:.1f}</td><td class="num">{D['buckets'][('split-1','8-10')]['n']}</td></tr>
      <tr><td rowspan="3">Split 2</td><td>先选 1-3</td><td class="num">{D['buckets'][('split-2','1-3')]['points']:.1f}</td><td class="num">{D['buckets'][('split-2','1-3')]['placement']:.1f}</td><td class="num">{D['buckets'][('split-2','1-3')]['kills']:.1f}</td><td class="num">{D['buckets'][('split-2','1-3')]['n']}</td></tr>
      <tr><td>中段 4-7</td><td class="num">{D['buckets'][('split-2','4-7')]['points']:.1f}</td><td class="num">{D['buckets'][('split-2','4-7')]['placement']:.1f}</td><td class="num">{D['buckets'][('split-2','4-7')]['kills']:.1f}</td><td class="num">{D['buckets'][('split-2','4-7')]['n']}</td></tr>
      <tr><td>后选 8-10</td><td class="num">{D['buckets'][('split-2','8-10')]['points']:.1f}</td><td class="num">{D['buckets'][('split-2','8-10')]['placement']:.1f}</td><td class="num">{D['buckets'][('split-2','8-10')]['kills']:.1f}</td><td class="num">{D['buckets'][('split-2','8-10')]['n']}</td></tr>
    </tbody>
  </table>

  <h2>6 · 各名次要多少分</h2>
  <div class="chart">
    <h3>比赛日名次 → 总分区间（最低 / 平均 / 最高）</h3>
    <p class="cap">Split 1 常规赛 36 个比赛日。竖线 = 分数波动范围，圆点 = 平均，横线 = 中位数</p>
    {range_chart(P)}
    <div class="legend"><span><span class="sw blue"></span>平均（圆点）</span><span><span class="sw blue-dark"></span>中位数（横线）</span><span><span class="sw blue-light"></span>最低 ~ 最高区间</span></div>
  </div>
  <table>
    <thead><tr><th>名次</th><th class="num">最低分</th><th class="num">中位数</th><th class="num">平均分</th><th class="num">最高分</th></tr></thead>
    <tbody>{pos_rows}</tbody>
  </table>

  <h2>7 · 方法与说明</h2>
  <ul>
    <li><strong>选点顺序</strong> = 每队 Round 1 组内排名（1 = 最先选）。常规赛用蛇形选秀，Round 2 是 Round 1 逆序，故取 Round 1 作为「选点优先度」。</li>
    <li><strong>得分</strong> = series stats 中的整场总分 <code>points</code>（= 排名分 + 击杀）。</li>
    <li>Regional Finals（Series 10）因机制不同（排名高者先选）已从主分析中排除。</li>
    <li>统计方法：Spearman 秩相关、Welch t 检验、Cohen's d。</li>
  </ul>

  <div class="callout warn">
    <p><strong>⚠️ Split 2 局限：</strong>今天（2026-08-22）才开赛，每赛区仅 Series 1 打完（共 4 个比赛日）。图上 Split 2 的 +15 分大效应很可能被小样本放大，<strong>需等 9/13 全部 36 个比赛日结束后再更新</strong>。</p>
  </div>

  <p class="small">数据来源：algs.ea.com · prod-api.algstools.com/v1 · 官方 Year 6 规则 Appendix F4</p>
</div>
</body>
</html>"""
    out = OUT / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"已生成: {out}")


if __name__ == "__main__":
    main()
