#!/usr/bin/env python3
"""生成分析图表（遵循 dataviz 规范配色）。"""
import json
import glob
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 注册中文字体（优先 Noto Sans SC / 微软雅黑）
for _fp in [
    "/mnt/c/Windows/Fonts/NotoSansSC-VF.ttf",
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]:
    try:
        font_manager.fontManager.addfont(_fp)
        _name = font_manager.FontProperties(fname=_fp).get_name()
        plt.rcParams["font.family"] = _name
        break
    except Exception:
        continue

ROOT = "/mnt/j/rank"
RAW = f"{ROOT}/data/raw"
OUT = f"{ROOT}/output"

# palette
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SEC = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"      # series 1
ORANGE = "#eb6834"    # series 2
BLUE_DARK = "#1c5cab"
BLUE_LIGHT = "#86b6ef"

plt.rcParams.update({
    "text.color": INK,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": SEC,
    "xtick.color": SEC,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_bucket_data():
    """返回 {split: {bucket: {'points':[], 'placement':[], 'kills':[]}}}"""
    data = {"split-1": defaultdict(lambda: defaultdict(list)),
            "split-2": defaultdict(lambda: defaultdict(list))}
    for f in glob.glob(f"{RAW}/*.json"):
        rec = json.load(open(f))
        s = rec.get("series") or {}
        if s.get("status") != "completed" or s.get("name") == "Regional Finals":
            continue
        stats = rec.get("stats") or {}
        picks = (rec.get("picks") or {}).get("picks", [])
        grp = {t["teamId"]: t.get("group") for t in s.get("teams", [])}
        st = {t["id"]: t for t in stats.get("teams", [])}
        picks = sorted(picks, key=lambda x: x["pickNumber"])
        r1 = [p for p in picks if p["pickNumber"] <= 20]
        g = defaultdict(list)
        for p in r1:
            g[grp.get(p["team"]["id"])].append(p)
        r1rank = {}
        for gg, pp in g.items():
            for i, p in enumerate(sorted(pp, key=lambda x: x["pickNumber"]), 1):
                r1rank[p["team"]["id"]] = i
        key = "split-1" if rec["slug"].startswith("split-1") else "split-2"
        for tid, rk in r1rank.items():
            t = st.get(tid)
            if not t:
                continue
            bucket = "先选1-3" if rk <= 3 else ("中4-7" if rk <= 7 else "后选8-10")
            data[key][bucket]["points"].append(t.get("points") or 0)
            data[key][bucket]["placement"].append(t.get("placementPoints") or 0)
            data[key][bucket]["kills"].append(t.get("kills") or 0)
    return data


def mean(v):
    return sum(v) / len(v) if v else 0.0


def main():
    data = load_bucket_data()
    buckets = ["先选1-3", "中4-7", "后选8-10"]

    # ---- 图1: Split 1 堆叠柱状图（排名分 + 击杀） ----
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(buckets))
    s1 = data["split-1"]
    place = [mean(s1[b]["placement"]) for b in buckets]
    kills = [mean(s1[b]["kills"]) for b in buckets]
    ax.bar(x, place, width=0.5, color=BLUE_DARK, label="排名分")
    ax.bar(x, kills, width=0.5, bottom=place, color=BLUE_LIGHT, label="击杀")
    for i, b in enumerate(buckets):
        tot = place[i] + kills[i]
        ax.text(i, tot + 0.7, f"{tot:.1f}", ha="center", va="bottom",
                fontsize=11, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylabel("平均分")
    ax.set_ylim(0, 45)
    ax.set_title("Split 1 选点顺序 → 平均得分（先选 vs 后选）", fontsize=12, color=INK, pad=12)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/chart_split1_buckets.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- 图2: Split 1 vs Split 2 分组柱状图（总分） ----
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(buckets))
    w = 0.35
    tot1 = [mean(data["split-1"][b]["points"]) for b in buckets]
    tot2 = [mean(data["split-2"][b]["points"]) for b in buckets]
    ax.bar(x - w/2, tot1, width=w, color=BLUE, label="Split 1 (40场)")
    ax.bar(x + w/2, tot2, width=w, color=ORANGE, label="Split 2 (4场, 样本小)")
    for i in range(len(buckets)):
        ax.text(x[i] - w/2, tot1[i] + 0.5, f"{tot1[i]:.1f}", ha="center", fontsize=10, color=INK)
        ax.text(x[i] + w/2, tot2[i] + 0.5, f"{tot2[i]:.1f}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylabel("平均总分")
    ax.set_ylim(0, 52)
    ax.set_title("选点顺序 → 平均总分：Split 1 vs Split 2", fontsize=12, color=INK, pad=12)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/chart_split_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- 图3: 散点图（选点 rank vs 总分，Split 1） ----
    ranks, pts = [], []
    for f in glob.glob(f"{RAW}/*.json"):
        rec = json.load(open(f))
        s = rec.get("series") or {}
        if s.get("status") != "completed" or s.get("name") == "Regional Finals":
            continue
        if not rec["slug"].startswith("split-1"):
            continue
        stats = rec.get("stats") or {}
        picks = (rec.get("picks") or {}).get("picks", [])
        grp = {t["teamId"]: t.get("group") for t in s.get("teams", [])}
        st = {t["id"]: t for t in stats.get("teams", [])}
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
            if t and t.get("points") is not None:
                ranks.append(rk)
                pts.append(t["points"])
    ranks = np.array(ranks)
    pts = np.array(pts)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ranks + np.random.uniform(-0.15, 0.15, len(ranks)),
               pts, s=18, alpha=0.35, color=BLUE, edgecolors="none")
    # 每 rank 的均值 + 趋势
    xs, ys = [], []
    for rk in sorted(set(ranks)):
        xs.append(rk)
        ys.append(pts[ranks == rk].mean())
    ax.plot(xs, ys, color=INK, linewidth=2, marker="o", markersize=4,
            label="每档选点顺序的平均分")
    # 拟合线
    z = np.polyfit(ranks, pts, 1)
    ax.plot([1, 10], np.polyval(z, [1, 10]), color=ORANGE, linewidth=2,
            linestyle="--", label=f"线性趋势 (斜率 {z[0]:.2f}/档)")
    ax.set_xlabel("Round 1 组内选点顺序（1=最先选）")
    ax.set_ylabel("总分")
    ax.set_xticks(range(1, 11))
    ax.set_title("Split 1：选点顺序与总分的散点分布", fontsize=12, color=INK, pad=12)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/chart_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("图表已生成:")
    import os
    for fn in sorted(os.listdir(OUT)):
        print(f"  output/{fn}")


if __name__ == "__main__":
    main()
