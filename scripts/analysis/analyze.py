#!/usr/bin/env python3
"""分析 ALGS POI Draft 选点顺序与比赛得分的关系。

思路：
- 每场比赛(series)有 20 队，POI Draft 分 2 张地图(常规赛) / 3 张地图(Regional Finals)。
- 每张地图上，20 队各选一个点位，选点顺序 = 该地图内 pickNumber 排序(rank 1~20)。
- series stats 给出每队整场总分 points。
- 分析 rank(选点顺序) 与 points(总分) 的关系。

注意区分：
- 常规赛(Series 1-9)：snake draft，选点顺序准随机 -> 干净的相关分析
- Regional Finals(Series 10)：排名 draft，排名高者先选 -> 反向因果，单独看
"""
import json
import glob
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# 赛事 slug -> 展示名
REGION = {
    "split-1-pro-league-americas": "Split1-Americas",
    "split-1-pro-league-apac-north": "Split1-APAC-N",
    "split-1-pro-league-apac-south": "Split1-APAC-S",
    "split-1-pro-league-emea": "Split1-EMEA",
    "split-2-pro-league-americas": "Split2-Americas",
    "split-2-pro-league-apac-north": "Split2-APAC-N",
    "split-2-pro-league-apac-south": "Split2-APAC-S",
    "split-2-pro-league-emea": "Split2-EMEA",
}


def load_completed():
    """加载所有 completed 比赛，返回记录列表。"""
    recs = []
    for f in glob.glob(str(RAW / "*.json")):
        rec = json.load(open(f))
        s = rec.get("series") or {}
        if s.get("status") != "completed":
            continue
        recs.append(rec)
    return recs


def build_records(rec):
    """从一场比赛记录里构建 (region, seriesNumber, isRegionalFinal, map_name, pick_rank, points, team) 列表。

    每张地图上：按 pickNumber 排序 -> rank 1..20，关联该队总分。
    """
    s = rec["series"]
    slug = rec["slug"]
    stats = rec.get("stats") or {}
    picks = (rec.get("picks") or {}).get("picks", [])
    if not picks:
        return []
    pts_by_team = {t["id"]: t.get("points") for t in stats.get("teams", [])}
    name_by_team = {t["id"]: t.get("name") for t in stats.get("teams", [])}
    is_rf = s.get("name") == "Regional Finals"

    # 按地图分组
    by_map = defaultdict(list)
    for p in picks:
        by_map[p["map"]["name"]].append(p)

    rows = []
    for map_name, plist in by_map.items():
        plist = sorted(plist, key=lambda x: x["pickNumber"])
        for rank, p in enumerate(plist, 1):
            tid = p["team"]["id"]
            pts = pts_by_team.get(tid)
            if pts is None:
                continue
            rows.append({
                "slug": slug,
                "region": REGION.get(slug, slug),
                "seriesNumber": s.get("seriesNumber"),
                "is_rf": is_rf,
                "map": map_name,
                "rank": rank,           # 1 = 该地图最先选
                "points": pts,
                "team": name_by_team.get(tid, tid),
            })
    return rows


def spearman(xs, ys):
    """Spearman 秩相关系数。"""
    n = len(xs)
    if n < 2:
        return 0.0, 0
    def ranks(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0] * n
        for pos, idx in enumerate(order):
            r[idx] = pos + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n))
    vy = sum((ry[i] - my) ** 2 for i in range(n))
    if vx == 0 or vy == 0:
        return 0.0, n
    return cov / (vx * vy) ** 0.5, n


def bucket_mean(rows):
    """按选点 rank 分桶统计平均分。"""
    buckets = {"1-5": [], "6-10": [], "11-15": [], "16-20": []}
    for r in rows:
        rk = r["rank"]
        if rk <= 5:
            buckets["1-5"].append(r["points"])
        elif rk <= 10:
            buckets["6-10"].append(r["points"])
        elif rk <= 15:
            buckets["11-15"].append(r["points"])
        else:
            buckets["16-20"].append(r["points"])
    return {k: (sum(v) / len(v), len(v)) for k, v in buckets.items() if v}


def main():
    recs = load_completed()
    all_rows = []
    for rec in recs:
        all_rows.extend(build_records(rec))
    print(f"总记录数(队×地图): {len(all_rows)}")

    regular = [r for r in all_rows if not r["is_rf"]]
    rf = [r for r in all_rows if r["is_rf"]]
    print(f"  常规赛记录: {len(regular)} | Regional Finals 记录: {len(rf)}")

    # ---- 1) 整体相关（常规赛） ----
    print("\n" + "=" * 70)
    print("【常规赛】选点顺序(rank) vs 总分(points) — Spearman 相关")
    print("=" * 70)
    xs = [r["rank"] for r in regular]
    ys = [r["points"] for r in regular]
    rho, n = spearman(xs, ys)
    print(f"  全体: rho={rho:+.4f} (n={n})")
    print("  说明: rho<0 => 先选(rank小)者分高; rho>0 => 后选者分高\n")

    # 分 Split
    for split in ["Split1", "Split2"]:
        sub = [r for r in regular if r["region"].startswith(split)]
        if not sub:
            continue
        rho, n = spearman([r["rank"] for r in sub], [r["points"] for r in sub])
        print(f"  {split}: rho={rho:+.4f} (n={n})")

    # 分地图
    print("\n  分地图:")
    maps = defaultdict(list)
    for r in regular:
        maps[r["map"]].append(r)
    for m, sub in sorted(maps.items(), key=lambda x: -len(x[1])):
        rho, n = spearman([r["rank"] for r in sub], [r["points"] for r in sub])
        print(f"    {m:<14} rho={rho:+.4f} (n={n})")

    # ---- 2) 分桶对比 ----
    print("\n" + "=" * 70)
    print("【常规赛】选点 rank 分桶平均分")
    print("=" * 70)
    for label, sub in [("全体", regular)]:
        b = bucket_mean(sub)
        print(f"  {label}: " + "  ".join(f"{k}[{v[1]}队]={v[0]:.1f}分" for k, v in b.items()))

    print("\n  Split 1 vs Split 2:")
    for split in ["Split1", "Split2"]:
        sub = [r for r in regular if r["region"].startswith(split)]
        if not sub:
            continue
        b = bucket_mean(sub)
        print(f"    {split}: " + "  ".join(f"{k}={v[0]:.1f}分" for k, v in b.items()))

    # ---- 3) Regional Finals 单独看 ----
    if rf:
        print("\n" + "=" * 70)
        print("【Regional Finals】排名 draft(排名高者先选) — 仅作对照")
        print("=" * 70)
        rho, n = spearman([r["rank"] for r in rf], [r["points"] for r in rf])
        print(f"  rank vs points: rho={rho:+.4f} (n={n})")
        print("  (注意: 此处 rank 由排名决定, 是反向因果, 不代表选点优势)")
        b = bucket_mean(rf)
        print("  分桶: " + "  ".join(f"{k}={v[0]:.1f}分" for k, v in b.items()))

    # 保存分析用的扁平数据
    out = ROOT / "data" / "analysis_rows.json"
    out.write_text(json.dumps(all_rows, ensure_ascii=False), encoding="utf-8")
    print(f"\n扁平数据已保存: {out}")


if __name__ == "__main__":
    main()
