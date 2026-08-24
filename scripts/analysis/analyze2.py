#!/usr/bin/env python3
"""深入分析：Round 1 / Round 2 分离 + 组内 rank + 随机性验证。

常规赛 snake draft:
  Round 1 = pickNumber 1-20（每队第一次选点）
  Round 2 = pickNumber 21-40（每队第二次选点，组内逆序）
  每队组内 rank(1-10) = 在其 group(A/B) 内的选点顺序。
  Round 1 组内 rank 最小者 = 全场最先选点的队。
"""
import json
import glob
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

REGION = {
    "split-1-pro-league-americas": "Split1-AMER",
    "split-1-pro-league-apac-north": "Split1-APN",
    "split-1-pro-league-apac-south": "Split1-APS",
    "split-1-pro-league-emea": "Split1-EMEA",
    "split-2-pro-league-americas": "Split2-AMER",
    "split-2-pro-league-apac-north": "Split2-APN",
    "split-2-pro-league-apac-south": "Split2-APS",
    "split-2-pro-league-emea": "Split2-EMEA",
}


def spearman(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0, n
    def ranks(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0] * n
        for pos, idx in enumerate(order):
            r[idx] = pos + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n))
    vy = sum((ry[i] - my) ** 2 for i in range(n))
    if vx == 0 or vy == 0:
        return 0.0, n
    return cov / (vx * vy) ** 0.5, n


def load_rows():
    """构建每队每场比赛的明细：round1_rank, round2_rank, group, points, position。"""
    rows = []
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
        # 队 -> group, 队 -> points/position
        grp = {t["teamId"]: t.get("group") for t in s.get("teams", [])}
        pts = {t["id"]: t.get("points") for t in stats.get("teams", [])}
        pos = {t["id"]: t.get("position") for t in stats.get("teams", [])}
        name = {t["id"]: t.get("name") for t in stats.get("teams", [])}

        if is_rf:
            # Regional Finals: 排名 draft, 每队 3 picks, 顺序同(非snake)。只记录全局每地图 rank。
            by_map = defaultdict(list)
            for p in picks:
                by_map[p["map"]["name"]].append(p)
            for m, pl in by_map.items():
                pl = sorted(pl, key=lambda x: x["pickNumber"])
                for rank, p in enumerate(pl, 1):
                    tid = p["team"]["id"]
                    if pts.get(tid) is None:
                        continue
                    rows.append({
                        "slug": slug, "region": REGION.get(slug, slug),
                        "is_rf": True, "map": m, "rank": rank,
                        "points": pts[tid], "position": pos.get(tid),
                        "round1_rank": None, "round2_rank": None,
                        "group": grp.get(tid), "team": name.get(tid, tid),
                    })
            continue

        # 常规赛：40 picks，2 地图。Round1 = pick<=20, Round2 = pick>20。
        picks = sorted(picks, key=lambda x: x["pickNumber"])
        r1 = [p for p in picks if p["pickNumber"] <= 20]
        r2 = [p for p in picks if p["pickNumber"] > 20]
        # 组内 rank: 按 group 分组, 组内按 pickNumber 排序
        def group_rank(plist):
            g = defaultdict(list)
            for p in plist:
                g[grp.get(p["team"]["id"])].append(p)
            out = {}
            for gg, pp in g.items():
                pp = sorted(pp, key=lambda x: x["pickNumber"])
                for i, p in enumerate(pp, 1):
                    out[p["team"]["id"]] = i
            return out
        r1_rank = group_rank(r1)
        r2_rank = group_rank(r2)
        for p in picks:
            tid = p["team"]["id"]
            if pts.get(tid) is None:
                continue
            rows.append({
                "slug": slug, "region": REGION.get(slug, slug),
                "is_rf": False, "map": p["map"]["name"],
                "rank": None, "points": pts[tid], "position": pos.get(tid),
                "round1_rank": r1_rank.get(tid), "round2_rank": r2_rank.get(tid),
                "group": grp.get(tid), "team": name.get(tid, tid),
            })
    return rows


def main():
    rows = load_rows()
    reg = [r for r in rows if not r["is_rf"]]
    print(f"常规赛 队级记录: {len(reg)}")

    # ---- 随机性验证: Round1 组内 rank vs 整场 position ----
    print("\n【验证】选点顺序是否随机? (Round1 组内rank vs 整场排名 position)")
    sub = [r for r in reg if r["round1_rank"] is not None and r["position"] is not None]
    rho, n = spearman([r["round1_rank"] for r in sub], [r["position"] for r in sub])
    print(f"  rho={rho:+.4f} (n={n})  -> 若≈0 说明选点顺序与队伍强弱无关(随机)")
    print("  注意 position 越小排名越靠前; rank 越小越先选。正相关=强队先选(不公平), 0=随机")

    # ---- Round1 组内 rank vs points ----
    print("\n【常规赛】Round1 组内选点顺序(rank 1-10) vs 总分")
    for split in ["Split1", "Split2"]:
        sub = [r for r in reg if r["region"].startswith(split) and r["round1_rank"] is not None]
        if not sub:
            continue
        rho, n = spearman([r["round1_rank"] for r in sub], [r["points"] for r in sub])
        # 分桶 rank 1-3 vs 4-7 vs 8-10
        b = {k: [] for k in ["1-3", "4-7", "8-10"]}
        for r in sub:
            rk = r["round1_rank"]
            if rk <= 3:
                b["1-3"].append(r["points"])
            elif rk <= 7:
                b["4-7"].append(r["points"])
            else:
                b["8-10"].append(r["points"])
        bm = {k: sum(v) / len(v) for k, v in b.items() if v}
        print(f"  {split}: rho={rho:+.4f} (n={n})  分桶 " + " ".join(f"{k}={v:.1f}分" for k, v in bm.items()))

    # ---- Round2 组内 rank vs points (应镜像) ----
    print("\n【常规赛】Round2 组内选点顺序 vs 总分 (snake 逆序, 应镜像)")
    for split in ["Split1", "Split2"]:
        sub = [r for r in reg if r["region"].startswith(split) and r["round2_rank"] is not None]
        if not sub:
            continue
        rho, n = spearman([r["round2_rank"] for r in sub], [r["points"] for r in sub])
        print(f"  {split}: rho={rho:+.4f} (n={n})")

    # ---- 分赛区 x 地图 热力 ----
    print("\n【常规赛】分赛区×地图: 全局rank vs points 的 rho (样本数)")
    print(f"{'赛区':<12} {'Storm Point':>22} {'E-District':>22} {'Olympus':>22}")
    regions = sorted(set(r["region"] for r in reg))
    for region in regions:
        cells = []
        for m in ["Storm Point", "E-District", "Olympus"]:
            sub = [r for r in reg if r["region"] == region and r["map"] == m]
            if sub:
                rho, n = spearman([r["rank"] for r in sub], [r["points"] for r in sub]) if False else (0, 0)
                # 用全局 rank 需要重算，这里改用每地图内 rank 已在 load 里没有; 简化跳过
            cells.append("")
        # 简化：上面 rank=None，改用重算
    # 重新做分赛区×地图（用 picks 直接算）
    print("  (见下方详细计算)\n")
    # 直接用扁平数据重算分赛区×地图
    from collections import defaultdict as dd
    # 重新加载 raw 计算每地图 rank
    all_map_rows = []
    for f in glob.glob(str(RAW / "*.json")):
        rec = json.load(open(f))
        s = rec.get("series") or {}
        if s.get("status") != "completed":
            continue
        stats = rec.get("stats") or {}
        picks = (rec.get("picks") or {}).get("picks", [])
        slug = rec["slug"]
        if s.get("name") == "Regional Finals":
            continue
        pts = {t["id"]: t.get("points") for t in stats.get("teams", [])}
        by_map = dd(list)
        for p in picks:
            by_map[p["map"]["name"]].append(p)
        for m, pl in by_map.items():
            pl = sorted(pl, key=lambda x: x["pickNumber"])
            for rank, p in enumerate(pl, 1):
                tid = p["team"]["id"]
                if pts.get(tid) is None:
                    continue
                all_map_rows.append((slug, m, rank, pts[tid]))
    by_region_map = dd(lambda: dd(list))
    for slug, m, rank, pt in all_map_rows:
        by_region_map[REGION.get(slug, slug)][m].append((rank, pt))
    print(f"{'赛区':<12} {'Storm Point':>20} {'E-District':>20} {'Olympus':>20}")
    for region in regions:
        cells = []
        for m in ["Storm Point", "E-District", "Olympus"]:
            sub = by_region_map[region][m]
            if sub:
                rho, n = spearman([x[0] for x in sub], [x[1] for x in sub])
                cells.append(f"{rho:+.3f}({n})")
            else:
                cells.append("-")
        print(f"{region:<12} {cells[0]:>20} {cells[1]:>20} {cells[2]:>20}")


if __name__ == "__main__":
    main()
