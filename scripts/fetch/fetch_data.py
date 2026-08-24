#!/usr/bin/env python3
"""抓取 ALGS Year 6 Split 1 / Split 2 所有已结束比赛的 series / stats / poi-draft 数据。

数据源：
- sitemap.xml 里枚举所有 match URL（含 seriesId）
- API base: https://prod-api.algstools.com/v1
  - GET /series/{seriesId}                    -> 系列详情(含 poiDraftId, teams, status)
  - GET /stats/series/{seriesId}              -> 各队总分(points/position/kills)
  - GET /poi-drafts/{poiDraftId}/pick         -> 选点顺序(pickNumber/map/spawnLocation/team)

输出：data/raw/*.json（每场比赛一个文件）+ data/matches_manifest.json（汇总清单）
"""
import json
import re
import sys
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://prod-api.algstools.com/v1"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "data" / "raw"
SITEMAP = ROOT / "data" / "sitemap.xml"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36",
    "Accept": "application/json",
}


def http_get(path, retries=4):
    url = BASE + path
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} 失败: {last_err}")


def extract_series_ids():
    """从 sitemap 提取 Split 1 / Split 2 的 seriesId，返回 {slug: [(lastmod, seriesId), ...]}"""
    text = SITEMAP.read_text(encoding="utf-8")
    blocks = re.findall(r"<url>(.*?)</url>", text, re.S)
    out = {}
    for b in blocks:
        loc = re.search(r"<loc>(.*?)</loc>", b)
        lm = re.search(r"<lastmod>(.*?)</lastmod>", b)
        if not loc:
            continue
        m = loc.group(1)
        if "/match/" not in m:
            continue
        parts = m.replace("https://algs.ea.com/", "").split("/")
        slug = parts[2]
        if not (slug.startswith("split-1-pro-league") or slug.startswith("split-2-pro-league")):
            continue
        series_id = parts[-1]
        lastmod = lm.group(1) if lm else ""
        # 只保留 Year 6（2026 年）的比赛
        if not lastmod.startswith("2026"):
            continue
        out.setdefault(slug, []).append((lastmod, series_id))
    return out


def fetch_one(slug, series_id):
    """抓取一场比赛的完整数据，返回 dict 或 None（未完成时也返回带 status 的对象）。"""
    try:
        series = http_get(f"/series/{series_id}")
        rec = {
            "slug": slug,
            "seriesId": series_id,
            "series": series,
            "stats": None,
            "picks": None,
        }
        status = series.get("status")
        if status != "completed":
            return rec
        # 已结束：抓 stats 和 poi draft
        rec["stats"] = http_get(f"/stats/series/{series_id}")
        poi_draft_id = series.get("poiDraftId")
        if poi_draft_id:
            rec["picks"] = http_get(f"/poi-drafts/{poi_draft_id}/pick")
        return rec
    except Exception as e:  # noqa: BLE001
        return {"slug": slug, "seriesId": series_id, "error": str(e)}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    series_map = extract_series_ids()
    tasks = []
    for slug, arr in series_map.items():
        for _lastmod, sid in arr:
            tasks.append((slug, sid))
    print(f"共 {len(tasks)} 场比赛（Split 1 + Split 2, Year 6）", file=sys.stderr)

    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_one, slug, sid): (slug, sid) for slug, sid in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            slug, sid = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                rec = {"slug": slug, "seriesId": sid, "error": str(e)}
            results.append(rec)
            # 逐个落盘
            fname = f"{sid}.json"
            (RAW / fname).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            st = rec.get("series", {}).get("status") if rec.get("series") else "error"
            if i % 10 == 0 or st == "completed":
                print(f"[{i}/{len(tasks)}] {sid} -> {st}", file=sys.stderr)

    # 汇总清单
    manifest = []
    for rec in results:
        s = rec.get("series") or {}
        manifest.append({
            "slug": rec["slug"],
            "seriesId": rec["seriesId"],
            "status": s.get("status"),
            "name": s.get("name"),
            "seriesNumber": s.get("seriesNumber"),
            "poiDraftId": s.get("poiDraftId"),
            "error": rec.get("error"),
        })
    (ROOT / "data" / "matches_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    completed = [m for m in manifest if m["status"] == "completed"]
    pending = [m for m in manifest if m["status"] != "completed"]
    print(f"\n完成: {len(completed)} 场, 未完成/错误: {len(pending)} 场", file=sys.stderr)


if __name__ == "__main__":
    main()
