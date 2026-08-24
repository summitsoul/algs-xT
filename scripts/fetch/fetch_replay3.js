// 在 apexlegendsstatus.com 的 replay 页面 ( /algs/game/{gid} ) 控制台 (F12 → Console) 粘贴运行。
// 一次性抓全 xT 模型所需的全部数据, 自动下载 replay_{gid}.json, 存到 replay/storm point/ 目录。
(async () => {
  const GID = window.REPLAY_GAME_ID;
  const JWT = window.REPLAY_API_JWT || "";
  const BASE = `https://algs-replay-api.apexlegendsstatus.com/${encodeURIComponent(GID)}`;
  const H = { Authorization: `Bearer ${JWT}` };

  const get = async (p) => {
    const r = await fetch(BASE + p, { credentials: "include", cache: "no-store", headers: H });
    if (!r.ok) throw new Error(`${p} -> HTTP ${r.status}`);
    return r.json();
  };
  const safe = async (p) => { try { return await get(p); } catch (e) { return { error: String(e) }; } };

  const summary = await get("/api/summary");
  const dur = summary.duration;
  const mapName = summary.map;
  console.log("gid =", GID, "| map =", mapName, "|", summary.headerDisplayName, "| duration =", dur);

  // 1) 队伍行动轨迹
  console.log("fetching /api/pathing ...");
  const pathing = await safe(`/api/pathing?fromTs=0&toTs=${dur}&teams=all&minStepSec=1`);

  // 2) 完整事件流 (击杀/倒地等, 分页拉全)
  console.log("fetching /api/events (分页) ...");
  const events = [];
  let offset = 0, limit = 500, total = Infinity;
  while (offset < total) {
    const r = await safe(`/api/events?offset=${offset}&limit=${limit}`);
    if (r.error) { events.push(r); break; }
    total = r.total ?? 0;
    const arr = r.events ?? [];
    if (!arr.length) break;
    events.push(...arr);
    offset += arr.length;
    console.log(`  events ${offset}/${total}`);
  }

  // 3) 时间线 (队伍淘汰顺序/时间 → 存活队数)
  console.log("fetching /api/timeline ...");
  const timeline = await safe("/api/timeline");

  // 4) 死亡盒 (死亡位置, 含 z)
  console.log("fetching /api/deathboxes ...");
  const deathboxes = await safe("/api/deathboxes");

  // 5) 圈型
  console.log("fetching /api/rings/all ...");
  const rings = await safe("/api/rings/all");

  // 6) POI 几何 (地图多边形)
  console.log("fetching /api/map-pois ...");
  const pois = await safe(`/api/map-pois?map=${encodeURIComponent(mapName)}`);

  const bundle = { gid: GID, fetchedAt: new Date().toISOString(), summary, pathing, events, timeline, deathboxes, rings, pois };
  const blob = new Blob([JSON.stringify(bundle)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `replay_${GID}.json`;
  a.click();
  console.log("DONE -> replay_" + GID + ".json",
    "| pathing.teams =", pathing?.teams?.length,
    "| events =", events.length,
    "| timeline =", timeline?.error ?? "ok",
    "| deathboxes =", Array.isArray(deathboxes) ? deathboxes.length : (deathboxes?.error ?? "ok"),
    "| pois.ok =", pois?.ok);
})();
