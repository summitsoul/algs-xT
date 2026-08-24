// 在 replay 页面(apexlegendsstatus.com/algs/game/...)的控制台粘贴运行。
// 作用: 补齐上一份文件缺失的数据 —— 队伍行动轨迹(state/pathing)、POI 位置(map-pois)、热力图(heatmap)、完整事件流(events)。
// 会自动下载一份 replay2_{gid}.json。
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

  // 先拿 duration 和 map 名
  const summary = await get("/api/summary");
  const dur = summary.duration;
  const mapName = summary.map;
  console.log("duration =", dur, "map =", mapName);

  // 1) 队伍行动轨迹 —— 逐秒全量状态 (每个玩家每秒钟的位置)
  console.log("fetching /api/state ...");
  const state = await safe(`/api/state?fromTs=0&toTs=${dur}`);

  // 2) 平滑轨迹 —— 每队的路径折线
  console.log("fetching /api/pathing ...");
  const pathing = await safe(`/api/pathing?fromTs=0&toTs=${dur}&teams=all&minStepSec=1`);

  // 3) POI 位置多边形 —— 用来算选点到圈中心距离的关键
  console.log("fetching /api/map-pois ...");
  const pois = await safe(`/api/map-pois?map=${encodeURIComponent(mapName)}`);

  // 4) 热力图 (fights 战斗 / looting 搜刮 / healing 打药)
  const heatmaps = {};
  for (const t of ["fights", "looting", "healing"]) {
    console.log(`fetching /api/heatmap?type=${t} ...`);
    heatmaps[t] = await safe(`/api/heatmap?type=${t}`);
  }

  // 5) 完整事件流 (分页拉全; 总量可能 2~4 万条, 体积较大)
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

  // 文件名: sp_{region}_d{day}_g{game}_{gid前8}.json  (region 小写, _ 转 -)
  const region = (summary.region || "").toLowerCase().replace(/_/g, "-");
  const day = String(summary.day ?? "").match(/(\d+)/)?.[0] ?? "?";
  const game = (summary.headerDisplayName?.match(/Game #(\d+)/) || [])[1] ?? "?";
  const fname = `sp_${region}_d${day}_g${game}_${GID.slice(0, 8)}.json`;

  const bundle = { gid: GID, fetchedAt: new Date().toISOString(), summary, state, pathing, pois, heatmaps, events };
  const blob = new Blob([JSON.stringify(bundle)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = fname;
  a.click();
  console.log("DONE -> " + fname,
    "| state.frames =", Array.isArray(state.frames) ? state.frames.length : state.error,
    "| pathing.teams =", pathing?.teams?.length,
    "| pois.ok =", pois?.ok,
    "| heatmaps =", Object.keys(heatmaps),
    "| events =", events.length);
})();
//我有一个问题啊，就是我刚才算了一下前11只队伍的xT值相加，发现才54，就算尽可能的考虑后面的9个队伍，总共的xT也才80多，而一场比赛应该至少一共可以得到109分，这牙膏是正常的吗

