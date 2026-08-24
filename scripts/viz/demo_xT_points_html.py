#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_xT_points_html.py — 点位 xT 单场演示 (自包含 HTML)

用新模型: xT(点位p, 第m圈) = β_p + φ_m(rel_bin)  当 rel<=1 (圈内)
                          = 0                       当 rel>1 (毒里清零)

展示 (对照 make_demo.py 的交互, 但价值场换成"点位 xT"):
  - 地图 + 6 圈型 + 451 个点位 (随圈实时着色: 圈内红=高 xT, 灰=毒清零)
  - 点位着色两种模式切换: "随圈 xT" / "基值 β_p"
  - 全部队伍路线 + 吃鸡队高亮 + 吃鸡队击杀点
  - 全员 xT 曲线 (随时间, 随圈刷新变化)
  - 时间滑块/播放 + 缩放平移 + 图例悬停高亮
"""
import json, math, bisect, io, base64
import numpy as np

RATIO = 24.93
BUCKET = 5
REL_BINS = [0.3, 0.7, 1.0, 1.15, 1.6]
REL_NAMES = ['圈心', '圈内', '圈内边缘', '圈外贴边', '圈外', '远圈外']
GAME = "replay/storm point/sp_na_d__g7_5542abdb.json"
MAP_PNG = "map/storm point.png"
BETA_NPY = "data/points_beta.npy"
POS_NPY = "data/points_pos.npy"
PHI_JSON = "data/phi.json"
OUT_HTML = "output/demo_xT_points.html"
BMAX = 3.0       # β_p 色标上限 (对应对 -3 ~ +3)
FALLBACK = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
            '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
            '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1']


def w2i(x, y):
    return (x / RATIO + 2048, -y / RATIO + 2048)


def r2i(r):
    return r / RATIO


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def centroid(ps):
    return (sum(p[0] for p in ps) / len(ps), sum(p[1] for p in ps) / len(ps))


def team_anchor(ps):
    n = len(ps)
    if n == 0:
        return None
    if n <= 2:
        return centroid(ps)
    p0, p1, p2 = ps
    d01, d02, d12 = dist(p0, p1), dist(p0, p2), dist(p1, p2)
    if d01 <= d02 and d01 <= d12:
        return centroid([p0, p1])
    if d02 <= d01 and d02 <= d12:
        return centroid([p0, p2])
    return centroid([p1, p2])


def pos_at(pts, ts, t):
    if t <= ts[0]:
        return (pts[0]['x'], pts[0]['y'])
    if t >= ts[-1]:
        return (pts[-1]['x'], pts[-1]['y'])
    i = bisect.bisect_left(ts, t)
    a, b = pts[i - 1], pts[i]
    frac = (t - a['tsGame']) / (b['tsGame'] - a['tsGame'])
    return (a['x'] + frac * (b['x'] - a['x']), a['y'] + frac * (b['y'] - a['y']))


def rel_bin(rel):
    for i, th in enumerate(REL_BINS):
        if rel < th:
            return i
    return len(REL_BINS)


def main():
    # ---- 模型产物 ----
    beta = np.load(BETA_NPY)
    pos_arr = np.load(POS_NPY)
    phij = json.load(open(PHI_JSON, encoding='utf-8'))['phi']
    PHI = [[phij.get(f"{ph},{rb}", 0.0) for rb in range(6)] for ph in range(6)]
    NP = len(beta)

    d = json.load(open(GAME, encoding='utf-8'))
    s = d['summary']
    dur = s['duration']
    g0 = s['gameStartTs']
    header = s['headerDisplayName']

    winner = next(t for t in s['teams'] if t['placement'] == 1)
    wid = winner['teamId']

    # ---- 圈时间线 ----
    sc, fc = {}, {}
    for r in s['ringPhases']:
        if r['type'] == 'startClosing':
            sc[r['stage']] = r
        elif r['type'] == 'finishedClosing':
            fc[r['stage']] = r
    order = sorted(sc.keys())
    stages = {}
    for k in order:
        a, b = sc[k], fc.get(k)
        c0 = (a['center']['x'], a['center']['y'])
        c1 = (b['center']['x'], b['center']['y']) if b else c0
        stages[k] = {'c0': c0, 'c1': c1, 'r0': a['startRadius'], 'r1': a['endRadius'],
                     't0': a['timestamp'] - g0, 't1': (b['timestamp'] - g0) if b else dur}

    def ring_at(t):
        for i in range(len(order) - 1, -1, -1):
            st = stages[order[i]]
            if t >= st['t0']:
                p = min(1.0, (t - st['t0']) / max(st['t1'] - st['t0'], 1e-6))
                cx = st['c0'][0] + (st['c1'][0] - st['c0'][0]) * p
                cy = st['c0'][1] + (st['c1'][1] - st['c0'][1]) * p
                r = st['r0'] + (st['r1'] - st['r0']) * p
                return i, (cx, cy), r
        st = stages[order[0]]
        return 0, st['c0'], st['r0']

    rings, stages_js = [], []
    for k in order:
        st = stages[k]
        cx, cy = w2i(st['c1'][0], st['c1'][1])
        rings.append({'n': k + 1, 'cx': cx, 'cy': cy, 'r': r2i(st['r1']), 't': st['t0']})
        c0x, c0y = w2i(st['c0'][0], st['c0'][1])
        c1x, c1y = w2i(st['c1'][0], st['c1'][1])
        stages_js.append({'t0': st['t0'], 't1': st['t1'],
                          'c0x': c0x, 'c0y': c0y, 'c1x': c1x, 'c1y': c1y,
                          'r0': r2i(st['r0']), 'r1': r2i(st['r1'])})

    # ---- 全部队伍轨迹 + 点位 xT ----
    meta = {t['teamId']: t for t in s['teams']}
    teams = []
    for i, t in enumerate(d['pathing']['teams']):
        tid = t['teamId']
        players, max_last = [], 0
        for pl in t['players']:
            pts = pl.get('points') or []
            if not pts:
                continue
            pts.sort(key=lambda p: p['tsGame'])
            players.append({'ts': [p['tsGame'] for p in pts], 'pts': pts})
            max_last = max(max_last, pts[-1]['tsGame'])
        if not players:
            continue
        elim = min(max_last, dur)
        mt = meta.get(tid, {})
        name = mt.get('teamName', f'T{tid}')
        color = mt.get('color') or FALLBACK[i % len(FALLBACK)]
        placement = mt.get('placement', 20)
        tpath = []
        for b in range(int(dur // BUCKET) + 1):
            tt = b * BUCKET
            if tt > elim:
                break
            ph, c, r = ring_at(tt)
            alive = [pos_at(pl['pts'], pl['ts'], tt) for pl in players
                     if pl['ts'][0] <= tt <= pl['ts'][-1]]
            if not alive:
                continue
            ax, ay = team_anchor(alive)
            rel = dist((ax, ay), c) / r
            rb = rel_bin(rel)
            if rel <= 1.0:
                dd = np.hypot(pos_arr[:, 0] - ax, pos_arr[:, 1] - ay)
                pi = int(np.argmin(dd))
                xT = beta[pi] + PHI[ph][rb]
            else:
                xT = 0.0
            ix, iy = w2i(ax, ay)
            tpath.append({'t': tt, 'x': ix, 'y': iy, 'phase': ph + 1, 'rel': rel,
                          'rel_bin': rb, 'xT': xT})
        teams.append({'id': tid, 'name': name, 'color': color, 'placement': placement,
                      'elim': elim, 'path': tpath})
    teams.sort(key=lambda x: x['placement'])

    # 吃鸡队击杀
    kills = []
    for e in d['events']:
        if e.get('category') == 'playerKilled' and e['actor']['teamId'] == wid:
            pa = e.get('posActor') or {}
            if 'x' in pa:
                ix, iy = w2i(pa['x'], pa['y'])
                kills.append({'t': e['tsGame'], 'x': ix, 'y': iy,
                              'victim': e['target'].get('teamName', '?')})

    wteam = next(t for t in teams if t['id'] == wid)

    # 全队 xT 范围 + 动态 Y 轴 (β_p 可负 -> xT 圈内可为负, 给负轴留量)
    all_xt = [p['xT'] for t in teams for p in t['path']]
    xt_max = max(all_xt) if all_xt else 10.0
    xt_min = min(all_xt) if all_xt else 0.0
    YMAX = max(12.0, math.ceil(xt_max / 2) * 2)
    YMIN = math.floor(xt_min) if xt_min < 0 else 0.0

    # 打印验证 (确认队伍之间 xT 有区分度)
    means = sorted(((sum(p['xT'] for p in t['path']) / len(t['path']), t['name']) for t in teams))
    print(f"吃鸡队 {wteam['name']} | 队伍数 {len(teams)} | 击杀 {len(kills)}")
    print(f"xT 范围: {min(all_xt):.2f} ~ {max(all_xt):.2f} | 图表 Y=[{YMIN}, {YMAX}]")
    print("队伍 xT 均值 (低→高):")
    for m, nm in means:
        mark = "  <-- 吃鸡" if nm == wteam['name'] else ""
        print(f"  {nm:24} {m:5.2f}{mark}")

    # ---- 点位 (世界坐标 -> 图像) ----
    px, py, pb = [], [], []
    for i in range(NP):
        ix, iy = w2i(pos_arr[i, 0], pos_arr[i, 1])
        px.append(round(ix, 1)); py.append(round(iy, 1)); pb.append(round(float(beta[i]), 4))

    # ---- 色带 (xT 顺序 + β_p 发散) ----
    from matplotlib import colormaps as _cm
    XTRAMP, BETARAMP = [], []
    for i in range(256):
        t = 0.15 + 0.85 * i / 255
        r, g, b, _ = _cm['YlOrRd'](t)
        XTRAMP.append(f'rgb({int(r*255)},{int(g*255)},{int(b*255)})')
    for i in range(256):
        r, g, b, _ = _cm['RdBu_r'](i / 255)
        BETARAMP.append(f'rgb({int(r*255)},{int(g*255)},{int(b*255)})')
    xt_grad = 'linear-gradient(90deg,' + ','.join(XTRAMP) + ')'
    beta_grad = 'linear-gradient(90deg,' + ','.join(BETARAMP) + ')'

    # ---- 内嵌地图 ----
    from PIL import Image
    _raw = Image.open(MAP_PNG).convert('RGBA')
    _bg = Image.new('RGBA', _raw.size, (13, 15, 19, 255))
    _flat = Image.alpha_composite(_bg, _raw).convert('RGB').resize((2048, 2048), Image.LANCZOS)
    _buf = io.BytesIO()
    _flat.save(_buf, 'JPEG', quality=82, optimize=True)
    map_b64 = base64.b64encode(_buf.getvalue()).decode()

    rings_json = json.dumps(rings)
    stages_json = json.dumps(stages_js)
    teams_json = json.dumps(teams)
    wteam_json = json.dumps(wteam)
    kills_json = json.dumps(kills)
    px_json = json.dumps(px); py_json = json.dumps(py); pb_json = json.dumps(pb)
    phi_json = json.dumps(PHI)
    xtramp_json = json.dumps(XTRAMP); betaramp_json = json.dumps(BETARAMP)
    relbins_json = json.dumps(REL_BINS)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>点位 xT — {header}</title>
<style>
:root{{--bg:#0d0f13;--panel:#161a22;--ink:#e8eaee;--muted:#98a0ab;--accent:#CCFF33;--kill:#ff5252;}}
*{{box-sizing:border-box;}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;}}
header{{padding:14px 20px;border-bottom:1px solid #232834;display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap;}}
header h1{{margin:0;font-size:16px;font-weight:600;}}
header .meta{{color:var(--muted);font-size:12px;}}
.layout{{display:grid;grid-template-columns:1fr 400px;gap:0;min-height:calc(100vh - 54px);}}
@media(max-width:900px){{.layout{{grid-template-columns:1fr;}}}}
.mapcol{{padding:12px;position:relative;}}
.mapwrap{{position:relative;}}
.mapwrap .map{{width:100%;max-height:82vh;display:block;border-radius:10px;cursor:grab;touch-action:none;}}
.mapwrap .map:active{{cursor:grabbing;}}
.zctrl{{position:absolute;top:12px;right:12px;display:flex;flex-direction:column;gap:6px;}}
.zctrl button{{width:34px;height:34px;background:rgba(22,26,34,.9);color:var(--ink);border:1px solid #2a3142;border-radius:8px;cursor:pointer;font-size:17px;line-height:1;}}
.zctrl button:hover{{border-color:#4a5468;}}
.mapui{{position:absolute;bottom:12px;left:12px;display:flex;flex-direction:column;gap:8px;align-items:flex-start;}}
.mapui .row{{display:flex;gap:8px;align-items:center;}}
.mapui label{{background:rgba(22,26,34,.9);border:1px solid #2a3142;border-radius:8px;padding:6px 10px;font-size:12px;color:var(--muted);display:flex;align-items:center;gap:6px;cursor:pointer;}}
.mapui input{{accent-color:var(--accent);}}
.seg{{display:flex;background:rgba(22,26,34,.9);border:1px solid #2a3142;border-radius:8px;overflow:hidden;}}
.seg button{{background:transparent;color:var(--muted);border:0;padding:6px 12px;font-size:12px;cursor:pointer;}}
.seg button.on{{background:#1c2130;color:var(--ink);}}
.legend{{background:rgba(22,26,34,.9);border:1px solid #2a3142;border-radius:8px;padding:6px 10px;font-size:11px;color:var(--muted);}}
.legend .bar{{display:inline-block;width:140px;height:8px;border-radius:4px;vertical-align:middle;margin:0 6px;}}
.legend .g{{color:var(--muted);}}
.chartcol{{background:var(--panel);border-left:1px solid #232834;padding:14px;display:flex;flex-direction:column;gap:12px;overflow-y:auto;}}
.chart h3{{margin:0;font-size:13px;color:var(--muted);font-weight:500;}}
.chart svg{{width:100%;height:230px;display:block;}}
.controls{{display:flex;align-items:center;gap:10px;}}
.controls button{{background:#1c2130;color:var(--ink);border:1px solid #2a3142;border-radius:6px;padding:6px 12px;cursor:pointer;font-size:13px;}}
.controls button:hover{{border-color:#4a5468;}}
.controls input[type=range]{{flex:1;}}
.tlist{{display:flex;flex-direction:column;gap:2px;max-height:240px;overflow-y:auto;}}
.titem{{display:flex;align-items:center;gap:8px;padding:4px 8px;border-radius:6px;cursor:pointer;font-size:12px;color:var(--muted);}}
.titem:hover{{background:#1c2130;}}
.titem.on{{background:#1c2130;color:var(--ink);}}
.titem .dot{{width:9px;height:9px;border-radius:50%;flex:0 0 auto;}}
.titem .nm{{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}}
.titem .pl{{color:var(--muted);font-size:11px;}}
.titem .v{{font-variant-numeric:tabular-nums;font-size:11px;}}
</style></head><body>
<header>
  <h1>点位 xT 演示 — {header}</h1>
  <div class="meta">20 队 · 点位 xT = β_p + φ(圈相对) 随圈刷新 · 毒区清零 · 吃鸡 {wteam['name']}</div>
</header>
<div class="layout">
  <div class="mapcol">
    <div class="mapwrap">
      <svg class="map" id="map" viewBox="0 0 4096 4096" preserveAspectRatio="xMidYMid meet">
        <g id="world">
          <image href="data:image/png;base64,{map_b64}" x="0" y="0" width="4096" height="4096"/>
          <g id="points"></g>
          <g id="rings"></g>
          <g id="paths"></g>
          <g id="kills"></g>
          <g id="spot"></g>
        </g>
      </svg>
      <div class="zctrl">
        <button id="zin" title="放大">+</button>
        <button id="zout" title="缩小">−</button>
        <button id="zreset" title="重置视图">⤢</button>
      </div>
      <div class="mapui">
        <div class="seg" id="mode">
          <button id="mxt" class="on">随圈 xT</button>
          <button id="mbeta">基值 β_p</button>
        </div>
        <label><input type="checkbox" id="pttoggle" checked> 点位层</label>
        <div class="legend" id="legend"><span id="lg0">低</span><span class="bar" id="lgbar"></span><span id="lg1">高</span><span class="g" id="lggray"> · 灰=毒清零</span></div>
      </div>
    </div>
  </div>
  <div class="chartcol">
    <div class="chart"><h3>全员 xT 曲线 (期望未来分 · 随圈刷新 + 点位切换)</h3><svg id="chart" viewBox="0 0 360 240" preserveAspectRatio="none"></svg></div>
    <div class="controls">
      <button id="play">▶ 播放</button>
      <input type="range" id="slider" min="0" max="{dur//BUCKET}" value="0" step="1">
    </div>
    <div class="chart"><h3>队伍 (按名次 · 悬停高亮)</h3><div class="tlist" id="tlist"></div></div>
  </div>
</div>
<script>
const teams={teams_json};
const rings={rings_json};
const stages={stages_json};
const kills={kills_json};
const dur={dur};
const PX={px_json};
const PY={py_json};
const PB={pb_json};
const PHI={phi_json};
const XTRAMP={xtramp_json};
const BETARAMP={betaramp_json};
const RELBINS={relbins_json};
const YMAX={YMAX};
const YMIN={YMIN};
const BMAX={BMAX};
const NP={NP};
const PHASE_COLOR=['#8a93a0','#b58900','#d33682','#6c71c4','#dc322f','#2aa198'];

function relbin(rel){{for(let i=0;i<RELBINS.length;i++){{if(rel<RELBINS[i])return i;}}return RELBINS.length;}}

// 圈
const rg=document.getElementById('rings');
rings.forEach((r,i)=>{{
  const c=document.createElementNS('http://www.w3.org/2000/svg','circle');
  c.setAttribute('cx',r.cx);c.setAttribute('cy',r.cy);c.setAttribute('r',Math.max(r.r,3));
  c.setAttribute('fill','none');c.setAttribute('stroke',PHASE_COLOR[r.n-1]);
  c.setAttribute('stroke-width', i===rings.length-1?5:2.5);
  c.setAttribute('stroke-dasharray', i===rings.length-1?'none':'7 6');
  c.setAttribute('opacity', i===rings.length-1?0.95:0.8);
  rg.appendChild(c);
  const t=document.createElementNS('http://www.w3.org/2000/svg','text');
  t.setAttribute('x',r.cx+r.r);t.setAttribute('y',r.cy);t.setAttribute('font-size','26');
  t.setAttribute('fill',PHASE_COLOR[r.n-1]);t.setAttribute('font-weight','700');
  t.textContent='圈'+r.n;rg.appendChild(t);
}});

// 点位层
const pg0=document.getElementById('points');
const ptEls=[];
for(let i=0;i<NP;i++){{
  const c=document.createElementNS('http://www.w3.org/2000/svg','circle');
  c.setAttribute('cx',PX[i]);c.setAttribute('cy',PY[i]);c.setAttribute('r',6);
  c.setAttribute('fill','#3a3f47');
  pg0.appendChild(c);ptEls.push(c);
}}

function ringAt(t){{
  for(let i=stages.length-1;i>=0;i--){{
    const s=stages[i];
    if(t>=s.t0){{
      const p=Math.min(1,(t-s.t0)/Math.max(s.t1-s.t0,1e-6));
      return {{phase:i, cx:s.c0x+(s.c1x-s.c0x)*p, cy:s.c0y+(s.c1y-s.c0y)*p, r:s.r0+(s.r1-s.r0)*p}};
    }}
  }}
  return {{phase:0, cx:stages[0].c0x, cy:stages[0].c0y, r:stages[0].r0}};
}}

let mode='xT';
function colorPoint(i, rng){{
  if(mode==='beta'){{
    const t=Math.max(0,Math.min(255,Math.round((PB[i]+BMAX)/(2*BMAX)*255)));
    return BETARAMP[t];
  }}
  const rel=Math.hypot(PX[i]-rng.cx,PY[i]-rng.cy)/rng.r;
  if(rel>1.0)return '#3a3f47';
  const xT=PB[i]+PHI[rng.phase][relbin(rel)];
  const t=Math.max(0,Math.min(255,Math.round(xT/YMAX*255)));
  return XTRAMP[t];
}}
function updatePoints(rng){{
  for(let i=0;i<NP;i++)ptEls[i].setAttribute('fill',colorPoint(i,rng));
}}
function updatePointsBeta(){{
  for(let i=0;i<NP;i++)ptEls[i].setAttribute('fill',colorPoint(i,null));
}}

// 全员路线
const pg=document.getElementById('paths');
const teamLines={{}};
teams.forEach((tm)=>{{
  const pl=document.createElementNS('http://www.w3.org/2000/svg','polyline');
  const dd=tm.path.map((p,i)=>(i?'L':'M')+p.x.toFixed(1)+' '+p.y.toFixed(1)).join(' ');
  pl.setAttribute('points',dd);
  pl.setAttribute('fill','none');pl.setAttribute('stroke',tm.color);
  pl.setAttribute('stroke-width', tm.id==={wid}?6:3);
  pl.setAttribute('stroke-linecap','round');pl.setAttribute('stroke-linejoin','round');
  pl.setAttribute('opacity', tm.id==={wid}?0.95:0.5);
  pg.appendChild(pl);teamLines[tm.id]=pl;
}});

// 击杀点(吃鸡队)
const kg=document.getElementById('kills');
kills.forEach(k=>{{
  const c=document.createElementNS('http://www.w3.org/2000/svg','circle');
  c.setAttribute('cx',k.x);c.setAttribute('cy',k.y);c.setAttribute('r',14);
  c.setAttribute('fill','none');c.setAttribute('stroke','var(--kill)');c.setAttribute('stroke-width',4);
  kg.appendChild(c);
}});

// 当前点(吃鸡队位置, 随播放)
const spot=document.getElementById('spot');
const dot=document.createElementNS('http://www.w3.org/2000/svg','circle');
dot.setAttribute('r',18);dot.setAttribute('fill','var(--accent)');dot.setAttribute('stroke','#000');dot.setAttribute('stroke-width',3);
spot.appendChild(dot);

// xT 曲线 (全员)
const chart=document.getElementById('chart');
const W=360,H=240;
function X(t){{return t/dur*W;}}
function Y(v){{return H-8-(v-YMIN)/(YMAX-YMIN)*(H-16);}}
for(let i=0;i<=4;i++){{
  const vv=YMIN+(YMAX-YMIN)*i/4;
  const g=document.createElementNS('http://www.w3.org/2000/svg','line');
  g.setAttribute('x1',0);g.setAttribute('x2',W);g.setAttribute('y1',Y(vv));g.setAttribute('y2',Y(vv));
  g.setAttribute('stroke','#2a3142');g.setAttribute('stroke-width',1);chart.appendChild(g);
  const tx=document.createElementNS('http://www.w3.org/2000/svg','text');
  tx.setAttribute('x',W-4);tx.setAttribute('y',Y(vv)-3);tx.setAttribute('font-size','10');
  tx.setAttribute('fill','#98a0ab');tx.setAttribute('text-anchor','end');tx.textContent=vv.toFixed(1);chart.appendChild(tx);
}}
// 零值基准线 (毒清零 / 负值参考)
const zline=document.createElementNS('http://www.w3.org/2000/svg','line');
zline.setAttribute('x1',0);zline.setAttribute('x2',W);zline.setAttribute('y1',Y(0));zline.setAttribute('y2',Y(0));
zline.setAttribute('stroke','#3a3f47');zline.setAttribute('stroke-width',1);
chart.appendChild(zline);
rings.forEach(r=>{{
  const l=document.createElementNS('http://www.w3.org/2000/svg','line');
  l.setAttribute('x1',X(r.t));l.setAttribute('x2',X(r.t));l.setAttribute('y1',0);l.setAttribute('y2',H);
  l.setAttribute('stroke',PHASE_COLOR[r.n-1]);l.setAttribute('stroke-width',1);l.setAttribute('stroke-dasharray','3 3');l.setAttribute('opacity',0.4);
  chart.appendChild(l);
}});
const chartLines={{}};
teams.forEach((tm)=>{{
  const pl=document.createElementNS('http://www.w3.org/2000/svg','polyline');
  pl.setAttribute('points', tm.path.map(p=>X(p.t).toFixed(1)+','+Y(p.xT).toFixed(1)).join(' '));
  pl.setAttribute('fill','none');pl.setAttribute('stroke',tm.color);
  pl.setAttribute('stroke-width', tm.id==={wid}?2.8:1.3);
  pl.setAttribute('opacity', tm.id==={wid}?1.0:0.55);
  chart.appendChild(pl);chartLines[tm.id]=pl;
}});
const cur=document.createElementNS('http://www.w3.org/2000/svg','line');
cur.setAttribute('y1',0);cur.setAttribute('y2',H);cur.setAttribute('stroke','#e8eaee');cur.setAttribute('stroke-width',1.5);
chart.appendChild(cur);

// 图例列表
const tl=document.getElementById('tlist');
teams.forEach((tm)=>{{
  const it=document.createElement('div');it.className='titem';it.dataset.id=tm.id;
  const peak=tm.path.reduce((m,p)=>Math.max(m,p.xT),-1);
  it.innerHTML='<span class="dot" style="background:'+tm.color+'"></span>'
    +'<span class="nm">'+tm.name+'</span>'
    +'<span class="v">峰值 '+peak.toFixed(1)+'</span>'
    +'<span class="pl">#'+tm.placement+'</span>';
  it.onmouseenter=()=>highlight(tm.id);
  it.onmouseleave=()=>highlight(null);
  tl.appendChild(it);
}});
function highlight(tid){{
  teams.forEach((tm)=>{{
    const on=tid==null||tm.id===tid;
    chartLines[tm.id].setAttribute('opacity', on?(tm.id==={wid}?1.0:0.9):0.08);
    teamLines[tm.id].setAttribute('opacity', on?(tm.id==={wid}?0.95:0.85):0.06);
  }});
  [...tl.children].forEach(el=>el.classList.toggle('on', el.dataset.id==tid));
}}

// 播放
const slider=document.getElementById('slider');
function update(i){{
  const t=i*{BUCKET};
  const rng=ringAt(t);
  if(mode==='xT')updatePoints(rng);else updatePointsBeta();
  cur.setAttribute('x1',X(t));cur.setAttribute('x2',X(t));
  const wt={wteam_json};
  const p=wt.path.find(q=>Math.abs(q.t-t)<=2)||wt.path[0];
  if(p){{dot.setAttribute('cx',p.x);dot.setAttribute('cy',p.y);}}
  slider.value=i;
}}
let timer=null,playing=false;
function tick(){{
  let i=parseInt(slider.value)+1;
  if(i>={dur//BUCKET}){{stop();return;}}
  update(i);
}}
function stop(){{clearInterval(timer);playing=false;document.getElementById('play').textContent='▶ 播放';}}
document.getElementById('play').onclick=()=>{{
  if(playing){{stop();return;}}
  if(parseInt(slider.value)>={dur//BUCKET})slider.value=0;
  playing=true;document.getElementById('play').textContent='⏸ 暂停';
  timer=setInterval(tick,55);
}};
slider.oninput=()=>update(parseInt(slider.value));

// 模式切换
const lgbar=document.getElementById('lgbar');
const lg0=document.getElementById('lg0'),lg1=document.getElementById('lg1'),lgg=document.getElementById('lggray');
function setMode(m){{
  mode=m;
  document.getElementById('mxt').classList.toggle('on',m==='xT');
  document.getElementById('mbeta').classList.toggle('on',m==='beta');
  if(m==='xT'){{
    lgbar.style.background='{xt_grad}';lg0.textContent='0';lg1.textContent=YMAX;lgg.style.display='inline';
  }}else{{
    lgbar.style.background='{beta_grad}';lg0.textContent='-'+BMAX;lg1.textContent='+'+BMAX;lgg.style.display='none';
  }}
  update(parseInt(slider.value));
}}
document.getElementById('mxt').onclick=()=>setMode('xT');
document.getElementById('mbeta').onclick=()=>setMode('beta');
setMode('xT');

// 缩放/平移
const svg=document.getElementById('map');
const world=document.getElementById('world');
let scale=1, tx=0, ty=0;
function applyT(){{world.setAttribute('transform','translate('+tx+' '+ty+') scale('+scale+')');}}
function viewPt(e){{
  const pt=svg.createSVGPoint(); pt.x=e.clientX; pt.y=e.clientY;
  return pt.matrixTransform(svg.getScreenCTM().inverse());
}}
svg.addEventListener('wheel',(e)=>{{
  e.preventDefault();
  const v=viewPt(e);
  const ns=Math.min(14,Math.max(0.4,scale*(e.deltaY<0?1.25:0.8)));
  const wx=(v.x-tx)/scale, wy=(v.y-ty)/scale;
  scale=ns; tx=v.x-wx*ns; ty=v.y-wy*ns; applyT();
}},{{passive:false}});
let drag=null;
svg.addEventListener('mousedown',(e)=>{{drag={{v:viewPt(e),tx0:tx,ty0:ty}};}});
window.addEventListener('mousemove',(e)=>{{if(!drag)return;const v=viewPt(e);tx=drag.tx0+(v.x-drag.v.x);ty=drag.ty0+(v.y-drag.v.y);applyT();}});
window.addEventListener('mouseup',()=>{{drag=null;}});
function zoomBy(f){{
  const v={{x:2048,y:2048}};
  const ns=Math.min(14,Math.max(0.4,scale*f));
  tx=v.x-(v.x-tx)*(ns/scale); ty=v.y-(v.y-ty)*(ns/scale); scale=ns; applyT();
}}
document.getElementById('zin').onclick=()=>zoomBy(1.5);
document.getElementById('zout').onclick=()=>zoomBy(1/1.5);
document.getElementById('zreset').onclick=()=>{{scale=1;tx=0;ty=0;applyT();}};
document.getElementById('pttoggle').onchange=(e)=>pg0.setAttribute('opacity', e.target.checked?1:0);

update(0);
</script></body></html>"""

    open(OUT_HTML, 'w', encoding='utf-8').write(html)
    print(f"HTML 已生成 {OUT_HTML}  ({len(html)//1024} KB)")


if __name__ == '__main__':
    main()
