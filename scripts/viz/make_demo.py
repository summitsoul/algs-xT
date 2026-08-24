#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_demo.py — 全场所有队伍的 xT 演示 (自包含 HTML, 修正版)

展示: 地图 + 6 圈型 + 实时价值场(随圈) + 全部队伍路线 + 全员 xT 曲线
  - 吃鸡队高亮; 悬停图例可单独高亮某队
  - xT = V(phase, rel) 相对本场圈(漂移+收缩插值)
"""
import json, math, bisect, io, base64

RATIO = 24.93
BUCKET = 5
REL_BINS = [0.3, 0.7, 1.0, 1.15, 1.6]
REL_NAMES = ['圈心', '圈内', '圈内边缘', '圈外贴边', '圈外', '远圈外']
BAND_MULT = [0.3, 0.7, 1.0, 1.15, 1.6, 3.0]
GAME = "replay/storm point/sp_na_d__g7_5542abdb.json"
MAP_PNG = "map/storm point.png"
VRING = "data/v_ring.json"
OUT_HTML = "output/demo_xT_all.html"
OUT_PNG = "output/verify_all.png"
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
    d = json.load(open(GAME, encoding='utf-8'))
    s = d['summary']
    dur = s['duration']
    g0 = s['gameStartTs']
    header = s['headerDisplayName']

    winner = next(t for t in s['teams'] if t['placement'] == 1)
    wid = winner['teamId']

    # ---- 圈时间线 (漂移 + 收缩插值) ----
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

    # ---- xT 表 + 价值场颜色 ----
    vr = json.load(open(VRING, encoding='utf-8'))
    Vtab = {tuple(map(int, k.split(','))): v for k, v in vr['V'].items()}
    from matplotlib import colormaps as _cm
    _cmap = _cm['coolwarm']
    VGRID = [[Vtab.get((ph, rb)) for rb in range(len(REL_NAMES))] for ph in range(6)]
    VCOLORS = []
    for ph in range(6):
        row = []
        for rb in range(len(REL_NAMES)):
            v = VGRID[ph][rb]
            if v is None:
                row.append('rgba(0,0,0,0)')
            else:
                t = max(0.0, min(1.0, v / 10.0))
                r, g, b, _ = _cmap(t)
                row.append(f'rgb({int(r*255)},{int(g*255)},{int(b*255)})')
        VCOLORS.append(row)

    # ---- 全部队伍轨迹 + xT ----
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
            V = Vtab.get((ph, rb))
            ix, iy = w2i(ax, ay)
            tpath.append({'t': tt, 'x': ix, 'y': iy, 'phase': ph + 1, 'rel': rel,
                          'rel_bin': rb, 'rel_name': REL_NAMES[rb], 'V': V})
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
    print(f"吃鸡队 {wteam['name']} | 队伍数 {len(teams)} | 击杀 {len(kills)}")
    print(f"xT 全程范围: {min((p['V'] for t in teams for p in t['path'] if p['V'] is not None), default=0):.2f} ~ {max((p['V'] for t in teams for p in t['path'] if p['V'] is not None), default=0):.2f}")

    # ---- 校验 PNG ----
    try:
        from PIL import Image, ImageDraw
        raw = Image.open(MAP_PNG).convert('RGBA')
        bg = Image.new('RGBA', raw.size, (13, 15, 19, 255))
        im = Image.alpha_composite(bg, raw).convert('RGB')
        dr = ImageDraw.Draw(im, 'RGBA')
        rc = ["#9aa0a8", "#b58900", "#d33682", "#6c71c4", "#dc322f", "#2aa198"]
        for r in rings:
            dr.ellipse([r['cx'] - r['r'], r['cy'] - r['r'], r['cx'] + r['r'], r['cy'] + r['r']],
                       outline=rc[r['n'] - 1], width=6)
        for t in teams:
            col = tuple(int(t['color'][j:j + 2], 16) for j in (1, 3, 5)) + (200,)
            for i in range(1, len(t['path'])):
                dr.line([t['path'][i - 1]['x'], t['path'][i - 1]['y'],
                         t['path'][i]['x'], t['path'][i]['y']], fill=col, width=5)
        for k in kills:
            dr.ellipse([k['x'] - 14, k['y'] - 14, k['x'] + 14, k['y'] + 14], outline=(255, 60, 60, 255), width=6)
        im.save(OUT_PNG)
        print(f"校验图 {OUT_PNG}")
    except Exception as e:
        print(f"PNG 校验失败(跳过): {e}")

    # ---- 内嵌地图 ----
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
    vgrid_json = json.dumps(VGRID)
    vcolors_json = json.dumps(VCOLORS)
    band_mult_json = json.dumps(BAND_MULT)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>全场 xT — {header}</title>
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
.mapui label{{background:rgba(22,26,34,.9);border:1px solid #2a3142;border-radius:8px;padding:6px 10px;font-size:12px;color:var(--muted);display:flex;align-items:center;gap:6px;cursor:pointer;}}
.mapui input{{accent-color:var(--accent);}}
.vlegend{{background:rgba(22,26,34,.9);border:1px solid #2a3142;border-radius:8px;padding:6px 10px;font-size:11px;color:var(--muted);}}
.vlegend .bar{{display:inline-block;width:120px;height:8px;border-radius:4px;background:linear-gradient(90deg,#3b4cc0,#f7f7f7,#b40426);vertical-align:middle;margin:0 6px;}}
.vlegend span{{vertical-align:middle;}}
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
  <h1>全场 xT 演示 — {header}</h1>
  <div class="meta">20 队 · 圈型 + 实时价值场 + 全员路线/xT · 悬停图例高亮某队 · 吃鸡 {wteam['name']}</div>
</header>
<div class="layout">
  <div class="mapcol">
    <div class="mapwrap">
      <svg class="map" id="map" viewBox="0 0 4096 4096" preserveAspectRatio="xMidYMid meet">
        <g id="world">
          <image href="data:image/png;base64,{map_b64}" x="0" y="0" width="4096" height="4096"/>
          <g id="valuefield" opacity="0.30"></g>
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
        <label><input type="checkbox" id="vftoggle" checked> 价值场(本场刷圈)</label>
        <div class="vlegend"><span>低价值</span><span class="bar"></span><span>高价值</span></div>
      </div>
    </div>
  </div>
  <div class="chartcol">
    <div class="chart"><h3>全员 xT 曲线 (期望未来分 · 随圈/点位变化)</h3><svg id="chart" viewBox="0 0 360 240" preserveAspectRatio="none"></svg></div>
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
const VGRID={vgrid_json};
const VCOLORS={vcolors_json};
const BANDMULT={band_mult_json};
const PHASE_COLOR=['#8a93a0','#b58900','#d33682','#6c71c4','#dc322f','#2aa198'];

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

// 价值场
const vf=document.getElementById('valuefield');
const bandEls=[];
for(let rb=BANDMULT.length-1;rb>=0;rb--){{
  const c=document.createElementNS('http://www.w3.org/2000/svg','circle');
  c.setAttribute('cx',0);c.setAttribute('cy',0);c.setAttribute('r',0);
  vf.appendChild(c);bandEls.push({{el:c,rb:rb}});
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
function updateField(t){{
  const rng=ringAt(t);
  bandEls.forEach((b)=>{{
    b.el.setAttribute('cx',rng.cx);b.el.setAttribute('cy',rng.cy);
    b.el.setAttribute('r',Math.max(BANDMULT[b.rb]*rng.r,2));b.el.setAttribute('fill',VCOLORS[rng.phase][b.rb]);
  }});
}}

// 全员路线
const pg=document.getElementById('paths');
const teamLines={{}};
teams.forEach((tm)=>{{
  const pl=document.createElementNS('http://www.w3.org/2000/svg','polyline');
  const d=tm.path.map((p,i)=>(i?'L':'M')+p.x.toFixed(1)+' '+p.y.toFixed(1)).join(' ');
  pl.setAttribute('points',d);
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
const YMAX=10;
function X(t){{return t/dur*W;}}
function Y(v){{return H-8-(v/YMAX)*(H-16);}}
// 网格
for(let i=0;i<=4;i++){{
  const vv=YMAX*i/4;
  const g=document.createElementNS('http://www.w3.org/2000/svg','line');
  g.setAttribute('x1',0);g.setAttribute('x2',W);g.setAttribute('y1',Y(vv));g.setAttribute('y2',Y(vv));
  g.setAttribute('stroke','#2a3142');g.setAttribute('stroke-width',1);chart.appendChild(g);
  const tx=document.createElementNS('http://www.w3.org/2000/svg','text');
  tx.setAttribute('x',W-4);tx.setAttribute('y',Y(vv)-3);tx.setAttribute('font-size','10');
  tx.setAttribute('fill','#98a0ab');tx.setAttribute('text-anchor','end');tx.textContent=vv.toFixed(1);chart.appendChild(tx);
}}
rings.forEach(r=>{{
  const l=document.createElementNS('http://www.w3.org/2000/svg','line');
  l.setAttribute('x1',X(r.t));l.setAttribute('x2',X(r.t));l.setAttribute('y1',0);l.setAttribute('y2',H);
  l.setAttribute('stroke',PHASE_COLOR[r.n-1]);l.setAttribute('stroke-width',1);l.setAttribute('stroke-dasharray','3 3');l.setAttribute('opacity',0.4);
  chart.appendChild(l);
}});
const chartLines={{}};
teams.forEach((tm)=>{{
  const pl=document.createElementNS('http://www.w3.org/2000/svg','polyline');
  pl.setAttribute('points', tm.path.map(p=>X(p.t).toFixed(1)+','+Y(p.V).toFixed(1)).join(' '));
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
  it.innerHTML='<span class="dot" style="background:'+tm.color+'"></span>'
    +'<span class="nm">'+tm.name+'</span>'
    +'<span class="v">峰值 '+(tm.path.reduce((m,p)=>Math.max(m,p.V==null?-1:p.V),-1)).toFixed(1)+'</span>'
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
  updateField(t);
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
document.getElementById('vftoggle').onchange=(e)=>vf.setAttribute('opacity', e.target.checked?0.30:0);

update(0);
</script></body></html>"""

    open(OUT_HTML, 'w', encoding='utf-8').write(html)
    print(f"HTML 已生成 {OUT_HTML}  ({len(html)//1024} KB)")


if __name__ == '__main__':
    main()
