from __future__ import annotations

import json
from typing import Any

import streamlit.components.v1 as components


def _safe_json(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def render_evidence_network_3d(
    nodes: list[dict[str, Any]],
    links: list[dict[str, Any]],
    *,
    title: str,
    tumour_label: str,
    compound_label: str,
    stats: dict[str, str],
    height: int = 900,
) -> None:
    """Render a dependency-free 3D evidence graph using a native canvas.

    Nodes have true XYZ positions. The renderer applies perspective projection,
    yaw/pitch camera rotation, depth scaling/occlusion, and click-to-inspect. It
    has no CDN, WebGL, Three.js, or external JavaScript dependencies.

    Edges are source semantics supplied by the page builder. The renderer does
    not infer or add biological relationships.
    """
    payload = _safe_json({
        "nodes": nodes,
        "links": links,
        "title": title,
        "tumourLabel": tumour_label,
        "compoundLabel": compound_label,
        "stats": stats,
    })

    document = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evidence Atlas · 3D Evidence Network</title>
<style>
:root{color-scheme:dark;--bg:#071522;--ink:#edf5f8;--muted:#9ab0bf;--line:rgba(169,198,215,.17);--panel:rgba(10,25,39,.97);--teal:#58c6b5;--blue:#79afe1;--violet:#b49aef;--amber:#f1bf73;--gold:#e9b969;--red:#df8585;--slate:#a8bac7}
*{box-sizing:border-box}
html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#071522;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--ink)}
button{cursor:pointer;border:1px solid rgba(158,192,210,.24);border-radius:9px;padding:8px 10px;background:rgba(16,39,57,.96);color:#e2edf3;font-size:10px;font-weight:750;letter-spacing:.02em;transition:.16s ease}
button:hover{background:#1b4554;border-color:rgba(88,198,181,.65);transform:translateY(-1px)}
button.active{color:#8be0d0;border-color:rgba(88,198,181,.7);background:rgba(24,104,91,.32)}
#app{position:relative;width:100%;height:100%;min-height:820px;overflow:hidden;background:radial-gradient(ellipse at 39% 43%,#19394c 0%,#102b40 37%,#0a1e2e 73%,#071522 100%)}
#app:before{content:"";position:absolute;inset:0;pointer-events:none;background-image:linear-gradient(rgba(120,170,190,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(120,170,190,.035) 1px,transparent 1px);background-size:44px 44px;mask-image:linear-gradient(90deg,black 0%,rgba(0,0,0,.6) 74%,transparent 100%)}
#header{position:absolute;z-index:10;left:15px;right:336px;top:12px;display:flex;align-items:flex-start;justify-content:space-between;gap:12px}
.brand{border:1px solid var(--line);background:rgba(6,20,32,.76);backdrop-filter:blur(12px);border-radius:13px;padding:10px 13px;min-width:260px}
.kicker{font-size:8px;letter-spacing:.2em;text-transform:uppercase;font-weight:900;color:var(--teal)}
.title{font-size:15px;font-weight:820;letter-spacing:-.025em;margin-top:4px;line-height:1.18}
.sub{font-size:9px;color:#9bb2c0;margin-top:4px;line-height:1.4}
#toolbar{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:6px}
#stats{position:absolute;z-index:9;left:16px;right:338px;top:84px;display:flex;gap:7px;flex-wrap:wrap;pointer-events:none}
.stat{border:1px solid rgba(164,195,212,.17);background:rgba(7,22,35,.72);backdrop-filter:blur(10px);border-radius:9px;padding:7px 10px;min-width:105px}
.stat b{display:block;color:#f0f6f8;font-size:13px;font-weight:850;font-variant-numeric:tabular-nums}
.stat span{display:block;color:#91a9ba;font-size:7px;letter-spacing:.1em;text-transform:uppercase;margin-top:2px}
#scene{position:absolute;left:0;right:324px;top:132px;bottom:88px;z-index:1;touch-action:none}
/* Presentation mode removes the fixed information rail and gives the graph almost the entire screen. */
#app.presentation{position:fixed;inset:0;width:100vw;height:100vh;min-height:100vh;z-index:999999;border-radius:0}
#app.presentation #side{display:none}
#app.presentation #header{left:20px;right:20px;top:12px;align-items:center}
#app.presentation .brand{padding:8px 12px;min-width:230px}
#app.presentation .title{font-size:14px}
#app.presentation .sub{font-size:8px}
#app.presentation #stats{left:auto;right:20px;top:75px;justify-content:flex-end;gap:6px}
#app.presentation .stat{min-width:92px;padding:6px 9px}
#app.presentation .stat b{font-size:12px}
#app.presentation #laneheads{left:20px;right:20px;top:126px;max-width:58%}
#app.presentation .lane{padding:6px 8px}
#app.presentation .lane b{font-size:7px}
#app.presentation .lane span{font-size:6px}
#app.presentation #scene{left:0;right:0;top:174px;bottom:58px}
#app.presentation #bottom{left:20px;right:20px;bottom:8px;padding:6px 10px}
#app.presentation #side{display:none}
#app.presentation #hint{left:20px;bottom:66px}
#app.presentation #toolbar button{font-size:11px;padding:9px 12px}
#app.presentation #stats{top:72px}
#app.presentation #laneheads{top:118px}
#app.presentation #hint{left:20px;bottom:78px}
#app.presentation #presentation{color:#8be0d0;border-color:rgba(88,198,181,.7);background:rgba(24,104,91,.32)}
#app.presentation:fullscreen{width:100vw;height:100vh;min-height:100vh}
#app.presentation:fullscreen #scene{top:132px;bottom:74px}
@media(max-width:1100px){#app.presentation #laneheads{max-width:100%;right:20px}#app.presentation #stats{left:20px;right:20px;top:76px;justify-content:flex-start}#app.presentation #laneheads{top:132px;max-width:100%}#app.presentation #scene{top:178px;bottom:76px}}

#scene canvas{width:100%;height:100%;display:block;cursor:grab}
#scene canvas.dragging{cursor:grabbing}
#laneheads{position:absolute;z-index:4;left:14px;right:337px;top:136px;display:flex;gap:8px;pointer-events:none}
.lane{flex:1;min-width:0;padding:7px 9px 6px;border-top:1px solid rgba(162,194,211,.24);background:linear-gradient(90deg,rgba(6,20,32,.68),rgba(6,20,32,.22));border-radius:0 0 7px 7px}
.lane b{display:block;font-size:8px;letter-spacing:.12em;text-transform:uppercase;color:#dce9ef;white-space:nowrap}
.lane span{display:block;font-size:7px;color:#8fa8b8;margin-top:3px;line-height:1.3}
.lane.compound{border-color:rgba(121,175,225,.78)}.lane.target{border-color:rgba(180,154,239,.8)}.lane.tumour{border-color:rgba(241,191,115,.84)}
#bottom{position:absolute;z-index:8;left:14px;right:337px;bottom:12px;border:1px solid var(--line);border-radius:12px;background:rgba(6,20,32,.89);backdrop-filter:blur(12px);padding:8px 11px}
#legendtop{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:6px;font-size:8px;color:#dbe7ed;font-weight:850;letter-spacing:.12em;text-transform:uppercase}
#legend{display:flex;gap:7px 15px;flex-wrap:wrap}
.legenditem{display:flex;align-items:center;gap:6px;color:#a9bdcb;font-size:8px;white-space:nowrap}
.legenddot{width:8px;height:8px;border-radius:50%;display:inline-block;box-shadow:0 0 9px currentColor}
#side{position:absolute;z-index:12;top:12px;right:12px;bottom:12px;width:300px;overflow:auto;border:1px solid rgba(151,188,208,.22);border-radius:16px;background:var(--panel);backdrop-filter:blur(16px);box-shadow:0 18px 46px rgba(0,0,0,.32);padding:15px}
.side-kicker{color:var(--teal);text-transform:uppercase;letter-spacing:.17em;font-size:8px;font-weight:900}
.side-title{color:#f2f7fa;font-size:19px;line-height:1.13;letter-spacing:-.04em;font-weight:850;margin:8px 0 7px;overflow-wrap:anywhere}
.typepill{display:inline-flex;border:1px solid rgba(121,175,225,.35);color:#b6d3ed;background:rgba(75,119,158,.15);padding:4px 7px;border-radius:999px;font-size:8px;font-weight:800}
.side-copy{color:#b3c4ce;font-size:10px;line-height:1.62;margin-top:12px;white-space:pre-wrap;overflow-wrap:anywhere}
.side-source{border-top:1px solid var(--line);margin-top:13px;padding-top:11px;color:#91a9b9;font-size:9px;line-height:1.55;overflow-wrap:anywhere}
.side-source b,.relation-list b{display:block;color:#d5e2e9;text-transform:uppercase;font-size:8px;letter-spacing:.12em;margin-bottom:5px}
.relation-list{border-top:1px solid var(--line);margin-top:12px;padding-top:10px;color:#9fb4c2;font-size:9px;line-height:1.5}
.relation-item{padding:6px 7px;margin:5px 0;border-radius:7px;background:rgba(255,255,255,.035);border:1px solid rgba(171,198,214,.09)}
.boundary{margin-top:13px;border-left:2px solid var(--amber);padding:9px 10px;background:rgba(148,101,37,.11);color:#d1b98f;font-size:8px;line-height:1.5}
#hint{position:absolute;z-index:7;left:16px;bottom:82px;color:#a9bfcc;font-size:8px;letter-spacing:.035em;background:rgba(7,21,34,.75);border:1px solid rgba(154,190,211,.15);border-radius:8px;padding:6px 8px;pointer-events:none}
#empty{display:none;position:absolute;z-index:6;inset:190px 35px 130px 35px;border:1px dashed rgba(174,198,213,.3);border-radius:18px;background:rgba(7,22,35,.75);align-items:center;justify-content:center;text-align:center;color:#c2d2dc;padding:30px;font-size:13px;line-height:1.7}
@media(max-width:900px){#header{right:10px;flex-direction:column}#stats{right:10px;top:124px}#scene{right:0;top:184px;bottom:320px}#laneheads{left:8px;right:8px;top:188px}#side{top:auto;left:10px;right:10px;bottom:10px;width:auto;height:270px}#bottom{left:8px;right:8px;bottom:286px}#hint{bottom:354px}}
@media(max-width:700px){#toolbar{justify-content:flex-start}#header{left:8px}#stats{left:8px;right:8px}#app.presentation #header{left:8px;right:8px;align-items:flex-start}#app.presentation .brand{max-width:calc(100vw - 16px)}#app.presentation #stats{left:8px;right:8px;top:112px}#app.presentation #laneheads{display:none}#app.presentation #scene{top:168px;bottom:118px}#app.presentation #bottom{left:8px;right:8px}#app.presentation #hint{left:8px;bottom:120px}}
</style>
</head>
<body>
<div id="app">
  <header id="header">
    <div class="brand"><div class="kicker">Evidence Atlas · 3D relationship explorer</div><div class="title">Follow the evidence from compound to tumour context</div><div class="sub">Source annotations · gene-symbol overlap · tumour observations · separate model assays</div></div>
    <div id="toolbar"><button id="presentation">Presentation view</button><button id="view2d">2D view</button><button id="labels" class="active">Labels on</button><button id="orbit" class="active">Auto rotate on</button><button id="fit">Fit view</button><button id="reset">Reset</button></div>
  </header>
  <div id="stats">
    <div class="stat"><b id="sTarget">—</b><span>ChEMBL target rows</span></div>
    <div class="stat"><b id="sGenes">—</b><span>Target gene symbols</span></div>
    <div class="stat"><b id="sOverlap">—</b><span>Exact tumour overlaps</span></div>
    <div class="stat"><b id="sResp">—</b><span>GDSC2 assay records</span></div>
  </div>
  <div id="laneheads">
    <div class="lane compound"><b>Compound mapping</b><span>GDSC drug record → ChEMBL molecule</span></div>
    <div class="lane target"><b>Target annotations</b><span>Target entity → gene symbols</span></div>
    <div class="lane tumour"><b>Tumour observations</b><span>Exact symbol overlap in selected label</span></div>
  </div>
  <main id="scene"><canvas id="graph" aria-label="Interactive 3D evidence relationship network"></canvas></main>
  <div id="empty">No graph entities were returned for this selection. Select another compound or tumour label, or inspect the source index status.</div>
  <aside id="side">
    <div class="side-kicker">Selected entity / relationship context</div>
    <div id="sideTitle" class="side-title">Selected compound</div>
    <span id="sideType" class="typepill">GDSC compound record</span>
    <div id="sideCopy" class="side-copy">Select a labelled node to inspect its meaning and provenance.</div>
    <div id="sideSource" class="side-source"><b>Source lineage</b>Local evidence index</div>
    <div class="relation-list"><b>Direct connections</b><div id="sideRelations">Select a node to inspect its connected evidence.</div></div>
    <div class="boundary">Edges are labelled by source meaning. A target-to-gene annotation and an exact gene-symbol overlap are not proof of tumour dependency, causality, efficacy or clinical benefit. GDSC2 model observations remain in a separate branch.</div>
  </aside>
  <div id="hint">Drag to orbit · Scroll to zoom · Click a node to inspect · Double-click to focus</div>
  <footer id="bottom">
    <div id="legendtop"><span>Evidence entity types</span><span>Node size indicates display role / record count, not biological importance</span></div>
    <div id="legend">
      <span class="legenditem"><i class="legenddot" style="color:#79afe1;background:#79afe1"></i>Compound / source mapping</span>
      <span class="legenditem"><i class="legenddot" style="color:#b49aef;background:#b49aef"></i>Target entity</span>
      <span class="legenditem"><i class="legenddot" style="color:#58c6b5;background:#58c6b5"></i>Gene annotation</span>
      <span class="legenditem"><i class="legenddot" style="color:#f1bf73;background:#f1bf73"></i>Tumour context</span>
      <span class="legenditem"><i class="legenddot" style="color:#9bb2c0;background:#9bb2c0"></i>GDSC2 assay / model</span>
      <span class="legenditem"><i class="legenddot" style="color:#e2a1a1;background:#e2a1a1"></i>Missing source mapping</span>
    </div>
  </footer>
</div>
<script>
'use strict';
const DATA = __GRAPH_DATA__;
const canvas = document.getElementById('graph');
const ctx = canvas.getContext('2d');
const scene = document.getElementById('scene');
const fmt = v => Number(v || 0).toLocaleString('en-US');
const statText = v => String(v ?? '—');
const esc = s => String(s ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const nodeById = new Map(DATA.nodes.map(n=>[n.id,n]));
const links = DATA.links.slice();
const nodes = DATA.nodes.slice();
const connected = new Map();
for (const n of nodes) connected.set(n.id, []);
for (const l of links) {
  if (connected.has(l.source)) connected.get(l.source).push({id:l.target,label:l.label,kind:l.kind,detail:l.detail});
  if (connected.has(l.target)) connected.get(l.target).push({id:l.source,label:l.label,kind:l.kind,detail:l.detail});
}

document.getElementById('sTarget').textContent=statText(DATA.stats.targetRows);
document.getElementById('sGenes').textContent=statText(DATA.stats.targetGenes);
document.getElementById('sOverlap').textContent=statText(DATA.stats.overlapGenes);
document.getElementById('sResp').textContent=statText(DATA.stats.responseRows);

let yaw=-0.28, pitch=0.30, zoom=1, autoOrbit=true, labelsOn=true;
let viewMode='3d', panX=0, panY=0, saved3DOrbit=true;
let selectedId=DATA.selectedNode || (nodes[0] && nodes[0].id);
let hoverId=null, drag=null, moved=false, lastPointer=null;
let projected=new Map(), bounds={minX:-700,maxX:800,minY:-390,maxY:390};
let canvasW=0,canvasH=0,dpr=1,raf=0;

function resize(){
  const rect=scene.getBoundingClientRect();
  dpr=Math.min(window.devicePixelRatio||1,2);
  canvasW=Math.max(10,rect.width); canvasH=Math.max(10,rect.height);
  canvas.width=Math.round(canvasW*dpr);canvas.height=Math.round(canvasH*dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0);
  fitBounds(); draw();
}
function fitBounds(){
  if(!nodes.length)return;
  const xs=nodes.map(n=>Number(n.x)||0),ys=nodes.map(n=>Number(n.y)||0);
  bounds={minX:Math.min(...xs)-90,maxX:Math.max(...xs)+90,minY:Math.min(...ys)-100,maxY:Math.max(...ys)+100};
  // Keep zoom independent from fit scale. The previous implementation multiplied
  // the viewport fit factor twice, making the network look unnecessarily small.
  zoom=1.0;
  panX=0;panY=0;
}
function project(n){
  const x=Number(n.x)||0,y=Number(n.y)||0,z=Number(n.z)||0;
  const scale=zoom*Math.min((canvasW-110)/(bounds.maxX-bounds.minX||1),(canvasH-130)/(bounds.maxY-bounds.minY||1));
  const midX=(bounds.minX+bounds.maxX)/2,midY=(bounds.minY+bounds.maxY)/2;
  if(viewMode==='2d'){
    // The layered 2D view uses the same source-backed nodes and edges, flattened
    // onto the XY plane. No records or relationships are added or removed.
    return {x:canvasW*.49+(x-midX)*scale+panX,y:canvasH*.53-(y-midY)*scale+panY,z:0,scale,persp:1};
  }
  const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const xr=x*cy-z*sy, zr=x*sy+z*cy;
  const yr=y*cp-zr*sp, zf=y*sp+zr*cp;
  const depth=1450;
  const persp=depth/(depth+zf);
  return {x:canvasW*.49+(xr-midX)*scale*persp,y:canvasH*.53-(yr-midY)*scale*persp,z:zf,scale:scale*persp,persp};
}
function rgba(hex,a){
  const h=String(hex||'#8fa8b8').replace('#','');
  const full=h.length===3?h.split('').map(c=>c+c).join(''):h;
  const n=parseInt(full,16)||0;
  return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`;
}
function roundedRect(x,y,w,h,r){
  ctx.beginPath();ctx.moveTo(x+r,y);ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath();
}
function drawBackground(){
  const g=ctx.createRadialGradient(canvasW*.42,canvasH*.42,10,canvasW*.46,canvasH*.45,Math.max(canvasW,canvasH)*.76);
  g.addColorStop(0,'rgba(31,75,91,.38)');g.addColorStop(.58,'rgba(16,42,61,.25)');g.addColorStop(1,'rgba(5,16,27,.1)');ctx.fillStyle=g;ctx.fillRect(0,0,canvasW,canvasH);
  // Fine technical grid, softened towards the right edge.
  ctx.save();ctx.strokeStyle='rgba(132,179,194,.055)';ctx.lineWidth=1;
  const step=42;for(let x=0;x<canvasW;x+=step){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvasH);ctx.stroke();}
  for(let y=0;y<canvasH;y+=step){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(canvasW,y);ctx.stroke();}
  ctx.restore();
  // Spatial rings are only shown in 3D; the flat view keeps the previous
  // clean layered-network look with the same records and relationship labels.
  if(viewMode==='3d'){
    const center={x:canvasW*.48,y:canvasH*.53};
    ctx.save();ctx.translate(center.x,center.y);ctx.strokeStyle='rgba(119,170,185,.11)';ctx.lineWidth=1;
    for(let r=90;r<Math.min(canvasW,canvasH)*.48;r+=75){ctx.beginPath();ctx.ellipse(0,0,r,r*.35,-.12,0,Math.PI*2);ctx.stroke();}
    ctx.restore();
  }
}
function projectAll(){projected.clear();for(const n of nodes)projected.set(n.id,project(n));}
function drawLink(link,t){
  const a=projected.get(link.source),b=projected.get(link.target);if(!a||!b)return;
  const active=selectedId===link.source||selectedId===link.target||((connected.get(selectedId)||[]).some(r=>r.id===link.source||r.id===link.target));
  const base=link.color||'#91a8b8';
  const alpha=selectedId?(active?.86:.15):.62;
  const dx=b.x-a.x,dy=b.y-a.y,dist=Math.hypot(dx,dy)||1;
  const bend=Math.min(32,dist*.08)*(link.curveSide||1);
  const cx=(a.x+b.x)/2-dy/dist*bend,cy=(a.y+b.y)/2+dx/dist*bend;
  ctx.save();ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.quadraticCurveTo(cx,cy,b.x,b.y);
  ctx.strokeStyle=rgba(base,alpha);ctx.lineWidth=(link.width||1.8)*(active?1.25:1);ctx.setLineDash(link.dashed?[5,5]:[]);ctx.shadowColor=rgba(base,active?.34:.12);ctx.shadowBlur=active?9:3;ctx.stroke();ctx.setLineDash([]);ctx.shadowBlur=0;
  // Direction arrow at the destination to clarify relationship reading order.
  const ax=b.x-cx,ay=b.y-cy,al=Math.hypot(ax,ay)||1,ux=ax/al,uy=ay/al;
  const tip={x:b.x-ux*9,y:b.y-uy*9};const size=5;
  ctx.beginPath();ctx.moveTo(tip.x,tip.y);ctx.lineTo(tip.x-ux*size-uy*size*.7,tip.y-uy*size+ux*size*.7);ctx.lineTo(tip.x-ux*size+uy*size*.7,tip.y-uy*size-ux*size*.7);ctx.closePath();ctx.fillStyle=rgba(base,alpha);ctx.fill();
  // Subtle travelling particle reinforces direction while keeping the graph calm.
  const phase=((t/2600)+(link.particleOffset||0))%1;const q=1-phase;
  const px=(1-q)*(1-q)*a.x+2*(1-q)*q*cx+q*q*b.x;
  const py=(1-q)*(1-q)*a.y+2*(1-q)*q*cy+q*q*b.y;
  ctx.beginPath();ctx.arc(px,py,active?2.5:1.7,0,Math.PI*2);ctx.fillStyle=rgba(base,alpha*.8);ctx.fill();
  if(selectedId===link.source||selectedId===link.target){
    const rawLabel=String(link.label||link.kind||'Evidence relationship');
    const edgeLabel=rawLabel.length>34?rawLabel.slice(0,32)+'…':rawLabel;
    const lx=(a.x+2*cx+b.x)/4, ly=(a.y+2*cy+b.y)/4;
    ctx.font='700 8px Inter,system-ui,sans-serif';
    const tw=Math.min(210,ctx.measureText(edgeLabel).width+13);
    roundedRect(lx-tw/2,ly-9,tw,18,5);
    ctx.fillStyle='rgba(5,18,29,.93)';ctx.fill();
    ctx.strokeStyle=rgba(base,.62);ctx.lineWidth=.8;ctx.stroke();
    ctx.fillStyle='#e4eef3';ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.fillText(edgeLabel,lx,ly,tw-10);
  }
  ctx.restore();
}
function drawNode(n){
  const p=projected.get(n.id);if(!p)return;
  const isSelected=n.id===selectedId,isHover=n.id===hoverId;
  const neighbour=(connected.get(selectedId)||[]).some(r=>r.id===n.id);
  const fade=selectedId&&!isSelected&&!neighbour?.25:1;
  const r=(Number(n.radius)||12)*Math.max(.68,Math.min(1.22,p.persp))* (isSelected?1.25:1);
  ctx.save();ctx.globalAlpha=fade;
  if(viewMode==='2d'){
    // Flat node marks restore the earlier layered 2D-network visual language.
    ctx.fillStyle=rgba(n.color,isSelected?.98:.88);ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle=isSelected?'#fff1d4':(isHover?'#ffffff':rgba(n.color,.96));ctx.lineWidth=isSelected?2.4:1.35;ctx.stroke();
    if(isSelected){ctx.beginPath();ctx.arc(p.x,p.y,r+5,0,Math.PI*2);ctx.strokeStyle=rgba(n.color,.42);ctx.lineWidth=1;ctx.stroke();}
  }else{
    // Halo and polished sphere shading for the perspective 3D view.
    const halo=ctx.createRadialGradient(p.x,p.y,r*.35,p.x,p.y,r*2.5);
    halo.addColorStop(0,rgba(n.color,.36));halo.addColorStop(1,rgba(n.color,0));ctx.fillStyle=halo;ctx.beginPath();ctx.arc(p.x,p.y,r*2.5,0,Math.PI*2);ctx.fill();
    const sphere=ctx.createRadialGradient(p.x-r*.37,p.y-r*.42,1,p.x,p.y,r*1.25);
    sphere.addColorStop(0,'#f7fbff');sphere.addColorStop(.16,rgba(n.color,1));sphere.addColorStop(.76,rgba(n.color,.92));sphere.addColorStop(1,'#091827');
    ctx.fillStyle=sphere;ctx.shadowColor=rgba(n.color,isSelected?.85:.38);ctx.shadowBlur=isSelected?25:12;ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;
    ctx.strokeStyle=isSelected?'#fff1d4':(isHover?'#ffffff':rgba(n.color,.85));ctx.lineWidth=isSelected?2.4:1.2;ctx.stroke();
    // A fine specular highlight makes larger hub nodes read as 3D objects.
    ctx.fillStyle='rgba(255,255,255,.48)';ctx.beginPath();ctx.arc(p.x-r*.31,p.y-r*.34,Math.max(1.2,r*.18),0,Math.PI*2);ctx.fill();
  }
  if(labelsOn){
    const label=String(n.displayLabel||n.label||'');const parts=label.split('\n');
    const fontSize=n.labelSize||10;ctx.font=`${n.bold?'800':'700'} ${fontSize}px Inter,system-ui,sans-serif`;
    const lines=parts.slice(0,2);const maxW=Math.max(...lines.map(s=>ctx.measureText(s).width),24);
    const boxW=Math.min(180,maxW+14),boxH=lines.length===2?31:21;
    const side=n.labelSide||'right';
    let lx=p.x+Math.max(10,r*.65),ly=p.y-boxH/2;
    if(side==='left'){lx=p.x-boxW-Math.max(10,r*.65);}
    else if(side==='top'){lx=p.x-boxW/2;ly=p.y-r-boxH-9;}
    else if(side==='bottom'){lx=p.x-boxW/2;ly=p.y+r+9;}
    if(lx+boxW>canvasW-8)lx=canvasW-boxW-8;
    if(lx<8)lx=8;
    if(ly+boxH>canvasH-5)ly=canvasH-boxH-5;
    if(ly<5)ly=5;
    roundedRect(lx,ly,boxW,boxH,6);ctx.fillStyle=isSelected?'rgba(61,49,31,.94)':'rgba(5,19,31,.88)';ctx.fill();
    ctx.strokeStyle=rgba(n.color,isSelected?.72:.4);ctx.lineWidth=1;ctx.stroke();
    ctx.textBaseline='middle';ctx.fillStyle=isSelected?'#fff0d6':'#e6f0f4';ctx.font=`800 ${fontSize}px Inter,system-ui,sans-serif`;ctx.fillText(lines[0],lx+7,ly+(lines.length===2?10:boxH/2),boxW-12);
    if(lines[1]){ctx.fillStyle='#9eb4c2';ctx.font='650 7px Inter,system-ui,sans-serif';ctx.fillText(lines[1],lx+7,ly+23,boxW-12);}
  }
  ctx.restore();
}
function draw(t=performance.now()){
  if(!ctx)return;
  ctx.clearRect(0,0,canvasW,canvasH);drawBackground();projectAll();
  const sortedLinks=links.slice().sort((a,b)=>((projected.get(a.source)?.z||0)+(projected.get(a.target)?.z||0))-((projected.get(b.source)?.z||0)+(projected.get(b.target)?.z||0)));
  for(const l of sortedLinks)drawLink(l,t);
  const sortedNodes=nodes.slice().sort((a,b)=>(projected.get(a.id)?.z||0)-(projected.get(b.id)?.z||0));
  for(const n of sortedNodes)drawNode(n);
  if(viewMode==='3d' && autoOrbit && !drag){yaw+=.00048;}
  raf=requestAnimationFrame(draw);
}
function selectNode(node){
  selectedId=node.id;const con=connected.get(node.id)||[];
  document.getElementById('sideTitle').textContent=node.label||node.id;
  document.getElementById('sideType').textContent=node.kind||'Evidence entity';
  document.getElementById('sideCopy').textContent=node.detail||'No additional details are available for this entity.';
  document.getElementById('sideSource').innerHTML='<b>Source lineage</b>'+esc(node.source||'Not specified');
  const rel=document.getElementById('sideRelations');
  rel.innerHTML=con.length?con.slice(0,12).map(r=>`<div class="relation-item"><strong>${esc(r.label||r.kind)}</strong><br>${esc(nodeById.get(r.id)?.label||r.id)}</div>`).join(''):'No direct relationships are present in this selection.';
}
function hitTest(x,y){let best=null,bestD=Infinity;for(const n of nodes){const p=projected.get(n.id);if(!p)continue;const d=Math.hypot(x-p.x,y-p.y);const r=(Number(n.radius)||12)*Math.max(.68,Math.min(1.22,p.persp));if(d<Math.max(20,r+10)&&d<bestD){best=n;bestD=d;}}return best;}
function fit(){yaw=-.18;pitch=.12;fitBounds();}
function setViewMode(next){
  if(next===viewMode)return;
  if(next==='2d'){
    saved3DOrbit=autoOrbit;autoOrbit=false;yaw=0;pitch=0;panX=0;panY=0;zoom=1;viewMode='2d';
    const btn=document.getElementById('view2d');btn.textContent='3D view';btn.classList.add('active');
    const orbitBtn=document.getElementById('orbit');orbitBtn.disabled=true;orbitBtn.textContent='Auto rotate off';orbitBtn.classList.remove('active');
  }else{
    viewMode='3d';yaw=-.28;pitch=.30;panX=0;panY=0;zoom=1;autoOrbit=saved3DOrbit;
    const btn=document.getElementById('view2d');btn.textContent='2D view';btn.classList.remove('active');
    const orbitBtn=document.getElementById('orbit');orbitBtn.disabled=false;orbitBtn.textContent=`Auto rotate ${autoOrbit?'on':'off'}`;orbitBtn.classList.toggle('active',autoOrbit);
  }
  requestAnimationFrame(resize);
}
function reset(){fit();zoom=Math.min(1.25,zoom||1);panX=0;panY=0;selectNode(nodeById.get(DATA.selectedNode)||nodes[0]);}
canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,yaw,pitch,panX,panY};moved=false;canvas.setPointerCapture(e.pointerId);canvas.classList.add('dragging');});
canvas.addEventListener('pointermove',e=>{
  const rect=canvas.getBoundingClientRect();const x=e.clientX-rect.left,y=e.clientY-rect.top;
  if(drag){const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>3)moved=true;
    if(viewMode==='2d'){panX=drag.panX+dx;panY=drag.panY+dy;}
    else{yaw=drag.yaw+dx*.006;pitch=Math.max(-1.15,Math.min(1.15,drag.pitch+dy*.005));}
  }
  else{const node=hitTest(x,y);hoverId=node?node.id:null;canvas.style.cursor=node?'pointer':'grab';}
});
canvas.addEventListener('pointerup',e=>{
  if(!drag)return;const rect=canvas.getBoundingClientRect();const x=e.clientX-rect.left,y=e.clientY-rect.top;
  if(!moved){const n=hitTest(x,y);if(n)selectNode(n);}
  drag=null;canvas.classList.remove('dragging');
});
canvas.addEventListener('pointercancel',()=>{drag=null;canvas.classList.remove('dragging');});
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.48,Math.min(2.8,zoom*(e.deltaY>0?.92:1.09)));},{passive:false});
canvas.addEventListener('dblclick',e=>{const rect=canvas.getBoundingClientRect();const n=hitTest(e.clientX-rect.left,e.clientY-rect.top);if(n){selectNode(n);}});
document.getElementById('labels').addEventListener('click',e=>{labelsOn=!labelsOn;e.currentTarget.textContent=`Labels ${labelsOn?'on':'off'}`;e.currentTarget.classList.toggle('active',labelsOn);});
document.getElementById('orbit').addEventListener('click',e=>{if(viewMode!=='3d')return;autoOrbit=!autoOrbit;e.currentTarget.textContent=`Auto rotate ${autoOrbit?'on':'off'}`;e.currentTarget.classList.toggle('active',autoOrbit);});
document.getElementById('view2d').addEventListener('click',()=>setViewMode(viewMode==='3d'?'2d':'3d'));
document.getElementById('fit').addEventListener('click',fit);
document.getElementById('reset').addEventListener('click',reset);
const appRoot=document.getElementById('app');
const presentationButton=document.getElementById('presentation');
let originalHostFrameStyle=null;
function expandHostFrame(){
  // components.html is rendered inside a same-origin Streamlit iframe. Expand
  // that iframe to the browser viewport as a fallback if fullscreen is blocked.
  try{
    const frame=window.frameElement;
    if(!frame)return false;
    if(originalHostFrameStyle===null)originalHostFrameStyle=frame.getAttribute('style');
    frame.style.setProperty('position','fixed','important');
    frame.style.setProperty('inset','0','important');
    frame.style.setProperty('width','100vw','important');
    frame.style.setProperty('height','100vh','important');
    frame.style.setProperty('max-width','none','important');
    frame.style.setProperty('z-index','999999','important');
    frame.style.setProperty('border','0','important');
    frame.setAttribute('allowfullscreen','');
    const allowed=(frame.getAttribute('allow')||'').split(';').map(v=>v.trim()).filter(Boolean);
    if(!allowed.some(v=>v.startsWith('fullscreen')))allowed.push('fullscreen');
    frame.setAttribute('allow',allowed.join('; '));
    return true;
  }catch(_error){return false;}
}
function restoreHostFrame(){
  try{
    const frame=window.frameElement;
    if(!frame)return;
    if(originalHostFrameStyle===null)frame.removeAttribute('style');
    else frame.setAttribute('style',originalHostFrameStyle);
  }catch(_error){}
  originalHostFrameStyle=null;
}
async function enterPresentation(){
  expandHostFrame();
  appRoot.classList.add('presentation');
  presentationButton.textContent='Exit presentation';
  presentationButton.classList.add('active');
  // Fullscreen is requested directly from the user's click. If denied, the
  // containing Streamlit iframe has already been expanded to the viewport.
  try{if(appRoot.requestFullscreen && !document.fullscreenElement){await appRoot.requestFullscreen();}}catch(_error){}
  requestAnimationFrame(resize);
}
async function exitPresentation(){
  appRoot.classList.remove('presentation');
  presentationButton.textContent='Presentation view';
  presentationButton.classList.remove('active');
  try{if(document.fullscreenElement){await document.exitFullscreen();}}catch(_error){}
  restoreHostFrame();
  requestAnimationFrame(resize);
}
presentationButton.addEventListener('click',()=>{
  if(appRoot.classList.contains('presentation')) exitPresentation();
  else enterPresentation();
});
document.addEventListener('fullscreenchange',()=>{
  if(!document.fullscreenElement && appRoot.classList.contains('presentation')){
    appRoot.classList.remove('presentation');
    presentationButton.textContent='Presentation view';
    presentationButton.classList.remove('active');
    restoreHostFrame();
  }
  requestAnimationFrame(resize);
});
window.addEventListener('resize',resize);
if(window.ResizeObserver){const observer=new ResizeObserver(()=>resize());observer.observe(scene);}
if(!nodes.length){document.getElementById('empty').style.display='flex';scene.style.display='none';}
// Start in ambient auto-rotation mode so the graph is presentation-ready on load.
document.getElementById('orbit').textContent='Auto rotate on';
document.getElementById('orbit').classList.add('active');
document.getElementById('view2d').textContent='2D view';
resize();selectNode(nodeById.get(selectedId)||nodes[0]);
</script>
</body>
</html>'''

    document = document.replace("__GRAPH_DATA__", payload)
    components.html(document, height=max(int(height), 1250), scrolling=False)
