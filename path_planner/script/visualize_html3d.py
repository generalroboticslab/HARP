"""Shared terrain and HTML helpers for the maintained visualizers."""

from __future__ import annotations

import json
from html import escape
from typing import Any

import numpy as np

from .terrain import TerrainMap, combined_ground_landing_cost


def _surface_layer(terrain: TerrainMap, surface_layer: str) -> np.ndarray:
    if surface_layer == "elevation":
        return terrain.elevation
    if surface_layer == "roughness":
        return terrain.roughness
    if surface_layer == "slope":
        return terrain.slope_deg
    if surface_layer == "landing_cost":
        return terrain.landing_cost
    if surface_layer == "combined_cost":
        return combined_ground_landing_cost(terrain)
    raise ValueError(f"unknown surface layer: {surface_layer}")


def _surface_display_range(layer: np.ndarray, surface_layer: str) -> tuple[float, float]:
    if surface_layer in {"combined_cost", "landing_cost"}:
        lo, hi = np.percentile(layer, [2.0, 90.0])
        if hi > lo:
            return float(lo), float(hi)
    return float(np.min(layer)), float(np.max(layer))


def _path_point(terrain: TerrainMap, point: dict[str, float], z_scale: float, lift: float) -> list[float]:
    if "z" in point and point["z"] > 0.0:
        z = (
            terrain.value("elevation", point["x"], point["y"]) + point["z"]
        ) * z_scale
    else:
        z = terrain.value("elevation", point["x"], point["y"]) * z_scale + lift
    return [float(point["x"]), float(point["y"]), float(z)]



def _state_point(terrain: TerrainMap, state: dict[str, Any], z_scale: float, lift: float) -> list[float]:
    if float(state.get("z", 0.0)) > 0.0:
        z = terrain.value("elevation", state["x"], state["y"]) * z_scale + float(state["z"]) * z_scale
    else:
        z = terrain.value("elevation", state["x"], state["y"]) * z_scale + lift
    return [float(state["x"]), float(state["y"]), float(z)]



def _html(scene: dict[str, Any]) -> str:
    scene_json = json.dumps(scene, separators=(",", ":"))
    timeline_items = "".join(
        f'<div class="timeline-item" data-start="{float(item.get("startTime", 0.0)):.6f}">'
        f'<span>{index}</span><div>{escape(str(item.get("label", "")))}'
        f'<small>{float(item.get("startTime", 0.0)):.1f}–{float(item.get("endTime", 0.0)):.1f} s</small></div></div>'
        for index, item in enumerate(scene.get("timeline", []), start=1)
    )
    timeline_html = (
        f'<div class="timeline"><div class="timeline-title">Mission timeline</div>{timeline_items}</div>'
        if timeline_items
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{scene["title"]}</title>
<style>
  html, body {{ margin: 0; height: 100%; background: #0f172a; color: #e2e8f0; font-family: Inter, system-ui, sans-serif; }}
  #canvas {{ width: 100vw; height: 100vh; display: block; cursor: grab; }}
  #canvas:active {{ cursor: grabbing; }}
  .panel {{ position: fixed; left: 16px; top: 16px; background: rgba(15, 23, 42, 0.86); border: 1px solid rgba(148, 163, 184, 0.35); border-radius: 8px; padding: 12px 14px; max-width: 360px; box-shadow: 0 18px 45px rgba(0,0,0,.28); }}
  h1 {{ margin: 0 0 8px; font-size: 15px; font-weight: 700; }}
  .hint {{ margin-top: 8px; color: #94a3b8; font-size: 12px; line-height: 1.35; }}
  .legend {{ display: grid; gap: 5px; font-size: 12px; }}
  .item {{ display: flex; align-items: center; gap: 8px; }}
  .swatch {{ width: 22px; height: 3px; border-radius: 999px; background: var(--c); }}
  .swatch.flight {{ height: 0; border-top: 3px dashed var(--c); background: none; }}
  .star {{ width: 9px; height: 9px; border-radius: 50%; background: var(--c); box-shadow: 0 0 0 2px #020617; }}
  .scale {{ margin-top: 10px; font-size: 11px; color: #cbd5e1; }}
  .scale-title {{ display: flex; justify-content: space-between; gap: 10px; margin-bottom: 5px; }}
  .scale-bar {{ height: 9px; border-radius: 999px; border: 1px solid rgba(226,232,240,.28); background: linear-gradient(90deg, rgb(68,1,84), rgb(59,82,139), rgb(33,145,140), rgb(94,201,98), rgb(253,231,37)); }}
  .scale-labels {{ display: flex; justify-content: space-between; margin-top: 4px; color: #94a3b8; }}
  .timeline {{ margin-top: 11px; padding-top: 9px; border-top: 1px solid rgba(148,163,184,.28); display: grid; gap: 5px; }}
  .timeline-title {{ color: #cbd5e1; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }}
  .timeline-item {{ display: grid; grid-template-columns: 18px 1fr; gap: 7px; align-items: start; color: #cbd5e1; font-size: 11px; line-height: 1.3; }}
  .timeline-item {{ cursor: pointer; border-radius: 4px; padding: 2px; margin: -2px; }}
  .timeline-item:hover, .timeline-item.active {{ background: rgba(56, 189, 248, 0.13); color: #f8fafc; }}
  .timeline-item span {{ display: grid; place-items: center; width: 17px; height: 17px; border-radius: 50%; background: #1e3a5f; color: #7dd3fc; font-size: 10px; font-weight: 700; }}
  .timeline-item small {{ display: block; color: #64748b; font-size: 9px; margin-top: 1px; }}
  .playback {{ position: fixed; left: 430px; right: 28px; bottom: 18px; background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(148, 163, 184, 0.35); border-radius: 8px; padding: 10px 12px; display: grid; grid-template-columns: auto 1fr auto auto; gap: 10px; align-items: center; box-shadow: 0 12px 30px rgba(0,0,0,.25); }}
  .playback button, .playback select {{ color: #e2e8f0; background: #1e293b; border: 1px solid #475569; border-radius: 5px; height: 30px; }}
  .playback button {{ min-width: 64px; cursor: pointer; }}
  .playback input {{ width: 100%; accent-color: #38bdf8; }}
  .playback-readout {{ min-width: 102px; text-align: right; font-variant-numeric: tabular-nums; font-size: 12px; }}
  .phase-readout {{ grid-column: 2 / 5; color: #94a3b8; font-size: 11px; min-height: 14px; }}
  @media (max-width: 899px) {{ .playback {{ left: 16px; right: 16px; }} .panel {{ max-height: 58vh; overflow-y: auto; }} }}
</style>
</head>
<body>
<canvas id="canvas"></canvas>
<div class="panel">
  <h1>{scene["title"]}</h1>
  <div class="legend">
    <div class="item"><span class="swatch flight" style="--c:#60a5fa"></span>flight path (dashed)</div>
    <div class="item"><span class="swatch" style="--c:#22c55e"></span>ground drive (solid)</div>
    <div class="item"><span class="swatch" style="--c:#f8fafc"></span>docked platform transport</div>
    <div class="item"><span class="star" style="--c:#f97316"></span>drill drone</div>
    <div class="item"><span class="star" style="--c:#facc15"></span>drill docked at target</div>
  </div>
  <div class="scale">
    <div class="scale-title"><span>ground + landing cost</span><span>p2-p90 color scale</span></div>
    <div class="scale-bar"></div>
    <div class="scale-labels"><span>{scene["layerDisplayMin"]:.1f}</span><span>{scene["layerDisplayMax"]:.1f}</span></div>
  </div>
  {timeline_html}
  <div class="hint">Drag to rotate. Wheel to zoom. Double click to reset view.</div>
</div>
<div id="playback" class="playback">
  <button id="playButton" type="button">Play</button>
  <input id="timeSlider" type="range" min="0" max="{float(scene.get('totalTime', 0.0)):.6f}" value="0" step="0.05" aria-label="mission time">
  <select id="speedSelect" aria-label="playback speed">
    <option value="1">1×</option><option value="2">2×</option><option value="5">5×</option><option value="10">10×</option>
  </select>
  <div id="timeDisplay" class="playback-readout">0.0 / {float(scene.get('totalTime', 0.0)):.1f} s</div>
  <div id="phaseDisplay" class="phase-readout"></div>
</div>
<script>
const scene = {scene_json};
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const playback = document.getElementById('playback');
const playButton = document.getElementById('playButton');
const timeSlider = document.getElementById('timeSlider');
const speedSelect = document.getElementById('speedSelect');
const timeDisplay = document.getElementById('timeDisplay');
const phaseDisplay = document.getElementById('phaseDisplay');
scene.totalTime = Number(scene.totalTime ?? 0);
scene.timeline = (scene.timeline ?? []).map(item => ({{
  ...item,
  startTime: Number(item.startTime ?? 0),
  endTime: Number(item.endTime ?? 0),
}}));
if (scene.totalTime <= 0) playback.style.display = 'none';
let yaw = -0.82;
let pitch = 0.5235987755982988;
let zoom = 1.0;
let currentTime = 0.0;
let playing = false;
let lastFrameTime = null;
let dragging = false;
let last = [0, 0];
let viewScale = 1.0;
let viewOffset = [0, 0];

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(innerWidth * dpr);
  canvas.height = Math.floor(innerHeight * dpr);
  canvas.style.width = innerWidth + 'px';
  canvas.style.height = innerHeight + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}
window.addEventListener('resize', resize);
canvas.addEventListener('mousedown', e => {{ dragging = true; last = [e.clientX, e.clientY]; }});
window.addEventListener('mouseup', () => dragging = false);
window.addEventListener('mousemove', e => {{
  if (!dragging) return;
  yaw += (e.clientX - last[0]) * 0.008;
  pitch = Math.max(0.25, Math.min(1.35, pitch + (e.clientY - last[1]) * 0.006));
  last = [e.clientX, e.clientY];
  draw();
}});
canvas.addEventListener('wheel', e => {{
  e.preventDefault();
  zoom *= Math.exp(-e.deltaY * 0.001);
  zoom = Math.max(0.45, Math.min(3.0, zoom));
  draw();
}}, {{ passive: false }});
canvas.addEventListener('dblclick', () => {{ yaw = -0.82; pitch = 0.5235987755982988; zoom = 1.0; draw(); }});
playButton.addEventListener('click', () => {{
  if (currentTime >= scene.totalTime - 1e-6) setMissionTime(0.0);
  playing = !playing;
  playButton.textContent = playing ? 'Pause' : 'Play';
  lastFrameTime = null;
  if (playing) requestAnimationFrame(animate);
}});
timeSlider.addEventListener('input', () => {{
  playing = false;
  playButton.textContent = 'Play';
  setMissionTime(Number(timeSlider.value));
}});
document.querySelectorAll('.timeline-item').forEach(item => {{
  item.addEventListener('click', () => {{
    playing = false;
    playButton.textContent = 'Play';
    setMissionTime(Number(item.dataset.start));
  }});
}});

function setMissionTime(value) {{
  currentTime = Math.max(0, Math.min(scene.totalTime, value));
  timeSlider.value = currentTime;
  timeDisplay.textContent = `${{currentTime.toFixed(1)}} / ${{scene.totalTime.toFixed(1)}} s`;
  updatePhaseDisplay();
  draw();
}}

function animate(timestamp) {{
  if (!playing) return;
  if (lastFrameTime === null) lastFrameTime = timestamp;
  const delta = (timestamp - lastFrameTime) / 1000 * Number(speedSelect.value);
  lastFrameTime = timestamp;
  setMissionTime(currentTime + delta);
  if (currentTime >= scene.totalTime - 1e-6) {{
    playing = false;
    playButton.textContent = 'Play';
    return;
  }}
  requestAnimationFrame(animate);
}}

function updatePhaseDisplay() {{
  if (scene.totalTime <= 0) {{
    phaseDisplay.textContent = 'Static view (no timing data)';
    return;
  }}
  const active = scene.timeline.find(item =>
    currentTime >= item.startTime &&
    (currentTime < item.endTime || (currentTime >= scene.totalTime && item.endTime >= scene.totalTime))
  );
  const drillActive = scene.paths.some(path => path.phase === 'drill_flight' && currentTime >= path.startTime && currentTime < path.endTime);
  phaseDisplay.textContent = active ? `Current: ${{active.label}}${{drillActive ? ' + drill flight' : ''}}` : 'Current: mission complete';
  document.querySelectorAll('.timeline-item').forEach((item, index) =>
    item.classList.toggle('active', Boolean(active) && scene.timeline[index] === active)
  );
}}

function project(p) {{
  const q = viewCoordinates(p);
  return [viewOffset[0] + q[0] * viewScale, viewOffset[1] + q[1] * viewScale, q[2]];
}}

function viewCoordinates(p) {{
  const cx = (scene.width - 1) * scene.resolution * 0.5;
  const cy = (scene.height - 1) * scene.resolution * 0.5;
  const x0 = p[0] - cx;
  const y0 = p[1] - cy;
  const z0 = p[2];
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const x1 = c * x0 - s * y0;
  const y1 = s * x0 + c * y0;
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const sy = y1 * cp - z0 * sp;
  const depth = y1 * sp + z0 * cp;
  return [x1, sy, depth];
}}

function updateView() {{
  const allPoints = [];
  const xs = [0, (scene.width - 1) * scene.resolution];
  const ys = [0, (scene.height - 1) * scene.resolution];
  const terrainHeights = scene.elevation.flat();
  const zs = [Math.min(...terrainHeights), Math.max(...terrainHeights)];
  xs.forEach(x => ys.forEach(y => zs.forEach(z => allPoints.push([x, y, z]))));
  scene.paths.forEach(path => path.points.forEach(point => allPoints.push(point)));
  scene.markers.forEach(marker => allPoints.push(marker.point));

  const projected = allPoints.map(viewCoordinates);
  const minX = Math.min(...projected.map(p => p[0]));
  const maxX = Math.max(...projected.map(p => p[0]));
  const minY = Math.min(...projected.map(p => p[1]));
  const maxY = Math.max(...projected.map(p => p[1]));
  const left = innerWidth >= 900 ? 430 : 24;
  const right = 28;
  const top = 28;
  const bottom = 28;
  const availableWidth = Math.max(100, innerWidth - left - right);
  const availableHeight = Math.max(100, innerHeight - top - bottom);
  const fitScale = Math.min(
    availableWidth / Math.max(maxX - minX, 1e-6),
    availableHeight / Math.max(maxY - minY, 1e-6)
  ) * 0.94;
  viewScale = fitScale * zoom;
  viewOffset = [
    left + availableWidth * 0.5 - (minX + maxX) * 0.5 * viewScale,
    top + availableHeight * 0.5 - (minY + maxY) * 0.5 * viewScale,
  ];
}}

function costRgb(v) {{
  const lo = scene.layerDisplayMin ?? scene.layerMin;
  const hi = scene.layerDisplayMax ?? scene.layerMax;
  const t = Math.max(0, Math.min(1, (v - lo) / Math.max(hi - lo, 1e-6)));
  const stops = [
    [0.00, [68, 1, 84]],
    [0.25, [59, 82, 139]],
    [0.50, [33, 145, 140]],
    [0.75, [94, 201, 98]],
    [1.00, [253, 231, 37]],
  ];
  for (let i = 0; i < stops.length - 1; i++) {{
    if (t <= stops[i + 1][0]) {{
      const a = stops[i], b = stops[i + 1];
      const u = (t - a[0]) / (b[0] - a[0]);
      const r = Math.round(a[1][0] + (b[1][0] - a[1][0]) * u);
      const g = Math.round(a[1][1] + (b[1][1] - a[1][1]) * u);
      const bl = Math.round(a[1][2] + (b[1][2] - a[1][2]) * u);
      return [r, g, bl];
    }}
  }}
  return [253, 231, 37];
}}

function costColor(v, shade = 1.0) {{
  const [r, g, b] = costRgb(v);
  const rr = Math.max(0, Math.min(255, Math.round(r * shade)));
  const gg = Math.max(0, Math.min(255, Math.round(g * shade)));
  const bb = Math.max(0, Math.min(255, Math.round(b * shade)));
  return `rgb(${{rr}},${{gg}},${{bb}})`;
}}

function terrainPoint(x, y) {{
  return [x * scene.resolution, y * scene.resolution, scene.elevation[y][x]];
}}

function cellShade(x, y) {{
  const xl = Math.max(0, x - 1), xr = Math.min(scene.width - 1, x + 1);
  const yu = Math.max(0, y - 1), yd = Math.min(scene.height - 1, y + 1);
  const dzdx = scene.elevation[y][xr] - scene.elevation[y][xl];
  const dzdy = scene.elevation[yd][x] - scene.elevation[yu][x];
  const slope = Math.min(1.0, Math.hypot(dzdx, dzdy) * 0.55);
  const directional = Math.max(-1.0, Math.min(1.0, (-dzdx - 0.6 * dzdy) * 0.45));
  return Math.max(0.62, Math.min(1.18, 0.98 - 0.18 * slope + 0.16 * directional));
}}

function drawPolygon(points, fill, stroke = null, alpha = 1.0) {{
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  points.forEach((p, i) => {{
    const q = project(p);
    if (i === 0) ctx.moveTo(q[0], q[1]);
    else ctx.lineTo(q[0], q[1]);
  }});
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) {{
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 0.35;
    ctx.stroke();
  }}
  ctx.globalAlpha = 1.0;
}}

function drawLine(points, color, width, alpha = 1.0, dashed = false) {{
  if (points.length < 2) return;
  ctx.globalAlpha = alpha;
  ctx.setLineDash(dashed ? [8, 6] : []);
  ctx.beginPath();
  points.forEach((p, i) => {{
    const q = project(p);
    if (i === 0) ctx.moveTo(q[0], q[1]);
    else ctx.lineTo(q[0], q[1]);
  }});
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.globalAlpha = 1.0;
}}

function drawMarker(marker) {{
  const q = project(marker.point);
  ctx.save();
  ctx.translate(q[0], q[1]);
  ctx.fillStyle = marker.color;
  ctx.strokeStyle = '#020617';
  ctx.lineWidth = 2;
  if (marker.shape === 'circle') {{
    ctx.beginPath();
    ctx.arc(0, 0, 7, 0, 2 * Math.PI);
  }} else if (marker.shape === 'triangle') {{
    ctx.beginPath();
    ctx.moveTo(0, -9);
    ctx.lineTo(8, 7);
    ctx.lineTo(-8, 7);
    ctx.closePath();
  }} else {{
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {{
      const r = i % 2 ? 5 : 11;
      const a = -Math.PI / 2 + i * Math.PI / 5;
      const x = Math.cos(a) * r;
      const y = Math.sin(a) * r;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }}
    ctx.closePath();
  }}
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}}

function drawObstacleCell(x, y) {{
  const z = scene.elevation[y][x];
  const h = 0.9 * scene.zScale;
  const p00 = [x, y, z], p10 = [x + 1, y, scene.elevation[y][Math.min(x + 1, scene.width - 1)]];
  const p11 = [x + 1, y + 1, scene.elevation[Math.min(y + 1, scene.height - 1)][Math.min(x + 1, scene.width - 1)]];
  const p01 = [x, y + 1, scene.elevation[Math.min(y + 1, scene.height - 1)][x]];
  const top = [p00, p10, p11, p01].map(p => [p[0], p[1], p[2] + h]);
  drawPolygon(top, '#1e293b', '#020617', 0.92);
}}

function drawPathSegment(path, p0, p1) {{
  if (path.shadow) drawLine([p0, p1], path.shadow, 7, 0.35);
  const isFlight = path.kind === 'flight' || path.kind === 'fly_then_drive';
  drawLine(
    [p0, p1],
    path.color,
    path.kind === 'platform' ? 4 : 2.6,
    isFlight ? 0.72 : 0.95,
    isFlight
  );
}}

function partialPolyline(points, progress) {{
  if (!points.length) return [];
  if (progress <= 0) return [points[0]];
  if (progress >= 1) return points.slice();
  const lengths = [];
  let total = 0;
  for (let i = 1; i < points.length; i++) {{
    const a = points[i - 1], b = points[i];
    const length = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    lengths.push(length);
    total += length;
  }}
  if (total <= 1e-9) return [points[0]];
  const target = progress * total;
  const visible = [points[0]];
  let covered = 0;
  for (let i = 1; i < points.length; i++) {{
    const length = lengths[i - 1];
    if (covered + length <= target) {{
      visible.push(points[i]);
      covered += length;
      continue;
    }}
    const ratio = Math.max(0, Math.min(1, (target - covered) / Math.max(length, 1e-9)));
    const a = points[i - 1], b = points[i];
    visible.push([
      a[0] + (b[0] - a[0]) * ratio,
      a[1] + (b[1] - a[1]) * ratio,
      a[2] + (b[2] - a[2]) * ratio,
    ]);
    break;
  }}
  return visible;
}}

function playbackState() {{
  if (scene.totalTime <= 0) return {{ paths: scene.paths, agents: [] }};
  const currentPhase = scene.timeline.find(item =>
    currentTime >= item.startTime &&
    (currentTime < item.endTime || (currentTime >= scene.totalTime && item.endTime >= scene.totalTime))
  );
  const paths = [];
  const agents = [];
  for (const path of scene.paths) {{
    const duration = Math.max(path.endTime - path.startTime, 1e-9);
    const active = currentTime >= path.startTime && currentTime < path.endTime;
    if (active) {{
      const points = partialPolyline(path.points, (currentTime - path.startTime) / duration);
      paths.push({{ ...path, points }});
      if (points.length) agents.push({{ point: points[points.length - 1], color: path.color, kind: path.kind }});
      continue;
    }}
    const sameFormationPhase = currentPhase &&
      path.eventIndex === currentPhase.event_index &&
      path.phase === currentPhase.phase &&
      (path.phase === 'assemble' || path.phase === 'disassemble');
    if (sameFormationPhase && path.points.length) {{
      agents.push({{
        point: currentTime < path.startTime ? path.points[0] : path.points[path.points.length - 1],
        color: path.color,
        kind: path.kind,
      }});
    }}
  }}
  return {{ paths, agents }};
}}

function drawAgent(agent) {{
  const q = project(agent.point);
  ctx.beginPath();
  ctx.arc(q[0], q[1], agent.kind === 'platform' ? 6 : 5, 0, 2 * Math.PI);
  ctx.fillStyle = agent.color;
  ctx.fill();
  ctx.strokeStyle = '#020617';
  ctx.lineWidth = 1.8;
  ctx.stroke();
}}

function draw() {{
  ctx.clearRect(0, 0, innerWidth, innerHeight);
  const grad = ctx.createLinearGradient(0, 0, 0, innerHeight);
  grad.addColorStop(0, '#172554');
  grad.addColorStop(1, '#020617');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, innerWidth, innerHeight);
  updateView();
  const renderState = playbackState();

  const drawables = [];
  for (let y = 0; y < scene.height - 1; y++) {{
    for (let x = 0; x < scene.width - 1; x++) {{
      const pts = [terrainPoint(x, y), terrainPoint(x + 1, y), terrainPoint(x + 1, y + 1), terrainPoint(x, y + 1)];
      const depth = pts.reduce((sum, p) => sum + project(p)[2], 0) / 4;
      const cost = (scene.layer[y][x] + scene.layer[y][x + 1] + scene.layer[y + 1][x] + scene.layer[y + 1][x + 1]) / 4;
      drawables.push({{ kind: 'cell', pts, depth, cost, x, y }});
      if (scene.obstacle[y][x]) {{
        const obstacleTop = scene.elevation[y][x] + 0.9 * scene.zScale;
        drawables.push({{
          kind: 'obstacle',
          x,
          y,
          depth: viewCoordinates([(x + 0.5) * scene.resolution, (y + 0.5) * scene.resolution, obstacleTop])[2],
        }});
      }}
    }}
  }}
  for (const path of renderState.paths) {{
    for (let i = 1; i < path.points.length; i++) {{
      const p0 = path.points[i - 1], p1 = path.points[i];
      drawables.push({{
        kind: 'path',
        path,
        p0,
        p1,
        depth: (viewCoordinates(p0)[2] + viewCoordinates(p1)[2]) * 0.5,
      }});
    }}
  }}
  drawables.sort((a, b) => a.depth - b.depth);
  for (const item of drawables) {{
    if (item.kind === 'cell') {{
      drawPolygon(item.pts, costColor(item.cost, cellShade(item.x, item.y)), 'rgba(15,23,42,.28)', 0.96);
    }} else if (item.kind === 'obstacle') {{
      drawObstacleCell(item.x, item.y);
    }} else {{
      drawPathSegment(item.path, item.p0, item.p1);
    }}
    if (item.kind === 'cell' && scene.unsafe[item.y][item.x] && !scene.obstacle[item.y][item.x]) {{
      const q = project([(item.x + 0.5) * scene.resolution, (item.y + 0.5) * scene.resolution, scene.elevation[item.y][item.x] + 0.12]);
      ctx.strokeStyle = '#fb923c';
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(q[0] - 4, q[1] - 4);
      ctx.lineTo(q[0] + 4, q[1] + 4);
      ctx.moveTo(q[0] + 4, q[1] - 4);
      ctx.lineTo(q[0] - 4, q[1] + 4);
      ctx.stroke();
    }}
  }}
  scene.markers.forEach(drawMarker);
  renderState.agents.forEach(drawAgent);
}}

resize();
const requestedTime = Number(new URLSearchParams(location.hash.slice(1)).get('t'));
setMissionTime(Number.isFinite(requestedTime) ? requestedTime : 0.0);
</script>
</body>
</html>
"""
