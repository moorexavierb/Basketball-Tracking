"""Assemble per-frame 3D scenes and export them.

* ``build_scene``   -> plain dict (JSON-serialisable) for any 3D front-end
                       (three.js, Unity, Blender import script).
* ``scene_to_plotly`` -> self-contained interactive HTML with a play/scrub bar.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..court.geometry import RIM_HEIGHT_M, CourtSpec
from ..types import PipelineResult


def build_scene(result: PipelineResult, court: CourtSpec, team_names: list[str]) -> dict:
    n = len(result.players)
    frames = []
    for f in range(n):
        players = []
        for tid, track in result.court_positions.items():
            p = track[f]
            if p is None:
                continue
            players.append({"id": int(tid), "team": int(result.team_of.get(tid, -1)),
                            "x": round(p[0], 3), "y": round(p[1], 3),
                            "has_ball": result.possessor[f] == tid})
        b = result.ball_3d[f]
        frames.append({"frame": f, "t": round(f / result.fps, 3), "players": players,
                       "ball": None if b is None else [round(v, 3) for v in b]})
    return {
        "court": {"standard": court.name, "length": court.length, "width": court.width,
                  "hoops": court.hoops.round(3).tolist()},
        "teams": team_names,
        "fps": result.fps,
        "events": [{"frame": e.frame, "type": e.type, "player": e.player_id, "team": e.team,
                    "target": e.target_id} for e in result.events],
        "frames": frames,
    }


def save_scene_json(scene: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scene))
    return path


def _court_lines(court: CourtSpec) -> list[np.ndarray]:
    L, W = court.length, court.width
    lines = [np.array([(0, 0), (L, 0), (L, W), (0, W), (0, 0)]), np.array([(L / 2, 0), (L / 2, W)])]
    th = np.linspace(0, 2 * np.pi, 48)
    lines.append(np.column_stack([L / 2 + court.center_circle_radius * np.cos(th),
                                  W / 2 + court.center_circle_radius * np.sin(th)]))
    lane_lo = (W - court.lane_width) / 2
    corner_y = W / 2 - court.corner_three_offset
    arc_x = np.sqrt(court.three_radius ** 2 - corner_y ** 2)
    ang = np.arctan2(corner_y, arc_x)
    for side, (hx, hy, _) in enumerate(court.hoops):
        bx, sign = (0, 1) if side == 0 else (L, -1)
        ft = bx + sign * court.free_throw_dist
        lines.append(np.array([(bx, lane_lo), (ft, lane_lo), (ft, W - lane_lo), (bx, W - lane_lo)]))
        a = np.linspace(-ang, ang, 40)
        arc = np.column_stack([hx + sign * court.three_radius * np.cos(a), hy + court.three_radius * np.sin(a)])
        lines.append(np.vstack([(bx, court.corner_three_offset), arc[0], arc, arc[-1], (bx, W - court.corner_three_offset)]))
    return lines


def scene_to_plotly(scene: dict, court: CourtSpec, path: str | Path, every: int = 2,
                    team_colors: tuple[str, str] = ("#e8e8e8", "#1f4fd1")) -> Path:
    import plotly.graph_objects as go

    static = []
    for ln in _court_lines(court):
        static.append(go.Scatter3d(x=ln[:, 0], y=ln[:, 1], z=np.zeros(len(ln)), mode="lines",
                                   line=dict(color="white", width=3), hoverinfo="skip", showlegend=False))
    for hx, hy, hz in court.hoops:
        th = np.linspace(0, 2 * np.pi, 24)
        static.append(go.Scatter3d(x=hx + 0.23 * np.cos(th), y=hy + 0.23 * np.sin(th), z=np.full(24, hz),
                                   mode="lines", line=dict(color="orangered", width=5), showlegend=False))

    def dynamic(fr):
        traces = []
        for team in (0, 1, -1):
            ps = [p for p in fr["players"] if p["team"] == team]
            color = team_colors[team] if team in (0, 1) else "#888888"
            name = scene["teams"][team] if team in (0, 1) else "Unassigned"
            traces.append(go.Scatter3d(
                x=[p["x"] for p in ps], y=[p["y"] for p in ps], z=[0.9] * len(ps), mode="markers+text",
                text=[str(p["id"]) for p in ps], textposition="top center", name=name,
                marker=dict(size=[11 if p["has_ball"] else 8 for p in ps], color=color,
                            line=dict(color="black", width=2))))
        b = fr["ball"]
        traces.append(go.Scatter3d(x=[b[0]] if b else [], y=[b[1]] if b else [], z=[b[2]] if b else [],
                                   mode="markers", name="Ball", marker=dict(size=6, color="orange")))
        return traces

    frames = scene["frames"][::every] or [{"frame": 0, "players": [], "ball": None}]
    fig = go.Figure(data=static + dynamic(frames[0]),
                    frames=[go.Frame(data=dynamic(fr), traces=list(range(len(static), len(static) + 4)),
                                     name=str(fr["frame"])) for fr in frames])
    fig.update_layout(
        scene=dict(xaxis=dict(range=[-1, court.length + 1], title="x (m)"),
                   yaxis=dict(range=[-1, court.width + 1], title="y (m)"),
                   zaxis=dict(range=[0, RIM_HEIGHT_M + 2], title="z (m)"),
                   aspectmode="manual", aspectratio=dict(x=2, y=court.width / court.length * 2, z=0.5),
                   bgcolor="#c8a26e"),
        updatemenus=[dict(type="buttons", buttons=[
            dict(label="Play", method="animate",
                 args=[None, dict(frame=dict(duration=1000 * every / scene["fps"], redraw=True), fromcurrent=True)]),
            dict(label="Pause", method="animate", args=[[None], dict(mode="immediate")])])],
        sliders=[dict(steps=[dict(method="animate", label=f["name"],
                                  args=[[f["name"]], dict(mode="immediate", frame=dict(redraw=True))])
                             for f in fig.frames])],
        margin=dict(l=0, r=0, t=30, b=0), title="3D play reconstruction")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path
