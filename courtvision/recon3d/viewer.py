"""Export a scene as a self-contained three.js page with animated player bodies.

The page (``player_viewer.html``, next to this module) poses an articulated
body for every player from ``scene.json`` alone: speed drives the gait, the
defending team faces the ball, shot and pass events drive jumps and throws,
and measured 3D skeletons (``player["pose"]``) override the synthesized limbs
when the pose stage ran. An optional rigged Mixamo-style ``.glb`` replaces the
built-in mannequin and is driven by the same skeleton.

The scene and character are embedded, so the file works offline apart from
three.js itself, which loads from jsDelivr.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Sequence

TEMPLATE = Path(__file__).with_name("player_viewer.html")
_HEAD = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
         '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')


def _js(value) -> str:
    # JSON is valid JS; escape "</" so embedded strings can't close the <script> tag.
    return json.dumps(value, separators=(",", ":")).replace("</", "<\\/")


def render_viewer(scene: dict | None = None, team_colors: Sequence[str] | None = None,
                  character_glb: str | Path | None = None, standalone: bool = True) -> str:
    """HTML for the viewer. With no scene it plays a built-in sample possession."""
    html = TEMPLATE.read_text()
    character = None
    if character_glb:
        data = base64.b64encode(Path(character_glb).read_bytes()).decode()
        character = f"data:model/gltf-binary;base64,{data}"
    for token, value in (("/*__CV_SCENE__*/null", scene), ("/*__CV_TEAM_COLORS__*/null", team_colors),
                         ("/*__CV_CHARACTER__*/null", character)):
        if value is not None:
            html = html.replace(token, _js(list(value) if token.endswith("COLORS__*/null") else value), 1)
    if standalone:
        body_start = html.index("</style>") + len("</style>")
        html = _HEAD + html[:body_start] + "\n</head>\n<body>" + html[body_start:] + "\n</body>\n</html>\n"
    return html


def scene_to_threejs(scene: dict, path: str | Path, team_colors: Sequence[str] | None = None,
                     character_glb: str | Path | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_viewer(scene, team_colors, character_glb))
    return path
