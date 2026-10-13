"""Command line: ``courtvision run input_videos/game.mp4 --config configs/default.yaml``."""
from __future__ import annotations

import argparse
import logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="courtvision")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Run the full pipeline on a video")
    run.add_argument("video")
    run.add_argument("--config", default=None)
    run.add_argument("--output", default=None, help="Output folder (default: output.dir in config)")
    run.add_argument("--no-video", action="store_true", help="Skip writing the annotated video")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from .config import Config
    from .pipeline import Pipeline

    cfg = Config.load(args.config)
    if args.no_video:
        cfg.output.write_video = False
    result = Pipeline(cfg).run(args.video, args.output)
    print(f"{len(result.players)} frames, {len(result.team_of)} player tracks, "
          f"{len(result.possessions)} touches, {len(result.events)} events")


if __name__ == "__main__":
    main()
