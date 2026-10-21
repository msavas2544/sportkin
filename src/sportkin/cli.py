"""Command line interface.

Examples::

    sportkin download-model                       # once, needs internet
    sportkin cmj jump.mp4 --fps 240 --out results/ --annotate --blur-face
    sportkin cmj jump.mp4 --fps 240 --frames 412:530 --frames 1210:1325
    sportkin cmj --pose results/jump_pose.csv --fps 240     # re-analyse, no video
    sportkin agreement data.csv --a video_cm --b mat_cm --label "CMJ height (cm)" --lang tr
    sportkin reliability data.csv --a trial1 --b trial2 --lang tr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _progress(i, n):
    if n and (i % 50 == 0 or i == n):
        sys.stderr.write(f"\r  pose: {i}/{n} frames")
        if i == n:
            sys.stderr.write("\n")
        sys.stderr.flush()


def _parse_frames(items):
    out = []
    for it in items or []:
        a, b = it.split(":")
        out.append((int(a), int(b)))
    return out


def cmd_download(args):
    from .pose import download_model

    path = download_model(args.variant)
    print(f"Model saved: {path}")


def cmd_cmj(args):
    from .cmj import analyze_cmj
    from .io import write_table
    from .pose import PoseSeries

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.pose:
        pose = PoseSeries.from_csv(args.pose, fps=args.fps)  # fps None -> stored times
        stem = Path(args.pose).stem.replace("_pose", "")
        video = None
    else:
        if not args.video:
            sys.exit("error: give a VIDEO or --pose CSV")
        from .pose import MediaPipeBackend
        from .video import iter_frames, probe

        info = probe(args.video)
        stem = Path(args.video).stem
        measured = info.measured_fps
        print(f"{info.path.name}: {info.width}x{info.height}, {info.n_frames} frames, "
              f"measured {measured:.2f} fps (container says {info.container_fps:.2f})")
        if info.variable_frame_rate:
            print("  note: variable frame rate detected; per-frame timestamps are used.")
        if args.fps:
            fps, times = args.fps, None
            print(f"  using capture rate --fps {args.fps:g} (timestamps from file ignored)")
        else:
            fps, times = measured, info.times - info.times[0]
            if measured < 100:
                print("  note: for slow-motion clips pass the capture rate, e.g. --fps 240")
        with MediaPipeBackend(args.model, args.variant) as backend:
            pose = backend.process(iter_frames(args.video), fps, info.width,
                                   info.height, info.n_frames, _progress, times=times)
        pose.to_csv(out / f"{stem}_pose.csv")
        video = args.video

    res = analyze_cmj(pose, side=args.side, cutoff_hz=args.cutoff,
                      manual_frames=_parse_frames(args.frames))
    write_table(res.frames, out / f"{stem}_frames.csv", decimal=args.decimal)
    if res.jumps:
        write_table(res.summary_rows(), out / f"{stem}_jumps.csv", decimal=args.decimal)
    (out / f"{stem}_summary.json").write_text(json.dumps({
        "version": __version__, "fps": res.fps, "side": res.side, "cutoff_hz": res.cutoff_hz,
        "jumps": res.summary_rows(), "warnings": res.warnings}, indent=2), encoding="utf-8")

    for w in res.warnings:
        print(f"  warning: {w}")
    for j in res.jumps:
        print(f"  jump {j.jump_index}: {j.height_cm:.1f} cm (±{j.height_resolution_cm:.1f} per frame), "
              f"flight {j.flight_time_s * 1000:.0f} ms, frames {j.takeoff_frame}-{j.landing_frame}, "
              f"peak knee flexion {j.peak_knee_flexion_cm_deg:.0f}°")

    if args.annotate and video:
        from .annotate import write_annotated_video
        from .video import iter_frames

        events = {}
        for j in res.jumps:
            events[j.takeoff_frame] = "take-off"
            events[j.landing_frame] = "landing"
        knee_col = next(c for c in res.frames if c.endswith("knee_flexion_deg"))
        path = write_annotated_video(iter_frames(video), pose, out / f"{stem}_annotated.mp4",
                                     blur_face=args.blur_face, events=events,
                                     labels={"knee flex": res.frames[knee_col]})
        print(f"  annotated video: {path}")
    print(f"Results in {out.resolve()}")


def _two_columns(args):
    from .io import read_columns

    return read_columns(args.csv, args.a, args.b)


def cmd_agreement(args, reliability: bool = False):
    from . import report, stats

    a, b = _two_columns(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    label = args.label or (args.a if not reliability else args.a.rstrip("12_"))
    dec = "," if args.lang == "tr" else "."
    if reliability:
        res = stats.reliability(a, b, args.icc)
        table = report.reliability_table([(label, res)], lang=args.lang, decimal=dec,
                                         title=args.title)
    else:
        res = stats.agreement(a, b, args.icc)
        table = report.agreement_table([(label, res)], name_a=args.name_a, name_b=args.name_b,
                                       lang=args.lang, decimal=dec, title=args.title)
    kind = "reliability" if reliability else "agreement"
    (out / f"{kind}.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    (out / f"{kind}_table.md").write_text(report.table_to_markdown(table), encoding="utf-8")
    try:
        report.table_to_docx(table, out / f"{kind}_table.docx")
        ba = stats.bland_altman(b, a) if reliability else stats.bland_altman(a, b)
        names = ("2", "1") if reliability else (args.name_a, args.name_b)
        report.bland_altman_plot(ba, out / f"{kind}_bland_altman.png", name_a=names[0],
                                 name_b=names[1], unit=args.unit, lang=args.lang)
    except ImportError:
        print("  (install 'sportkin[report]' for Word tables and figures)")
    print(report.table_to_markdown(table))
    print(f"\nResults in {out.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sportkin", description="Offline kinematics for sport science")
    p.add_argument("--version", action="version", version=f"sportkin {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download-model", help="download the MediaPipe pose model once")
    d.add_argument("--variant", choices=["lite", "full", "heavy"], default="full")
    d.set_defaults(func=cmd_download)

    c = sub.add_parser("cmj", help="analyse countermovement jumps in a side-view video")
    c.add_argument("video", nargs="?", help="video file (side view)")
    c.add_argument("--pose", help="re-analyse an existing *_pose.csv instead of a video")
    c.add_argument("--fps", type=float, default=None,
                   help="capture frame rate for slow-motion clips (e.g. 240); "
                        "default: measured from the file's frame timestamps")
    c.add_argument("--side", choices=["auto", "left", "right"], default="auto")
    c.add_argument("--cutoff", type=float, default=6.0, help="filter cut-off in Hz (default 6)")
    c.add_argument("--frames", action="append", metavar="TAKEOFF:LANDING",
                   help="manual frames, repeatable")
    c.add_argument("--model", help="path to pose_landmarker_*.task (offline use)")
    c.add_argument("--variant", choices=["lite", "full", "heavy"], default="full")
    c.add_argument("--out", default="sportkin_results")
    c.add_argument("--decimal", choices=[".", ","], default=".",
                   help="',' writes ;-separated CSV for Turkish Excel")
    c.add_argument("--annotate", action="store_true", help="write a skeleton overlay video")
    c.add_argument("--blur-face", action="store_true", help="blur faces in the overlay video")
    c.set_defaults(func=cmd_cmj)

    for name, rel in (("agreement", False), ("reliability", True)):
        s = sub.add_parser(name, help=("method comparison (validity)" if not rel
                                       else "test-retest reliability"))
        s.add_argument("csv")
        s.add_argument("--a", required=True, help="column: new method / trial 1")
        s.add_argument("--b", required=True, help="column: criterion / trial 2")
        s.add_argument("--label", default="", help="row label, e.g. 'CMJ height (cm)'")
        s.add_argument("--title", default="", help="table title")
        s.add_argument("--name-a", default="Video")
        s.add_argument("--name-b", default="Criterion")
        s.add_argument("--unit", default="cm")
        s.add_argument("--icc", default="2,1", choices=["1,1", "2,1", "3,1", "1,k", "2,k", "3,k"])
        s.add_argument("--lang", choices=["en", "tr"], default="en")
        s.add_argument("--out", default="sportkin_results")
        s.set_defaults(func=lambda a, rel=rel: cmd_agreement(a, rel))
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
