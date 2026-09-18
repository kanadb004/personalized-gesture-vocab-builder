#!/usr/bin/env python
"""Command line driver for the recognition pipeline (O2, O3, O4, O5), with no GUI.

Modes:
  --enroll NAME --message TEXT --session FILE [FILE ...]   enroll one gesture, pooling frames
                                                            from one or more recorded sessions
  --session FILE [FILE ...]                     replay sessions, print per-frame decisions
  --session FILE [FILE ...] --smooth            replay sessions, print triggers instead
  --speak                                       speak triggers as they fire (any mode)
  (no --session)                                run live on the camera, with an overlay window

Prints mean per-pipeline-stage timing at exit.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import cv2

from pgvb.camera import Camera, FpsMeter
from pgvb.config import load
from pgvb.enroll import Enroller
from pgvb.output import MessageBoard, Speaker
from pgvb.pipeline import Pipeline
from pgvb.profile import Profile, load as load_profile, save as save_profile
from pgvb.replay import SessionPlayer


def _load_or_create_profile(path: Path, backbone_id: str) -> Profile:
    if path.exists():
        return load_profile(path)
    return Profile(user_id=path.stem, backbone_id=backbone_id)


def _print_timings(timings: dict[str, list[float]]) -> None:
    if not timings:
        return
    print("mean per-stage timing (ms):")
    for stage, values in timings.items():
        print(f"  {stage}: {sum(values) / len(values):.3f}")


def _accumulate(totals: dict[str, list[float]], timings_ms: dict[str, float]) -> None:
    for stage, value in timings_ms.items():
        totals.setdefault(stage, []).append(value)


def _run_enroll(args: argparse.Namespace, profile: Profile, pipeline: Pipeline) -> None:
    enroller = Enroller(pipeline.backbone, profile, pipeline.cfg.enroll)
    enroller.start(args.enroll, args.message)
    total_frames = 0
    for session_path in args.session:
        player = SessionPlayer(session_path)
        total_frames += len(player)
        for hand_frame, _ts_ms in player:
            if enroller.progress[0] >= pipeline.cfg.enroll.max_samples:
                break
            status = enroller.add_frame(hand_frame)
            if status.ready:
                enroller.accept_sample()
        if enroller.progress[0] >= pipeline.cfg.enroll.max_samples:
            break

    collected, needed = enroller.progress
    print(f"collected {collected} samples ({needed} max) from {total_frames} frames")
    if collected < pipeline.cfg.enroll.min_samples:
        print(
            f"not enough stable samples ({collected} < {pipeline.cfg.enroll.min_samples}); "
            "not committing",
            file=sys.stderr,
        )
        return

    quality = enroller.quality()
    print(
        f"quality: intra-sample distance {quality.intra_sample_distance:.4f}, "
        f"closest existing gesture {quality.closest_gesture} "
        f"(distance {quality.closest_distance}), collision={quality.collision}"
    )
    gesture = enroller.commit()
    save_profile(profile, args.profile)
    print(f"enrolled {gesture.name!r} with {len(gesture.examples)} examples, saved {args.profile}")


def _run_replay(args: argparse.Namespace, pipeline: Pipeline, speaker: Speaker | None) -> None:
    timings: dict[str, list[float]] = {}
    for session_path in args.session:
        player = SessionPlayer(session_path)
        for hand_frame, ts_ms in player:
            result = pipeline.process_hand_frame(hand_frame, ts_ms)
            _accumulate(timings, result.timings_ms)
            if args.smooth:
                if result.trigger is not None:
                    print(f"{result.trigger.ts_ms:.0f} ms: trigger {result.trigger.label!r}")
                    if speaker is not None:
                        gesture = next(
                            (g for g in pipeline.profile.gestures if g.name == result.trigger.label),
                            None,
                        )
                        if gesture is not None:
                            speaker.say(gesture.message)
            else:
                label = result.decision.label
                distance = result.decision.distance
                distance_str = f"{distance:.4f}" if distance is not None else "None"
                print(f"{ts_ms:.0f} ms: label={label} distance={distance_str}")
    _print_timings(timings)


def _run_live(
    args: argparse.Namespace,
    cfg,
    pipeline: Pipeline,
    speaker: Speaker | None,
    pending_trigger_ts: deque[float] | None = None,
) -> None:
    board = MessageBoard(cfg.output.board_size)
    timings: dict[str, list[float]] = {}
    fps_meter = FpsMeter()
    with Camera(
        index=cfg.camera.index, width=cfg.camera.width, height=cfg.camera.height, mirror=cfg.camera.mirror
    ) as cam:
        print("press q to quit")
        while True:
            result = cam.read()
            if result is None:
                print("camera read failed", file=sys.stderr)
                break
            frame, ts_ms = result
            frame_result = pipeline.process(frame, ts_ms)
            _accumulate(timings, frame_result.timings_ms)
            fps = fps_meter.tick()

            if frame_result.trigger is not None:
                gesture = next(
                    (g for g in pipeline.profile.gestures if g.name == frame_result.trigger.label),
                    None,
                )
                message = gesture.message if gesture is not None else frame_result.trigger.label
                board.add(message)
                if speaker is not None:
                    if pending_trigger_ts is not None:
                        pending_trigger_ts.append(time.time() * 1000.0)
                    speaker.say(message)

            label = frame_result.confirmed_label or "no gesture"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(
                frame, f"fps: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )
            for i, message in enumerate(reversed(board.latest())):
                cv2.putText(
                    frame, message, (10, 90 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1
                )
            cv2.imshow("pgvb recognize", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
    _print_timings(timings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--enroll", default=None, help="gesture name to enroll")
    parser.add_argument("--message", default=None, help="message spoken for the enrolled gesture")
    parser.add_argument(
        "--session",
        default=None,
        nargs="+",
        help="replay these .npz files (in order) instead of the camera; with --enroll, their "
        "frames are pooled into one gesture",
    )
    parser.add_argument("--smooth", action="store_true", help="print triggers instead of per-frame decisions")
    parser.add_argument("--speak", action="store_true", help="speak triggers as they fire")
    args = parser.parse_args()

    cfg = load()
    profile = _load_or_create_profile(args.profile, cfg.backbone.id)
    pipeline = Pipeline(cfg, profile)

    speaker = None
    pending_trigger_ts: deque[float] = deque()
    latencies_ms: list[float] = []
    if args.speak:
        speaker = Speaker(
            backend=cfg.output.tts_backend,
            rate=cfg.output.tts_rate,
            voice=cfg.output.tts_voice,
            on_started=lambda event: latencies_ms.append(event.ts_ms - pending_trigger_ts.popleft())
            if pending_trigger_ts
            else None,
        )

    try:
        if args.enroll is not None:
            if args.session is None:
                print("--enroll requires --session", file=sys.stderr)
                return 1
            _run_enroll(args, profile, pipeline)
        elif args.session is not None:
            _run_replay(args, pipeline, speaker)
        else:
            _run_live(args, cfg, pipeline, speaker, pending_trigger_ts)
    finally:
        if speaker is not None:
            speaker.stop()
    if latencies_ms:
        print(
            f"trigger to speech start: median {statistics.median(latencies_ms):.1f} ms "
            f"over {len(latencies_ms)} triggers"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
