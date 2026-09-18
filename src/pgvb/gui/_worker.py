"""Background thread owning the camera, hand tracker, and pipeline for the desktop app.

All mutation of the shared Profile happens on this thread only, driven by commands from the
GUI thread; the GUI thread only reads `profile.gestures` for display, after a `gestures_changed`
message tells it the list changed. Frames, decisions, and enrollment/refinement progress are
reported back on `out_queue` as tagged dicts; see the `_emit_*` helpers below for the shapes.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pgvb.camera import Camera, FpsMeter
from pgvb.config import Config
from pgvb.embedding import Backbone
from pgvb.enroll import Enroller, Refiner
from pgvb.features import normalize_landmarks
from pgvb.landmarks import HandFrame, HandTracker
from pgvb.output import Speaker
from pgvb.pipeline import Pipeline
from pgvb.profile import Profile, save as save_profile
from pgvb.prototypes import nearest

# MediaPipe's 21-landmark hand skeleton connections (index pairs), for the overlay.
_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _draw_overlay(frame: np.ndarray, hand_frame: HandFrame | None) -> None:
    if hand_frame is None:
        return
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in hand_frame.landmarks]
    for a, b in _CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


class CameraWorker(threading.Thread):
    """Owns the camera and pipeline for the app's lifetime. Runs in three modes: `live`
    (recognition, smoothing, triggers, speech), `enroll` (feeds frames to an Enroller, auto
    accepting stable samples), and `flag` (holds the last stable frame and reports the nearest
    existing gesture, for the flag-miss dialog to confirm)."""

    def __init__(
        self,
        cfg: Config,
        profile: Profile,
        profile_path: Path,
        backbone: Backbone,
        speaker: Speaker,
        out_queue: "queue.Queue[dict]",
    ) -> None:
        super().__init__(daemon=True)
        self.cfg = cfg
        self.profile = profile
        self.profile_path = profile_path
        self.backbone = backbone
        self.speaker = speaker
        self.out_queue = out_queue
        self.cmd_queue: "queue.Queue[dict]" = queue.Queue()

        self._mode = "live"
        self._stop = False

        self._pipeline = Pipeline(cfg, profile, backbone)
        self._tracker: HandTracker | None = None
        self._enroller = Enroller(backbone, profile, cfg.enroll)
        self._refiner = Refiner(backbone, profile)

        self._flag_recent: list[tuple[np.ndarray, HandFrame]] = []
        self._flag_candidate: HandFrame | None = None

    def send(self, cmd: dict) -> None:
        self.cmd_queue.put(cmd)

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        self._tracker = HandTracker(
            num_hands=self.cfg.landmarks.num_hands,
            min_hand_score=self.cfg.landmarks.min_hand_score,
            min_hand_detection_confidence=self.cfg.landmarks.min_hand_detection_confidence,
            min_tracking_confidence=self.cfg.landmarks.min_tracking_confidence,
        )
        fps_meter = FpsMeter()
        try:
            with Camera(
                index=self.cfg.camera.index,
                width=self.cfg.camera.width,
                height=self.cfg.camera.height,
                mirror=self.cfg.camera.mirror,
            ) as cam:
                while not self._stop:
                    self._drain_commands()
                    read = cam.read()
                    if read is None:
                        self.out_queue.put({"type": "error", "msg": "camera read failed"})
                        break
                    frame, ts_ms = read
                    hand_frame = self._tracker.track(frame, ts_ms)
                    fps = fps_meter.tick()
                    self._step(frame, hand_frame, ts_ms, fps)
        except RuntimeError as exc:
            self.out_queue.put({"type": "error", "msg": str(exc)})

    def _drain_commands(self) -> None:
        while True:
            try:
                cmd = self.cmd_queue.get_nowait()
            except queue.Empty:
                return
            self._handle_command(cmd)

    def _handle_command(self, cmd: dict) -> None:
        kind = cmd["cmd"]
        if kind == "enroll_start":
            self._enroller.start(cmd["name"], cmd["message"])
            self._mode = "enroll"
        elif kind == "enroll_cancel":
            self._enroller.cancel()
            self._mode = "live"
        elif kind == "enroll_commit":
            self._do_enroll_commit()
        elif kind == "flag_start":
            self._flag_recent = []
            self._flag_candidate = None
            self._mode = "flag"
        elif kind == "flag_cancel":
            self._mode = "live"
        elif kind == "flag_confirm":
            self._do_flag_confirm(cmd["gesture_id"])
        elif kind == "gesture_rename":
            self.profile.rename(cmd["gesture_id"], cmd["name"])
            self._after_profile_change()
        elif kind == "gesture_set_message":
            self.profile.set_message(cmd["gesture_id"], cmd["message"])
            self._after_profile_change()
        elif kind == "gesture_delete":
            self.profile.remove_gesture(cmd["gesture_id"])
            self._after_profile_change()
        elif kind == "gesture_undo_refinement":
            try:
                self._refiner.undo_last(cmd["gesture_id"])
            except (KeyError, RuntimeError) as exc:
                self.out_queue.put({"type": "error", "msg": str(exc)})
                return
            self._after_profile_change()

    def _after_profile_change(self) -> None:
        self._pipeline.reload_profile()
        save_profile(self.profile, self.profile_path)
        self.out_queue.put({"type": "gestures_changed"})

    def _do_enroll_commit(self) -> None:
        try:
            gesture = self._enroller.commit()
        except RuntimeError as exc:
            self.out_queue.put({"type": "enroll_error", "msg": str(exc)})
            return
        self._pipeline.reload_profile()
        save_profile(self.profile, self.profile_path)
        self._mode = "live"
        self.out_queue.put({"type": "enroll_committed", "name": gesture.name})
        self.out_queue.put({"type": "gestures_changed"})

    def _do_flag_confirm(self, gesture_id: str) -> None:
        if self._flag_candidate is None:
            self.out_queue.put({"type": "error", "msg": "no stable sample captured yet"})
            return
        try:
            self._refiner.refine(gesture_id, self._flag_candidate)
        except KeyError as exc:
            self.out_queue.put({"type": "error", "msg": str(exc)})
            return
        self._pipeline.reload_profile()
        save_profile(self.profile, self.profile_path)
        self._mode = "live"
        self.out_queue.put({"type": "flag_committed", "gesture_id": gesture_id})
        self.out_queue.put({"type": "gestures_changed"})

    def _step(self, frame: np.ndarray, hand_frame: HandFrame | None, ts_ms: float, fps: float) -> None:
        _draw_overlay(frame, hand_frame)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if self._mode == "enroll":
            self._step_enroll(hand_frame)
        elif self._mode == "flag":
            self._step_flag(hand_frame)
        else:
            self._step_live(hand_frame, ts_ms)

        self.out_queue.put(
            {
                "type": "frame",
                "image": image,
                "hand_found": hand_frame is not None,
                "fps": fps,
                "mode": self._mode,
            }
        )

    def _step_live(self, hand_frame: HandFrame | None, ts_ms: float) -> None:
        result = self._pipeline.process_hand_frame(hand_frame, ts_ms)
        self.out_queue.put(
            {
                "type": "decision",
                "label": result.decision.label,
                "distance": result.decision.distance,
                "confirmed_label": result.confirmed_label,
            }
        )
        if result.trigger is not None:
            gesture = next(
                (g for g in self.profile.gestures if g.name == result.trigger.label), None
            )
            message = gesture.message if gesture is not None else result.trigger.label
            self.speaker.say(message)
            self.out_queue.put({"type": "trigger", "label": result.trigger.label, "message": message})

    def _step_enroll(self, hand_frame: HandFrame | None) -> None:
        status = self._enroller.add_frame(hand_frame)
        count, max_count = self._enroller.progress
        if status.ready and count < max_count:
            self._enroller.accept_sample()
            count, max_count = self._enroller.progress
            quality = self._enroller.quality()
            self.out_queue.put(
                {
                    "type": "enroll_sample",
                    "count": count,
                    "max": max_count,
                    "min": self.cfg.enroll.min_samples,
                    "intra_distance": quality.intra_sample_distance,
                    "closest_gesture": quality.closest_gesture,
                    "closest_distance": quality.closest_distance,
                    "collision": quality.collision,
                }
            )
        else:
            self.out_queue.put(
                {
                    "type": "enroll_progress",
                    "ready": status.ready,
                    "reason": status.reason,
                    "count": count,
                    "max": max_count,
                    "min": self.cfg.enroll.min_samples,
                }
            )

    def _step_flag(self, hand_frame: HandFrame | None) -> None:
        cfg = self.cfg.enroll
        if hand_frame is None:
            self._flag_recent = []
            self.out_queue.put({"type": "flag_progress", "ready": False})
            return

        feature = normalize_landmarks(hand_frame.landmarks, hand_frame.handedness)
        self._flag_recent.append((feature, hand_frame))
        if len(self._flag_recent) > cfg.stability_frames:
            self._flag_recent.pop(0)
        if len(self._flag_recent) < cfg.stability_frames:
            self.out_queue.put({"type": "flag_progress", "ready": False})
            return

        feats = np.stack([f for f, _ in self._flag_recent])
        motion = float(np.linalg.norm(feats - feats.mean(axis=0), axis=1).max())
        if motion > cfg.stability_max_motion:
            self.out_queue.put({"type": "flag_progress", "ready": False})
            return

        self._flag_candidate = self._flag_recent[-1][1]
        embedding = self.backbone.embed(feature)
        nearest_id = nearest_name = None
        nearest_distance = None
        if self.profile.gestures:
            protos = np.stack([g.prototype for g in self.profile.gestures])
            idx, d1, _ = nearest(embedding, protos)
            nearest_id = self.profile.gestures[idx].id
            nearest_name = self.profile.gestures[idx].name
            nearest_distance = d1
        self.out_queue.put(
            {
                "type": "flag_progress",
                "ready": True,
                "nearest_gesture_id": nearest_id,
                "nearest_name": nearest_name,
                "nearest_distance": nearest_distance,
            }
        )
