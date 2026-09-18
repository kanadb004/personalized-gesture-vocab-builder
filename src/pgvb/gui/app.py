"""Main window: live camera view with landmark overlay, status strip, message board, gesture
list, and the Enroll / Flag miss / Manage / Mute buttons (O2, O5, document 5.5)."""

from __future__ import annotations

import queue
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import ImageTk

from pgvb.config import Config, load as load_config
from pgvb.embedding import Backbone
from pgvb.gui._worker import CameraWorker
from pgvb.gui.dialogs import FlagMissDialog, ManageDialog
from pgvb.gui.enroll_wizard import EnrollWizard
from pgvb.output import MessageBoard, Speaker
from pgvb.profile import Profile, load as load_profile, save as save_profile

_MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
_POLL_MS = 30


def _load_or_create_profile(path: Path, backbone_id: str) -> Profile:
    if path.exists():
        return load_profile(path)
    profile = Profile(user_id=path.stem, backbone_id=backbone_id)
    save_profile(profile, path)
    return profile


class App:
    def __init__(self, profile_path: Path, camera_index: int | None = None) -> None:
        self.cfg: Config = load_config()
        if camera_index is not None:
            self.cfg.camera.index = camera_index
        self.profile_path = profile_path
        self.profile = _load_or_create_profile(profile_path, self.cfg.backbone.id)
        self.backbone = Backbone.load(_MODELS_DIR / f"{self.cfg.backbone.id}.npz")

        self.root = tk.Tk()
        self.root.title(f"PGVB - {self.profile.user_id}")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.speaker = Speaker(
            backend=self.cfg.output.tts_backend,
            rate=self.cfg.output.tts_rate,
            voice=self.cfg.output.tts_voice,
        )
        self.board = MessageBoard(self.cfg.output.board_size)

        self._active_view = None  # EnrollWizard or FlagMissDialog while open
        self._manage_view = None  # ManageDialog while open, independent of _active_view
        self._photo = None  # keep a reference so Tk does not garbage collect it
        self._last_hand_found = False
        self._last_fps = 0.0
        self._last_label: str | None = None
        self._last_distance: float | None = None

        self._build_widgets()
        self._refresh_gestures()

        self.out_queue: "queue.Queue[dict]" = queue.Queue()
        self.worker = CameraWorker(
            self.cfg, self.profile, self.profile_path, self.backbone, self.speaker, self.out_queue
        )
        self.worker.start()
        self.root.after(_POLL_MS, self._poll)

    def _build_widgets(self) -> None:
        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="nsew")
        right = ttk.Frame(self.root, padding=8)
        right.grid(row=0, column=1, sticky="nsew")

        self.video_label = ttk.Label(left)
        self.video_label.grid(row=0, column=0)

        self.status_var = tk.StringVar(value="starting camera...")
        ttk.Label(left, textvariable=self.status_var, font=("TkDefaultFont", 12)).grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )

        buttons = ttk.Frame(left)
        buttons.grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Button(buttons, text="Enroll", command=self.open_enroll).grid(row=0, column=0, padx=2)
        ttk.Button(buttons, text="Flag miss", command=self.open_flag).grid(row=0, column=1, padx=2)
        ttk.Button(buttons, text="Manage", command=self.open_manage).grid(row=0, column=2, padx=2)
        self.mute_var = tk.StringVar(value="Mute")
        ttk.Button(buttons, textvariable=self.mute_var, command=self.toggle_mute).grid(
            row=0, column=3, padx=2
        )

        ttk.Label(right, text="Message board", font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.board_text = tk.Text(right, width=36, height=6, state="disabled")
        self.board_text.grid(row=1, column=0, pady=(0, 8))

        ttk.Label(right, text="Gestures", font=("TkDefaultFont", 12, "bold")).grid(
            row=2, column=0, sticky="w"
        )
        self.gesture_tree = ttk.Treeview(
            right, columns=("name", "message", "count"), show="headings", height=10
        )
        self.gesture_tree.heading("name", text="name")
        self.gesture_tree.heading("message", text="message")
        self.gesture_tree.heading("count", text="examples")
        self.gesture_tree.column("name", width=100)
        self.gesture_tree.column("message", width=200)
        self.gesture_tree.column("count", width=70, anchor="center")
        self.gesture_tree.grid(row=3, column=0, sticky="nsew")

    def _refresh_gestures(self) -> None:
        self.gesture_tree.delete(*self.gesture_tree.get_children())
        for gesture in self.profile.gestures:
            self.gesture_tree.insert(
                "", "end", iid=gesture.id,
                values=(gesture.name, gesture.message, len(gesture.examples)),
            )

    def open_enroll(self) -> None:
        if self._active_view is not None:
            return
        self._active_view = EnrollWizard(self)

    def open_flag(self) -> None:
        if self._active_view is not None:
            return
        if not self.profile.gestures:
            self.status_var.set("enroll a gesture before flagging a miss")
            return
        self._active_view = FlagMissDialog(self)

    def open_manage(self) -> None:
        if self._manage_view is not None:
            return
        self._manage_view = ManageDialog(self)

    def close_manage_view(self) -> None:
        self._manage_view = None

    def toggle_mute(self) -> None:
        self.speaker.mute = not self.speaker.mute
        self.mute_var.set("Unmute" if self.speaker.mute else "Mute")

    def close_active_view(self) -> None:
        self._active_view = None

    def send_command(self, cmd: dict) -> None:
        self.worker.send(cmd)

    def _poll(self) -> None:
        try:
            while True:
                msg = self.out_queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        self.root.after(_POLL_MS, self._poll)

    def _handle_message(self, msg: dict) -> None:
        kind = msg["type"]
        if kind == "gestures_changed":
            self._refresh_gestures()
            if self._manage_view is not None:
                self._manage_view.on_gestures_changed()
            return
        if kind == "error":
            self.status_var.set(f"error: {msg['msg']}")
            return
        if kind == "trigger":
            self.board.add(msg["message"])
            self._render_board()
            return

        if self._active_view is not None and kind in (
            "frame", "enroll_progress", "enroll_sample", "enroll_committed",
            "enroll_error", "flag_progress", "flag_committed",
        ):
            self._active_view.handle_message(msg)
            return

        if kind == "frame":
            self._render_frame(msg["image"])
            self._last_hand_found = msg["hand_found"]
            self._last_fps = msg["fps"]
            self._render_status()
        elif kind == "decision":
            self._last_label = msg["confirmed_label"]
            self._last_distance = msg["distance"]
            self._render_status()

    def _render_status(self) -> None:
        hand = "hand found" if self._last_hand_found else "no hand"
        label = self._last_label or "no gesture"
        distance = self._last_distance
        distance_str = f"{distance:.3f}" if distance is not None else "-"
        self.status_var.set(
            f"{hand}  |  {label}  |  distance: {distance_str}  |  fps: {self._last_fps:.1f}"
        )

    def _render_frame(self, image) -> None:
        self._photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self._photo)

    def _render_board(self) -> None:
        self.board_text.configure(state="normal")
        self.board_text.delete("1.0", "end")
        for message in self.board.latest():
            self.board_text.insert("end", message + "\n")
        self.board_text.configure(state="disabled")

    def on_close(self) -> None:
        self.worker.stop()
        self.worker.join(timeout=2.0)
        self.speaker.stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main(profile_path: Path, camera_index: int | None = None) -> int:
    try:
        app = App(profile_path, camera_index)
    except RuntimeError as exc:
        print(f"pgvb app failed to start: {exc}", file=sys.stderr)
        return 1
    app.run()
    return 0
