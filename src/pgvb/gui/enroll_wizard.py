"""Three-step enrollment wizard: name and message, capture, review (O2). Frames and progress
arrive from the App's CameraWorker via `handle_message`; the wizard itself never touches the
camera or the profile directly, it only sends commands and reacts to what comes back."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from PIL import ImageTk


class EnrollWizard(tk.Toplevel):
    def __init__(self, app) -> None:
        super().__init__(app.root)
        self.app = app
        self.title("Enroll a gesture")
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.resizable(False, False)

        self._started = False
        self._name = ""
        self._message = ""
        self._count = 0
        self._min = app.cfg.enroll.min_samples
        self._max = app.cfg.enroll.max_samples
        self._last_quality: dict | None = None
        self._photo = None

        self._frame1 = self._build_step1()
        self._frame2 = self._build_step2()
        self._frame3 = self._build_step3()
        self._show(self._frame1)

    def _show(self, frame: ttk.Frame) -> None:
        for f in (self._frame1, self._frame2, self._frame3):
            f.pack_forget()
        frame.pack(padx=12, pady=12)

    def _build_step1(self) -> ttk.Frame:
        frame = ttk.Frame(self)
        ttk.Label(frame, text="Gesture name").grid(row=0, column=0, sticky="w")
        self.name_entry = ttk.Entry(frame, width=30)
        self.name_entry.grid(row=0, column=1, pady=4)
        ttk.Label(frame, text="Message to speak").grid(row=1, column=0, sticky="w")
        self.message_entry = ttk.Entry(frame, width=30)
        self.message_entry.grid(row=1, column=1, pady=4)
        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, columnspan=2, pady=(8, 0))
        ttk.Button(buttons, text="Cancel", command=self.cancel).grid(row=0, column=0, padx=4)
        ttk.Button(buttons, text="Next", command=self._start_capture).grid(row=0, column=1, padx=4)
        return frame

    def _build_step2(self) -> ttk.Frame:
        frame = ttk.Frame(self)
        self.video_label = ttk.Label(frame)
        self.video_label.grid(row=0, column=0, columnspan=2)
        self.progress_var = tk.StringVar(value=f"0 of {self._max} (need {self._min})")
        ttk.Label(frame, textvariable=self.progress_var, font=("TkDefaultFont", 12)).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        self.status_var = tk.StringVar(value="hold the pose steady")
        ttk.Label(frame, textvariable=self.status_var).grid(
            row=2, column=0, columnspan=2, sticky="w"
        )
        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, columnspan=2, pady=(8, 0))
        ttk.Button(buttons, text="Cancel", command=self.cancel).grid(row=0, column=0, padx=4)
        self.next_button = ttk.Button(buttons, text="Next", command=self._go_review, state="disabled")
        self.next_button.grid(row=0, column=1, padx=4)
        return frame

    def _build_step3(self) -> ttk.Frame:
        frame = ttk.Frame(self)
        self.review_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.review_var, justify="left").grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        buttons = ttk.Frame(frame)
        buttons.grid(row=1, column=0, columnspan=3, pady=(8, 0))
        ttk.Button(buttons, text="Cancel", command=self.cancel).grid(row=0, column=0, padx=4)
        ttk.Button(buttons, text="Redo", command=self._redo).grid(row=0, column=1, padx=4)
        ttk.Button(buttons, text="Save", command=self._save).grid(row=0, column=2, padx=4)
        return frame

    def _start_capture(self) -> None:
        name = self.name_entry.get().strip()
        message = self.message_entry.get().strip()
        if not name or not message:
            messagebox.showwarning("Enroll", "Name and message are both required.", parent=self)
            return
        self._name = name
        self._message = message
        self._count = 0
        self.app.send_command({"cmd": "enroll_start", "name": name, "message": message})
        self._started = True
        self._show(self._frame2)

    def _redo(self) -> None:
        self.app.send_command({"cmd": "enroll_start", "name": self._name, "message": self._message})
        self._count = 0
        self.progress_var.set(f"0 of {self._max} (need {self._min})")
        self.next_button.configure(state="disabled")
        self._show(self._frame2)

    def _go_review(self) -> None:
        q = self._last_quality
        lines = [f"Collected {self._count} samples for {self._name!r}."]
        if q is not None:
            lines.append(f"Intra-sample distance: {q['intra_distance']:.4f}")
            if q["closest_gesture"] is not None:
                lines.append(
                    f"Closest existing gesture: {q['closest_gesture']} "
                    f"(distance {q['closest_distance']:.4f})"
                )
                if q["collision"]:
                    lines.append("Warning: this looks close to an existing gesture.")
        self.review_var.set("\n".join(lines))
        self._show(self._frame3)

    def _save(self) -> None:
        self.app.send_command({"cmd": "enroll_commit"})

    def cancel(self) -> None:
        if self._started:
            self.app.send_command({"cmd": "enroll_cancel"})
        self.app.close_active_view()
        self.destroy()

    def handle_message(self, msg: dict) -> None:
        kind = msg["type"]
        if kind == "frame":
            self._photo = ImageTk.PhotoImage(msg["image"])
            self.video_label.configure(image=self._photo)
        elif kind == "enroll_progress":
            self._count = msg["count"]
            self.progress_var.set(f"{self._count} of {msg['max']} (need {msg['min']})")
            reason = msg["reason"] or ""
            hint = {
                "no_hand": "no hand detected",
                "collecting": "hold the pose steady",
                "moving": "still moving, hold steady",
            }.get(reason, "hold the pose steady")
            self.status_var.set(hint)
            self.next_button.configure(state="normal" if self._count >= self._min else "disabled")
        elif kind == "enroll_sample":
            self._count = msg["count"]
            self._last_quality = msg
            self.progress_var.set(f"{self._count} of {msg['max']} (need {msg['min']})")
            self.status_var.set("sample captured, change the angle slightly" if self._count % 3 == 0 else "sample captured")
            self.next_button.configure(state="normal" if self._count >= self._min else "disabled")
        elif kind == "enroll_committed":
            self.app.close_active_view()
            self.destroy()
        elif kind == "enroll_error":
            messagebox.showerror("Enroll", msg["msg"], parent=self)
