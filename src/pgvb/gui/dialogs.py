"""Flag-miss (one-sample refinement, document 5.5) and Manage gestures dialogs."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from PIL import ImageTk


class FlagMissDialog(tk.Toplevel):
    """Captures one stable sample from the live camera, defaults the gesture picker to the
    nearest existing prototype, and appends the sample to that gesture on confirm."""

    def __init__(self, app) -> None:
        super().__init__(app.root)
        self.app = app
        self.title("Flag a missed gesture")
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.resizable(False, False)

        self._photo = None
        self._names = [g.name for g in app.profile.gestures]
        self._ids = [g.id for g in app.profile.gestures]

        self.video_label = ttk.Label(self)
        self.video_label.grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 0))

        self.status_var = tk.StringVar(value="hold the missed pose steady")
        ttk.Label(self, textvariable=self.status_var).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=12, pady=(6, 0)
        )

        ttk.Label(self, text="Intended gesture").grid(row=2, column=0, sticky="w", padx=12)
        self.gesture_var = tk.StringVar(value="")
        self.gesture_combo = ttk.Combobox(
            self, textvariable=self.gesture_var, values=self._names, state="readonly", width=24
        )
        self.gesture_combo.grid(row=2, column=1, padx=12, pady=4)

        buttons = ttk.Frame(self)
        buttons.grid(row=3, column=0, columnspan=2, pady=(8, 12))
        ttk.Button(buttons, text="Cancel", command=self.cancel).grid(row=0, column=0, padx=4)
        self.confirm_button = ttk.Button(
            buttons, text="Confirm", command=self._confirm, state="disabled"
        )
        self.confirm_button.grid(row=0, column=1, padx=4)

        self.app.send_command({"cmd": "flag_start"})

    def _confirm(self) -> None:
        name = self.gesture_var.get()
        if name not in self._names:
            return
        gesture_id = self._ids[self._names.index(name)]
        self.app.send_command({"cmd": "flag_confirm", "gesture_id": gesture_id})

    def cancel(self) -> None:
        self.app.send_command({"cmd": "flag_cancel"})
        self.app.close_active_view()
        self.destroy()

    def handle_message(self, msg: dict) -> None:
        kind = msg["type"]
        if kind == "frame":
            self._photo = ImageTk.PhotoImage(msg["image"])
            self.video_label.configure(image=self._photo)
        elif kind == "flag_progress":
            if msg["ready"]:
                self.status_var.set("captured, confirm the gesture")
                if msg["nearest_name"] is not None:
                    self.gesture_var.set(msg["nearest_name"])
                self.confirm_button.configure(state="normal")
            else:
                self.status_var.set("hold the missed pose steady")
                self.confirm_button.configure(state="disabled")
        elif kind == "flag_committed":
            self.app.close_active_view()
            self.destroy()


class ManageDialog(tk.Toplevel):
    """Rename, edit message, delete, or undo the last refinement of an enrolled gesture."""

    def __init__(self, app) -> None:
        super().__init__(app.root)
        self.app = app
        self.title("Manage gestures")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.resizable(False, False)

        self.listbox = tk.Listbox(self, width=40, height=8)
        self.listbox.grid(row=0, column=0, columnspan=4, padx=12, pady=12)
        self._refresh_list()

        ttk.Button(self, text="Rename", command=self._rename).grid(row=1, column=0, padx=4, pady=(0, 12))
        ttk.Button(self, text="Edit message", command=self._edit_message).grid(
            row=1, column=1, padx=4, pady=(0, 12)
        )
        ttk.Button(self, text="Delete", command=self._delete).grid(row=1, column=2, padx=4, pady=(0, 12))
        ttk.Button(self, text="Undo last refinement", command=self._undo).grid(
            row=1, column=3, padx=4, pady=(0, 12)
        )

    def _refresh_list(self) -> None:
        self.listbox.delete(0, "end")
        for gesture in self.app.profile.gestures:
            self.listbox.insert(
                "end", f"{gesture.name}  ({gesture.message})  [{len(gesture.examples)} examples]"
            )

    def _selected_gesture(self):
        selection = self.listbox.curselection()
        if not selection:
            return None
        return self.app.profile.gestures[selection[0]]

    def _rename(self) -> None:
        gesture = self._selected_gesture()
        if gesture is None:
            return
        name = simpledialog.askstring("Rename", "New name", initialvalue=gesture.name, parent=self)
        if name:
            self.app.send_command({"cmd": "gesture_rename", "gesture_id": gesture.id, "name": name})

    def _edit_message(self) -> None:
        gesture = self._selected_gesture()
        if gesture is None:
            return
        message = simpledialog.askstring(
            "Edit message", "New message", initialvalue=gesture.message, parent=self
        )
        if message:
            self.app.send_command(
                {"cmd": "gesture_set_message", "gesture_id": gesture.id, "message": message}
            )

    def _delete(self) -> None:
        gesture = self._selected_gesture()
        if gesture is None:
            return
        if messagebox.askyesno("Delete", f"Delete gesture {gesture.name!r}?", parent=self):
            self.app.send_command({"cmd": "gesture_delete", "gesture_id": gesture.id})

    def _undo(self) -> None:
        gesture = self._selected_gesture()
        if gesture is None:
            return
        self.app.send_command({"cmd": "gesture_undo_refinement", "gesture_id": gesture.id})

    def on_gestures_changed(self) -> None:
        self._refresh_list()

    def close(self) -> None:
        self.app.close_manage_view()
        self.destroy()
