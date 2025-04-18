"""
Tk GUI (now queue‑driven).  No direct Playwright calls: all heavy work
runs in BrowserWorker.
"""

from __future__ import annotations

import queue
import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import Any

# ------------------------------------------------------------------- GUI
class ControlPanel(tk.Tk):
    REFRESH_INTERVAL_MS = 100  # how often we poll the update queue

    def __init__(
        self,
        command_q: "queue.Queue[str]",
        update_q: "queue.Queue[list[tuple[int,str,bool]]]",
    ):
        super().__init__()
        self.cmd_q = command_q
        self.up_q = update_q
        self.elements: list[tuple[int, str, bool]] = []  # (idx, label, hover)
        self._build_widgets()
        self.after(self.REFRESH_INTERVAL_MS, self._poll_updates)

    # ----------------------------- UI -----------------------------------
    def _build_widgets(self) -> None:
        self.title("Playwright helper")
        self.geometry("900x550")
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        # listbox
        self.listbox = tk.Listbox(self, font=("Helvetica", 11))
        self.listbox.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(self, orient="vertical", command=self.listbox.yview)
        sb.grid(row=0, column=0, sticky="nse")
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.bind("<Double-1>", self._on_list_click)
        self.listbox.bind("<Return>", self._on_list_click)

        # right panel
        right = tk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=1)

        self.log = scrolledtext.ScrolledText(right, state="disabled", height=8)
        self.log.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        btns = tk.Frame(right)
        btns.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        for i in range(2):
            btns.columnconfigure(i, weight=1)

        def make(r: int, c: int, txt: str, cmd: str) -> None:
            ttk.Button(btns, text=txt, command=lambda: self._send(cmd)).grid(
                row=r, column=c, sticky="ew"
            )

        make(0, 0, "▲ Scroll 100", "scroll up 100")
        make(0, 1, "▼ Scroll 100", "scroll down 100")
        make(1, 0, "Start ▲", "start scroll up")
        make(1, 1, "Start ▼", "start scroll down")
        make(2, 0, "Stop scroll", "stop scroll")
        make(3, 0, "New tab", "new tab")
        make(3, 1, "Close tab", "close tab")  # closes active

        # command bar
        bar = tk.Frame(self)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=4)
        bar.columnconfigure(1, weight=1)
        tk.Label(bar, text="Command:").grid(row=0, column=0, sticky="w")
        self.cmd_var = tk.StringVar()
        entry = tk.Entry(bar, textvariable=self.cmd_var)
        entry.grid(row=0, column=1, sticky="ew")
        entry.bind("<Return>", lambda e: self._send_from_entry())
        ttk.Button(bar, text="Send", command=self._send_from_entry).grid(row=0, column=2)

    # ----------------------------- events --------------------------------
    def _on_list_click(self, _event: Any) -> None:
        sel = self.listbox.curselection()
        if sel:
            idx = sel[0] + 1  # 1‑based to match worker
            self._send(f"click {idx}")

    def _send_from_entry(self) -> None:
        text = self.cmd_var.get().strip()
        self.cmd_var.set("")
        if text:
            self._send(text)

    def _send(self, text: str) -> None:
        try:
            self.cmd_q.put_nowait(text)
            self._log(f"> {text}")
        except queue.Full:
            self._log("Command queue full – try again")

    # ----------------------- update polling ------------------------------
    def _poll_updates(self) -> None:
        updated = False
        while True:
            try:
                self.elements = self.up_q.get_nowait()
                updated = True
            except queue.Empty:
                break
        if updated:
            self.listbox.delete(0, "end")
            for idx, label, hover in self.elements:
                self.listbox.insert(
                    "end", f"{idx:>2}. {label}" + (" (on hover)" if hover else "")
                )
        self.after(self.REFRESH_INTERVAL_MS, self._poll_updates)

    # ----------------------------- logging -------------------------------
    def _log(self, msg: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.configure(state="disabled")
        self.log.yview_moveto(1.0)
