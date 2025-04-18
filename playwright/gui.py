"""
Tk‑based front‑end.

• All Playwright work is done in BrowserWorker (background thread)
• This file now accepts *arbitrary English* in the command bar, sends it to
  o3‑mini (OpenAI) via `agent.parse_instruction`, converts the structured
  result into the low‑level command strings understood by BrowserWorker,
  and shows everything in the log window.
"""

from __future__ import annotations

import queue
import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import Any

from agent import Action, parse_instruction


class ControlPanel(tk.Tk):
    """Main GUI window.  No Playwright calls occur on this thread."""

    REFRESH_INTERVAL_MS = 100  # how often we poll the update queue

    def __init__(
        self,
        command_q: "queue.Queue[str]",
        update_q: "queue.Queue[list[tuple[int, str, bool]]]",
    ):
        super().__init__()
        self.cmd_q = command_q        # GUI → worker
        self.up_q = update_q          # worker → GUI
        self.elements: list[tuple[int, str, bool]] = []

        self._build_widgets()
        self.after(self.REFRESH_INTERVAL_MS, self._poll_updates)

    # ------------------------------------------------------------------ UI
    def _build_widgets(self) -> None:
        self.title("Playwright helper")
        self.geometry("900x550")

        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        # -------- element list ------------------------------------------
        self.listbox = tk.Listbox(self, font=("Helvetica", 11))
        self.listbox.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(self, orient="vertical", command=self.listbox.yview)
        sb.grid(row=0, column=0, sticky="nse")
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.bind("<Double-1>", self._on_list_click)
        self.listbox.bind("<Return>", self._on_list_click)

        # -------- right panel (log + buttons) ---------------------------
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
            ttk.Button(
                btns, text=txt, command=lambda: self._handle_input(cmd)
            ).grid(row=r, column=c, sticky="ew")

        make(0, 0, "▲ Scroll 100", "scroll up 100")
        make(0, 1, "▼ Scroll 100", "scroll down 100")
        make(1, 0, "Start ▲", "start scroll up")
        make(1, 1, "Start ▼", "start scroll down")
        make(2, 0, "Stop scroll", "stop scroll")
        make(3, 0, "New tab", "new tab")
        make(3, 1, "Close tab", "close tab")

        # -------- command bar ------------------------------------------
        bar = tk.Frame(self)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=4)
        bar.columnconfigure(1, weight=1)
        tk.Label(bar, text="Command:").grid(row=0, column=0, sticky="w")
        self.cmd_var = tk.StringVar()
        entry = tk.Entry(bar, textvariable=self.cmd_var)
        entry.grid(row=0, column=1, sticky="ew")
        entry.bind("<Return>", lambda _e: self._send_from_entry())
        ttk.Button(bar, text="Send", command=self._send_from_entry).grid(
            row=0, column=2
        )

    # ---------------------------------------------------------------- events
    def _on_list_click(self, _e: Any) -> None:
        sel = self.listbox.curselection()
        if sel:
            idx = sel[0] + 1  # listbox is 0‑based, worker expects 1‑based
            self._queue_command(f"click {idx}")

    def _send_from_entry(self) -> None:
        text = self.cmd_var.get().strip()
        self.cmd_var.set("")
        if text:
            self._handle_input(text)

    # ---------------------------------------------------------------- logic
    def _handle_input(self, text: str) -> None:
        """
        Decide whether `text` is a low‑level command or English, then act.
        """
        self._log(f"> {text}")
        low = text.lower()
        if low.startswith(
            ("scroll", "start scroll", "stop scroll", "new tab", "close tab", "switch")
        ):
            self._queue_command(text)
            return

        # free English → LLM --------------------------------------------
        action = parse_instruction(text)
        if not action:
            self._log("❗ Could not interpret instruction")
            return

        cmd = self._action_to_runner_cmd(action)
        if cmd:
            self._log(f"  ↳ {cmd}")
            self._queue_command(cmd)
        else:
            self._log("❗ Could not map instruction to command")

    # convert structured Action to low‑level string
    @staticmethod
    def _action_to_runner_cmd(act: Action) -> str | None:
        match act.action:
            case "click_button" if act.button_text:
                return f"click button {act.button_text}"
            case "scroll" if act.direction and act.pixels:
                return f"scroll {act.direction} {act.pixels}"
            case "start_scroll" if act.direction:
                return f"start scroll {act.direction}"
            case "stop_scroll":
                return "stop scroll"
            case "new_tab":
                return "new tab"
            case "close_tab":
                return (
                    f"close tab {act.tab_text}"
                    if act.tab_text
                    else "close tab"
                )
            case "switch_tab" if act.tab_text:
                return f"switch to tab {act.tab_text}"
        return None

    # put message onto command queue
    def _queue_command(self, cmd: str) -> None:
        try:
            self.cmd_q.put_nowait(cmd)
        except queue.Full:
            self._log("⚠ command queue full – retry shortly")

    # ---------------------------------------------------------------- queue polling
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

    # ---------------------------------------------------------------- logging
    def _log(self, msg: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.configure(state="disabled")
        self.log.yview_moveto(1.0)
