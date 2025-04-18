"""
Tk‑based UI that talks to CommandRunner & browser utilities.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext

from commands import CommandRunner


class ControlPanel(tk.Tk):
    REFRESH_INTERVAL_MS = 500

    def __init__(self, runner: CommandRunner):
        super().__init__()
        self.runner = runner
        self.elements: list[dict] = []
        self._build_widgets()
        self.after(100, self._periodic_refresh)

    # ---------- UI --------------------------------------------------------
    def _build_widgets(self):
        self.title("Playwright helper")
        self.geometry("900x550")
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        # element list
        self.listbox = tk.Listbox(self, font=("Helvetica", 11))
        self.listbox.grid(row=0, column=0, sticky="nsew")
        self.listbox.bind("<Double-1>", self._on_list_click)
        self.listbox.bind("<Return>", self._on_list_click)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.listbox.yview)
        sb.grid(row=0, column=0, sticky="nse")
        self.listbox.config(yscrollcommand=sb.set)

        # right panel
        right = tk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=1)

        # log
        self.log = scrolledtext.ScrolledText(right, state="disabled", height=8)
        self.log.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # buttons
        btns = tk.Frame(right)
        btns.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        for i in range(2):
            btns.columnconfigure(i, weight=1)

        make_btn = lambda r, c, txt, cmd: ttk.Button(
            btns, text=txt, command=lambda: self.runner.run(cmd)
        ).grid(row=r, column=c, sticky="ew")

        make_btn(0, 0, "▲ Scroll 100", "scroll up 100")
        make_btn(0, 1, "▼ Scroll 100", "scroll down 100")
        make_btn(1, 0, "Start ▲", "start scroll up")
        make_btn(1, 1, "Start ▼", "start scroll down")
        make_btn(2, 0, "Stop scroll", "stop scroll")
        make_btn(3, 0, "New tab", "new tab")
        make_btn(3, 1, "Close tab", "close tab")  # closes active

        # command bar
        bar = tk.Frame(self)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=4)
        bar.columnconfigure(1, weight=1)
        tk.Label(bar, text="Command:").grid(row=0, column=0, sticky="w")
        self.cmd_var = tk.StringVar()
        entry = tk.Entry(bar, textvariable=self.cmd_var)
        entry.grid(row=0, column=1, sticky="ew")
        entry.bind("<Return>", lambda e: self._send_cmd())
        ttk.Button(bar, text="Send", command=self._send_cmd).grid(row=0, column=2)

    # ---------- logging ---------------------------------------------------
    def log_msg(self, msg: str):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.configure(state="disabled")
        self.log.yview_moveto(1.0)

    # ---------- events ----------------------------------------------------
    def _on_list_click(self, _event):
        sel = self.listbox.curselection()
        if sel and sel[0] < len(self.elements):
            try:
                self.elements[sel[0]]["handle"].click()
            except Exception as e:
                self.log_msg(f"Click failed: {e}")

    def _send_cmd(self):
        cmd = self.cmd_var.get().strip()
        self.cmd_var.set("")
        if cmd:
            self.log_msg("> " + cmd)
            self.runner.run(cmd)

    # ---------- periodic overlay / list refresh --------------------------
    def _periodic_refresh(self):
        self.elements = self.runner.refresh_overlay()
        self.listbox.delete(0, "end")
        for idx, e in enumerate(self.elements, 1):
            text = f"{idx:>2}. {e['label']}" + (" (on hover)" if e["hover"] else "")
            self.listbox.insert("end", text)
        self.after(self.REFRESH_INTERVAL_MS, self._periodic_refresh)
