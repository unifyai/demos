"""
Playwright helper with a friendlier Tkinter GUI.

 • Live list of clickable elements (double‑click to click in page)
 • Smooth one‑off scroll, continuous auto‑scroll
 • Tab management
 • Text‑command entry + quick‑action buttons
"""

from math import floor
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading, queue, re, textwrap, sys
from playwright.sync_api import sync_playwright

# ────────────── tunables ────────────────────────────────────────────────────
URL               = "https://unify.ai"
MARGIN            = 100
ANIMATION_WAIT    = 2
REFRESH_INTERVAL  = 0.5      # s
SCROLL_DURATION   = 400      # ms
AUTO_SCROLL_SPEED = 100 / SCROLL_DURATION   # px / ms  ≈ 250 px / s

# ────────────── playwright element CSS ──────────────────────────────────────
CLICKABLE_CSS = """
button:not([disabled]):visible,
input[type=button]:not([disabled]):visible,
input[type=submit]:not([disabled]):visible,
input[type=reset]:not([disabled]):visible,
[role=button]:not([disabled]):visible,
a[href]:visible,
[onclick]:visible
"""

# ────────────── JavaScript helpers ──────────────────────────────────────────
ELEMENT_INFO_JS = """
(el) => {
  function hasFixedAncestor(node){
    while (node && node !== document.body && node !== document.documentElement){
      const p = getComputedStyle(node).position;
      if (p === 'fixed' || p === 'sticky') return true;
      node = node.parentElement;
    }
    return false;
  }
  const r = el.getBoundingClientRect();
  if (!r.width || !r.height) return null;
  return {
    fixed  : hasFixedAncestor(el),
    hover  : el.matches(':hover'),
    vleft  : r.left,
    vtop   : r.top,
    left   : r.left + scrollX,
    top    : r.top  + scrollY,
    width  : r.width,
    height : r.height,
    label  : (el.innerText.trim()           ||
              el.getAttribute('aria-label') ||
              el.getAttribute('alt')        ||
              el.getAttribute('title')      ||
              el.getAttribute('href')       ||
              '<no label>')
  };
}
"""

UPDATE_OVERLAY_JS = """
(boxes) => {
  let rootPage  = document.getElementById("__pw_rootPage__");
  let rootFixed = document.getElementById("__pw_rootFixed__");
  if (!rootPage){
    rootPage  = Object.assign(document.createElement('div'),{
      id:"__pw_rootPage__",  style:"position:absolute;left:0;top:0;pointer-events:none;z-index:2147483646"});
    rootFixed = Object.assign(document.createElement('div'),{
      id:"__pw_rootFixed__", style:"position:fixed;inset:0;pointer-events:none;z-index:2147483647"});
    document.body.append(rootPage, rootFixed);
  } else {
    rootPage.replaceChildren(); rootFixed.replaceChildren();
  }
  boxes.forEach(({i,fixed,x,y,px,py,w,h}, n,{length})=>{
    const hue = 360*n/length;
    const div = document.createElement('div');
    div.style.cssText = `
      position:${fixed?'fixed':'absolute'};
      left:${fixed?px:x}px; top:${fixed?py:y}px;
      width:${w}px; height:${h}px;
      outline:2px solid hsl(${h} 100% 50%);
      background:hsl(${h} 100% 50% / .12);
      font:700 12px/1 sans-serif;
      color:hsl(${h} 100% 30%);
    `;
    const tag = document.createElement('span');
    tag.textContent = i;
    tag.style.cssText = "position:absolute;left:0;top:0;background:#fff;padding:0 2px";
    div.append(tag);
    (fixed?rootFixed:rootPage).append(div);
  });
}
"""

HANDLE_SCROLL_JS = """
({delta, duration}) => {
  const startY   = window.scrollY;
  const targetY  = startY + delta;
  const startTs  = performance.now();
  const ease = p => p<.5 ? 2*p*p : -1+(4-2*p)*p;
  const step = ts=>{
    const p = Math.min(1,(ts-startTs)/duration);
    window.scrollTo(0,startY+(targetY-startY)*ease(p));
    if(p<1)requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
"""

AUTO_SCROLL_JS = """
({dir,speed})=>{
  if(!window.__pw_stop){window.__pw_stop=()=>{cancelAnimationFrame(window.__pw_id);}}
  window.__pw_stop();
  if(dir==='stop')return;
  const sign=dir==='down'?1:-1;let last=performance.now();
  const step=ts=>{
    const dt=ts-last;last=ts;
    window.scrollBy(0,sign*speed*dt);
    window.__pw_id=requestAnimationFrame(step);
  };
  window.__pw_id=requestAnimationFrame(step);
}
"""

# ────────────── helpers ─────────────────────────────────────────────────────
def collect_elements(page):
    vp = page.evaluate("()=>({w:innerWidth,h:innerHeight})")
    vL,vT = -MARGIN,-MARGIN ; vR,vB = vp['w']+MARGIN, vp['h']+MARGIN
    elems=[]
    for h in page.locator(CLICKABLE_CSS).element_handles():
        info = page.evaluate(ELEMENT_INFO_JS,h)
        if not info: continue
        vl,vt,w,hgt = info['vleft'],info['vtop'],info['width'],info['height']
        if (vl+w)<vL or vl>vR or (vt+hgt)<vT or vt>vB: continue
        info['handle']=h; elems.append(info)
    return elems

def build_boxes(elements):
    from math import floor
    return [dict(i=i+1,fixed=e['fixed'],px=floor(e['vleft']),py=floor(e['vtop']),
                 x=floor(e['left']),y=floor(e['top']),
                 w=floor(e['width']),h=floor(e['height']))
            for i,e in enumerate(elements)]

# ────────────── GUI application class ───────────────────────────────────────
class ControlPanel(tk.Tk):
    def __init__(self, browser, page):
        super().__init__()
        self.browser, self.page = browser, page
        self.pages=[page]; self.active=page
        self.elements=[]
        self.title("Playwright helper")
        self.geometry("800x500")

        # layout
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        # listbox
        self.listbox=tk.Listbox(self,font=("Helvetica",11))
        self.listbox.grid(row=0,column=0,sticky="nsew")
        self.listbox.bind("<Double-1>", self.on_list_click)
        self.listbox.bind("<Return>",   self.on_list_click)
        scr=ttk.Scrollbar(self,orient="vertical",command=self.listbox.yview)
        scr.grid(row=0,column=0,sticky="nse")
        self.listbox.config(yscrollcommand=scr.set)

        # right side: log + buttons
        right=tk.Frame(self); right.grid(row=0,column=1,sticky="nsew")
        right.rowconfigure(0,weight=3); right.rowconfigure(1,weight=1)
        right.columnconfigure(0,weight=1)

        self.log=scrolledtext.ScrolledText(right,height=8,state="disabled")
        self.log.grid(row=0,column=0,sticky="nsew",padx=5,pady=5)

        btns=tk.Frame(right); btns.grid(row=1,column=0,padx=5,pady=5,sticky="nsew")
        for i in range(2): btns.columnconfigure(i,weight=1)
        ttk.Button(btns,text="▲ Scroll 100",command=lambda:self.command("scroll up 100")).grid(row=0,column=0,sticky="ew")
        ttk.Button(btns,text="▼ Scroll 100",command=lambda:self.command("scroll down 100")).grid(row=0,column=1,sticky="ew")
        ttk.Button(btns,text="Start ▲",command=lambda:self.command("start scroll up")).grid(row=1,column=0,sticky="ew")
        ttk.Button(btns,text="Start ▼",command=lambda:self.command("start scroll down")).grid(row=1,column=1,sticky="ew")
        ttk.Button(btns,text="Stop scroll",command=lambda:self.command("stop scroll")).grid(row=2,column=0,columnspan=2,sticky="ew")
        ttk.Button(btns,text="New tab",command=lambda:self.command("new tab")).grid(row=3,column=0,sticky="ew")
        ttk.Button(btns,text="Close tab",command=lambda:self.command(f"close tab {self.active.title()}")).grid(row=3,column=1,sticky="ew")

        # command entry
        bar=tk.Frame(self); bar.grid(row=1,column=0,columnspan=2,sticky="ew",padx=5,pady=3)
        bar.columnconfigure(0,weight=1)
        tk.Label(bar,text="Command:").grid(row=0,column=0,sticky="w")
        self.cmd_var=tk.StringVar()
        cmd_entry=tk.Entry(bar,textvariable=self.cmd_var)
        cmd_entry.grid(row=0,column=1,sticky="ew")
        cmd_entry.bind("<Return>", lambda e:self.send_command())
        ttk.Button(bar,text="Send",command=self.send_command).grid(row=0,column=2,sticky="e")

        # kick off periodic updates
        self.after(100, self.refresh)

    # ── GUI helpers ────────────────────────────────────────────────────────
    def log_msg(self,msg):
        self.log.configure(state="normal"); self.log.insert("end",msg+"\n"); self.log.configure(state="disabled")
        self.log.yview_moveto(1.0)

    def send_command(self):
        cmd=self.cmd_var.get().strip()
        if cmd: self.command(cmd)
        self.cmd_var.set("")

    def on_list_click(self,_):
        sel=self.listbox.curselection()
        if sel and sel[0]<len(self.elements):
            try:
                self.elements[sel[0]]['handle'].click()
            except Exception as e:
                self.log_msg(f"Click failed: {e}")

    # ── periodic update ────────────────────────────────────────────────────
    def refresh(self):
        elems=collect_elements(self.active) if self.active else []
        boxes=build_boxes(elems)
        self.listbox.delete(0,"end")
        for b,e in zip(boxes,elems):
            txt=f"{b['i']:>2}. {e['label']}" + (" (on hover)" if e['hover'] else "")
            self.listbox.insert("end",txt)
        self.active.evaluate(UPDATE_OVERLAY_JS,boxes)
        self.elements=elems
        self.after(int(REFRESH_INTERVAL*1000),self.refresh)

    # ── command dispatcher ────────────────────────────────────────────────
    def command(self,raw):
        self.log_msg(f"> {raw}")
        rlow=raw.lower()
        # smooth scroll
        m=re.fullmatch(r"scroll\s+(up|down)\s+(\d+)",rlow)
        if m and self.active:
            delta=(-1 if m.group(1)=="up" else 1)*int(m.group(2))
            self.active.evaluate(HANDLE_SCROLL_JS,{"delta":delta,"duration":SCROLL_DURATION})
            return
        # auto‑scroll
        if rlow in {"start scroll up","start scroll down"} and self.active:
            dir_="up" if "up" in rlow else "down"
            self.active.evaluate(AUTO_SCROLL_JS,{"dir":dir_,"speed":AUTO_SCROLL_SPEED})
            return
        if rlow=="stop scroll" and self.active:
            self.active.evaluate(AUTO_SCROLL_JS,{"dir":"stop","speed":0}); return
        # tab ops
        if rlow=="new tab":
            newp=self.browser.new_page(); self.pages.append(newp); self.active=newp
            self.log_msg(f"New tab: {newp.title() or '<blank>'}"); return
        m=re.fullmatch(r"close\s+tab\s+(.+)",rlow)
        if m:
            txt=m.group(1).strip().lower()
            tgt=next((pg for pg in self.pages if txt in pg.title().lower()),None)
            if tgt:
                tgt.close(); self.pages.remove(tgt)
                self.active=self.pages[-1] if self.pages else None
                self.log_msg("Tab closed"); return
            self.log_msg("No tab matches"); return
        m=re.fullmatch(r"switch\s+to\s+tab\s+(.+)",rlow)
        if m:
            txt=m.group(1).strip().lower()
            tgt=next((pg for pg in self.pages if txt in pg.title().lower()),None)
            if tgt:
                self.active=tgt; tgt.bring_to_front(); self.log_msg(f"Switched to {tgt.title()}"); return
            self.log_msg("No tab matches"); return
        self.log_msg("Unrecognised command")

# ────────────── boot everything ─────────────────────────────────────────────
with sync_playwright() as p:
    browser=p.chromium.launch(headless=False)
    page=browser.new_page(); page.goto(URL,wait_until="domcontentloaded")
    page.wait_for_timeout(ANIMATION_WAIT*1000)
    app=ControlPanel(browser,page)
    try: app.mainloop()
    finally: browser.close()
