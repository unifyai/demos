"""
GUI helper for Playwright pages

 • Persistent Chromium context → “New tab” adds a real tab
 • Live list of clickable elements (double‑click to click)
 • Smooth one‑off scroll + continuous auto‑scroll
 • Command entry + quick buttons + scrolling log
"""

from math import floor
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext
import tempfile, shutil
import re, textwrap
from playwright.sync_api import sync_playwright

# ───────── configuration ─────────
URL               = "https://unify.ai"
MARGIN            = 100          # overscan px around viewport
ANIMATION_WAIT    = 2            # s to wait after page load
REFRESH_INTERVAL  = 0.5          # s GUI update cadence
SCROLL_DURATION   = 400          # ms for smooth scroll
AUTO_SCROLL_SPEED = 100/SCROLL_DURATION  # px / ms  ≈ 250 px / s

CLICKABLE_CSS = """
button:not([disabled]):visible,
input[type=button]:not([disabled]):visible,
input[type=submit]:not([disabled]):visible,
input[type=reset]:not([disabled]):visible,
[role=button]:not([disabled]):visible,
a[href]:visible,
[onclick]:visible
"""

# ───────── JavaScript snippets ─────────
ELEMENT_INFO_JS = """
(el) => {
  function hasFixedAncestor(n){
    while(n && n!==document.body && n!==document.documentElement){
      const p=getComputedStyle(n).position;
      if(p==='fixed'||p==='sticky') return true;
      n=n.parentElement;
    }
    return false;
  }
  const r=el.getBoundingClientRect();
  if(!r.width||!r.height) return null;
  return {
    fixed:hasFixedAncestor(el),
    hover:el.matches(':hover'),
    vleft:r.left, vtop:r.top,
    left:r.left+scrollX, top:r.top+scrollY,
    width:r.width, height:r.height,
    label:(el.innerText.trim()||
           el.getAttribute('aria-label')||
           el.getAttribute('alt')||
           el.getAttribute('title')||
           el.getAttribute('href')||
           '<no label>')
  };
}
"""

UPDATE_OVERLAY_JS = """
(boxes)=>{
  let rootPage=document.getElementById('__pw_rootPage__');
  let rootFixed=document.getElementById('__pw_rootFixed__');
  if(!rootPage){
    rootPage=Object.assign(document.createElement('div'),{
      id:'__pw_rootPage__',
      style:'position:absolute;left:0;top:0;pointer-events:none;z-index:2147483646'});
    rootFixed=Object.assign(document.createElement('div'),{
      id:'__pw_rootFixed__',
      style:'position:fixed;inset:0;pointer-events:none;z-index:2147483647'});
    document.body.append(rootPage,rootFixed);
  }else{rootPage.replaceChildren();rootFixed.replaceChildren();}
  boxes.forEach(({i,fixed,x,y,px,py,w,h},n,{length})=>{
    const hue=360*n/length;
    const div=document.createElement('div');
    div.style.cssText=`
      position:${fixed?'fixed':'absolute'};
      left:${fixed?px:x}px;top:${fixed?py:y}px;
      width:${w}px;height:${h}px;
      outline:2px solid hsl(${hue} 100% 50%);
      background:hsl(${hue} 100% 50%/.12);
      font:700 12px/1 sans-serif;
      color:hsl(${hue} 100% 30%);
    `;
    const tag=document.createElement('span');
    tag.textContent=i;
    tag.style.cssText='position:absolute;left:0;top:0;background:#fff;padding:0 2px';
    div.append(tag);
    (fixed?rootFixed:rootPage).append(div);
  });
}
"""

HANDLE_SCROLL_JS = """
({delta,duration})=>{
  const y0=scrollY, y1=y0+delta, t0=performance.now();
  const ease=p=>p<.5?2*p*p:-1+(4-2*p)*p;
  const step=ts=>{
    const p=Math.min(1,(ts-t0)/duration);
    scrollTo(0,y0+(y1-y0)*ease(p));
    if(p<1)requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
"""

AUTO_SCROLL_JS = """
({dir,speed})=>{
  if(!window.__as_stop){
    window.__as_stop=()=>{cancelAnimationFrame(window.__as_id);};
  }
  window.__as_stop();
  if(dir==='stop') return;
  const s=dir==='down'?1:-1; let last=performance.now();
  const step=ts=>{
    const dt=ts-last; last=ts;
    scrollBy(0,s*speed*dt);
    window.__as_id=requestAnimationFrame(step);
  };
  window.__as_id=requestAnimationFrame(step);
}
"""

# ───────── Python helpers ─────────
def collect_elements(page):
    vp = page.evaluate("()=>({w:innerWidth,h:innerHeight})")
    vL,vT = -MARGIN,-MARGIN; vR,vB = vp['w']+MARGIN, vp['h']+MARGIN
    elems=[]
    for h in page.locator(CLICKABLE_CSS).element_handles():
        info = page.evaluate(ELEMENT_INFO_JS,h)
        if not info: continue
        vl,vt,w,hgt = info['vleft'],info['vtop'],info['width'],info['height']
        if (vl+w)<vL or vl>vR or (vt+hgt)<vT or vt>vB: continue
        info['handle']=h; elems.append(info)
    return elems

def build_boxes(elements):
    return [dict(i=i+1,
                 fixed=e['fixed'],
                 px=floor(e['vleft']), py=floor(e['vtop']),
                 x=floor(e['left']), y=floor(e['top']),
                 w=floor(e['width']), h=floor(e['height']))
            for i,e in enumerate(elements)]

# ───────── Tkinter GUI class ─────────
class ControlPanel(tk.Tk):
    def __init__(self, ctx, first_page):
        super().__init__()
        self.ctx      = ctx
        self.active   = first_page
        self.elements = []

        self.title("Playwright helper")
        self.geometry("900x550")

        # layout grid
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        # --- element list --------------------------------------------------
        self.listbox=tk.Listbox(self,font=("Helvetica",11))
        self.listbox.grid(row=0,column=0,sticky="nsew")
        self.listbox.bind("<Double-1>", self.list_click)
        self.listbox.bind("<Return>",   self.list_click)
        sb=ttk.Scrollbar(self,orient="vertical",command=self.listbox.yview)
        sb.grid(row=0,column=0,sticky="nse")
        self.listbox.config(yscrollcommand=sb.set)

        # --- right panel (log + buttons) -----------------------------------
        right=tk.Frame(self); right.grid(row=0,column=1,sticky="nsew")
        right.rowconfigure(0,weight=3); right.rowconfigure(1,weight=1)
        right.columnconfigure(0,weight=1)

        # log
        self.log=scrolledtext.ScrolledText(right,state="disabled",height=8)
        self.log.grid(row=0,column=0,sticky="nsew",padx=5,pady=5)

        # buttons
        btns=tk.Frame(right); btns.grid(row=1,column=0,padx=5,pady=5,sticky="nsew")
        for i in range(2): btns.columnconfigure(i,weight=1)
        ttk.Button(btns,text="▲ Scroll 100",
                   command=lambda:self.run_cmd("scroll up 100")).grid(row=0,column=0,sticky="ew")
        ttk.Button(btns,text="▼ Scroll 100",
                   command=lambda:self.run_cmd("scroll down 100")).grid(row=0,column=1,sticky="ew")
        ttk.Button(btns,text="Start ▲",
                   command=lambda:self.run_cmd("start scroll up")).grid(row=1,column=0,sticky="ew")
        ttk.Button(btns,text="Start ▼",
                   command=lambda:self.run_cmd("start scroll down")).grid(row=1,column=1,sticky="ew")
        ttk.Button(btns,text="Stop scroll",
                   command=lambda:self.run_cmd("stop scroll")).grid(row=2,column=0,columnspan=2,sticky="ew")
        ttk.Button(btns,text="New tab",
                   command=lambda:self.run_cmd("new tab")).grid(row=3,column=0,sticky="ew")
        ttk.Button(btns,text="Close tab",
                   command=lambda:self.run_cmd(f"close tab {self.active.title()}")).grid(row=3,column=1,sticky="ew")

        # --- command bar ----------------------------------------------------
        bar=tk.Frame(self); bar.grid(row=1,column=0,columnspan=2,sticky="ew",padx=5,pady=4)
        bar.columnconfigure(1,weight=1)
        tk.Label(bar,text="Command:").grid(row=0,column=0,sticky="w")
        self.cmd_var=tk.StringVar()
        entry=tk.Entry(bar,textvariable=self.cmd_var)
        entry.grid(row=0,column=1,sticky="ew")
        entry.bind("<Return>",lambda e:self.send_cmd())
        ttk.Button(bar,text="Send",command=self.send_cmd).grid(row=0,column=2,sticky="e")

        # kickoff periodic GUI refresh
        self.after(100, self.refresh)

    # —— GUI mechanics —————————————————————————————————————————
    def log_msg(self,msg):
        self.log.configure(state="normal")
        self.log.insert("end",msg+"\n"); self.log.configure(state="disabled")
        self.log.yview_moveto(1.0)

    def list_click(self,_event):
        sel=self.listbox.curselection()
        if sel and sel[0]<len(self.elements):
            try:
                self.elements[sel[0]]['handle'].click()
            except Exception as e:
                self.log_msg(f"Click failed: {e}")

    def send_cmd(self):
        cmd=self.cmd_var.get().strip()
        if cmd: self.run_cmd(cmd)
        self.cmd_var.set("")

    # —— periodic refresh ————————————————————————————————
    def refresh(self):
        if not self.active: return
        elems=collect_elements(self.active)
        boxes=build_boxes(elems)
        self.listbox.delete(0,"end")
        for b,e in zip(boxes,elems):
            txt=f"{b['i']:>2}. {e['label']}" + (" (on hover)" if e['hover'] else "")
            self.listbox.insert("end",txt)
        self.active.evaluate(UPDATE_OVERLAY_JS,boxes)
        self.elements=elems
        self.after(int(REFRESH_INTERVAL*1000), self.refresh)

    # —— command dispatcher ——————————————————————————————
    def run_cmd(self,raw):
        self.log_msg(f"> {raw}")
        r=raw.lower()
        # smooth scroll
        m=re.fullmatch(r"scroll\s+(up|down)\s+(\d+)",r)
        if m:
            delta=(-1 if m.group(1)=="up" else 1)*int(m.group(2))
            self.active.evaluate(HANDLE_SCROLL_JS,{"delta":delta,"duration":SCROLL_DURATION})
            return
        # auto scroll
        if r in {"start scroll up","start scroll down"}:
            dir_="up" if "up" in r else "down"
            self.active.evaluate(AUTO_SCROLL_JS,{"dir":dir_,"speed":AUTO_SCROLL_SPEED})
            return
        if r=="stop scroll":
            self.active.evaluate(AUTO_SCROLL_JS,{"dir":"stop","speed":0}); return
        # new tab
        if r=="new tab":
            newp=self.ctx.new_page()
            self.active=newp
            self.log_msg("Opened new tab"); return
        # close tab
        m=re.fullmatch(r"close\s+tab\s+(.+)",r)
        if m:
            txt=m.group(1).strip().lower()
            tgt=next((pg for pg in self.ctx.pages if txt in (pg.title() or "").lower()),None)
            if tgt:
                tgt.close(); self.log_msg("Tab closed")
                self.active=self.ctx.pages[0] if self.ctx.pages else None
            else: self.log_msg("No tab matches")
            return
        # switch tab
        m=re.fullmatch(r"switch\s+to\s+tab\s+(.+)",r)
        if m:
            txt=m.group(1).strip().lower()
            tgt=next((pg for pg in self.ctx.pages if txt in (pg.title() or "").lower()),None)
            if tgt:
                self.active=tgt; tgt.bring_to_front(); self.log_msg("Switched tab")
            else: self.log_msg("No tab matches")
            return
        self.log_msg("Unrecognised command")

# ───────── main bootstrap ─────────
with sync_playwright() as p:
    profile_dir = Path(tempfile.mkdtemp(prefix="pw_profile_"))
    ctx = p.chromium.launch_persistent_context(profile_dir, headless=False)
    # ensure at least one page
    if not ctx.pages: first_page = ctx.new_page()
    else: first_page = ctx.pages[0]
    first_page.goto(URL, wait_until="domcontentloaded")
    first_page.wait_for_timeout(ANIMATION_WAIT*1000)

    gui = ControlPanel(ctx, first_page)
    try:
        gui.mainloop()
    finally:
        ctx.close()
        shutil.rmtree(profile_dir, ignore_errors=True)
