"""
Playwright helper GUI   –   persistent‑context tabs, live element overlay,
smooth & auto scroll, command entry, quick‑action buttons, scrolling log.

> pip install playwright && playwright install
"""

from math import floor
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext
import tempfile, shutil, re
from playwright.sync_api import sync_playwright

# ───────── configuration ─────────
URL               = "https://unify.ai"
MARGIN            = 100
ANIMATION_WAIT    = 2
REFRESH_INTERVAL  = 0.5
SCROLL_DURATION   = 400
AUTO_SCROLL_SPEED = 100 / SCROLL_DURATION

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
  function hasFixed(n){
    while(n&&n!==document.body&&n!==document.documentElement){
      const p=getComputedStyle(n).position;
      if(p==='fixed'||p==='sticky') return true;
      n=n.parentElement;
    }
    return false;
  }
  const r=el.getBoundingClientRect();
  if(!r.width||!r.height) return null;
  return {
    fixed:hasFixed(el),
    hover:el.matches(':hover'),
    vleft:r.left,vtop:r.top,
    left:r.left+scrollX, top:r.top+scrollY,
    width:r.width,height:r.height,
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
(b)=>{
 let rp=document.getElementById('__pw_rootPage__');
 let rf=document.getElementById('__pw_rootFixed__');
 if(!rp){
   rp=Object.assign(document.createElement('div'),{
     id:'__pw_rootPage__',style:'position:absolute;left:0;top:0;pointer-events:none;z-index:2147483646'});
   rf=Object.assign(document.createElement('div'),{
     id:'__pw_rootFixed__',style:'position:fixed;inset:0;pointer-events:none;z-index:2147483647'});
   document.body.append(rp,rf);
 }else{rp.replaceChildren();rf.replaceChildren();}
 b.forEach(({i,fixed,x,y,px,py,w,h},n,{length:l})=>{
   const hue=360*n/l;
   const d=document.createElement('div');
   d.style.cssText=`
     position:${fixed?'fixed':'absolute'};
     left:${fixed?px:x}px;top:${fixed?py:y}px;
     width:${w}px;height:${h}px;
     outline:2px solid hsl(${hue} 100% 50%);
     background:hsl(${hue} 100% 50%/.12);
     font:700 12px/1 sans-serif;
     color:hsl(${hue} 100% 30%);
   `;
   const s=document.createElement('span');
   s.textContent=i;
   s.style.cssText='position:absolute;left:0;top:0;background:#fff;padding:0 2px';
   d.append(s);
   (fixed?rf:rp).append(d);
 });
}
"""

HANDLE_SCROLL_JS = """
({delta,duration})=>{
  const y0=scrollY,y1=y0+delta,t0=performance.now();
  const ease=p=>p<.5?2*p*p:-1+(4-2*p)*p;
  const step=t=>{
    const p=Math.min(1,(t-t0)/duration);
    scrollTo(0,y0+(y1-y0)*ease(p));
    if(p<1)requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
"""

AUTO_SCROLL_JS = """
({dir,speed})=>{
 if(!window.__asStop){window.__asStop=()=>cancelAnimationFrame(window.__asId);}
 window.__asStop();
 if(dir==='stop')return;
 const s=dir==='down'?1:-1;let last=performance.now();
 const step=t=>{
   const dt=t-last;last=t;scrollBy(0,s*speed*dt);
   window.__asId=requestAnimationFrame(step);
 };
 window.__asId=requestAnimationFrame(step);
}
"""

# ───────── Python helpers ─────────
def collect_elements(page):
    vp=page.evaluate("()=>({w:innerWidth,h:innerHeight})")
    vL,vT=-MARGIN,-MARGIN; vR,vB=vp['w']+MARGIN, vp['h']+MARGIN
    elems=[]
    for h in page.locator(CLICKABLE_CSS).element_handles():
        info=page.evaluate(ELEMENT_INFO_JS,h)
        if not info: continue
        vl,vt,w,hgt=info['vleft'],info['vtop'],info['width'],info['height']
        if(vl+w)<vL or vl>vR or(vt+hgt)<vT or vt>vB: continue
        info['handle']=h; elems.append(info)
    return elems

def build_boxes(elems):
    from math import floor
    return [dict(i=i+1,fixed=e['fixed'],
                 px=floor(e['vleft']),py=floor(e['vtop']),
                 x=floor(e['left']), y=floor(e['top']),
                 w=floor(e['width']),h=floor(e['height']))
            for i,e in enumerate(elems)]

# ───────── GUI class ─────────
class Panel(tk.Tk):
    def __init__(self, ctx, page):
        super().__init__()
        self.ctx=ctx
        self.active=page
        self.elements=[]
        self.init_ui()
        self.after(100,self.refresh)

    def init_ui(self):
        self.title("Playwright helper")
        self.geometry("900x550")
        self.columnconfigure(0,weight=3)
        self.columnconfigure(1,weight=2)
        self.rowconfigure(0,weight=1)
        self.rowconfigure(1,weight=0)

        self.list=tk.Listbox(self,font=("Helvetica",11))
        self.list.grid(row=0,column=0,sticky="nsew")
        sb=ttk.Scrollbar(self,orient="vertical",command=self.list.yview)
        sb.grid(row=0,column=0,sticky="nse")
        self.list.config(yscrollcommand=sb.set)
        self.list.bind("<Double-1>",self.list_click)
        self.list.bind("<Return>",self.list_click)

        right=tk.Frame(self); right.grid(row=0,column=1,sticky="nsew")
        right.rowconfigure(0,weight=3); right.rowconfigure(1,weight=1)
        right.columnconfigure(0,weight=1)
        self.log=scrolledtext.ScrolledText(right,state="disabled",height=8)
        self.log.grid(row=0,column=0,sticky="nsew",padx=5,pady=5)
        btns=tk.Frame(right); btns.grid(row=1,column=0,sticky="nsew",padx=5,pady=5)
        for i in range(2): btns.columnconfigure(i,weight=1)
        ttk.Button(btns,text="▲ Scroll 100",command=lambda:self.cmd("scroll up 100")).grid(row=0,column=0,sticky="ew")
        ttk.Button(btns,text="▼ Scroll 100",command=lambda:self.cmd("scroll down 100")).grid(row=0,column=1,sticky="ew")
        ttk.Button(btns,text="Start ▲",command=lambda:self.cmd("start scroll up")).grid(row=1,column=0,sticky="ew")
        ttk.Button(btns,text="Start ▼",command=lambda:self.cmd("start scroll down")).grid(row=1,column=1,sticky="ew")
        ttk.Button(btns,text="Stop scroll",command=lambda:self.cmd("stop scroll")).grid(row=2,column=0,columnspan=2,sticky="ew")
        ttk.Button(btns,text="New tab",command=lambda:self.cmd("new tab")).grid(row=3,column=0,sticky="ew")
        ttk.Button(btns,text="Close tab",command=lambda:self.cmd("close tab")).grid(row=3,column=1,sticky="ew")  # ★ simpler

        bar=tk.Frame(self); bar.grid(row=1,column=0,columnspan=2,sticky="ew",padx=5,pady=4)
        bar.columnconfigure(1,weight=1)
        tk.Label(bar,text="Command:").grid(row=0,column=0,sticky="w")
        self.cmd_var=tk.StringVar()
        ent=tk.Entry(bar,textvariable=self.cmd_var)
        ent.grid(row=0,column=1,sticky="ew")
        ent.bind("<Return>",lambda e:self.send_cmd())
        ttk.Button(bar,text="Send",command=self.send_cmd).grid(row=0,column=2)

    # ── util helpers
    def log_msg(self,msg):
        self.log.configure(state="normal"); self.log.insert("end",msg+"\n")
        self.log.configure(state="disabled"); self.log.yview_moveto(1)

    def list_click(self,_):
        sel=self.list.curselection()
        if sel and sel[0]<len(self.elements):
            try:self.elements[sel[0]]['handle'].click()
            except Exception as e:self.log_msg(f"Click failed: {e}")

    def send_cmd(self):
        c=self.cmd_var.get().strip(); self.cmd_var.set("")
        if c: self.cmd(c)

    # ── periodic update
    def refresh(self):
        elems=collect_elements(self.active)
        boxes=build_boxes(elems)
        self.list.delete(0,"end")
        for b,e in zip(boxes,elems):
            self.list.insert("end",f"{b['i']:>2}. {e['label']}" + (" (on hover)" if e['hover'] else ""))
        self.active.evaluate(UPDATE_OVERLAY_JS,boxes)
        self.elements=elems
        self.after(int(REFRESH_INTERVAL*1000),self.refresh)

    # ── command dispatcher
    def cmd(self,raw):
        self.log_msg("> "+raw)
        r=raw.lower().strip()
        if not r: return
        # smooth scroll
        m=re.fullmatch(r"scroll\s+(up|down)\s+(\d+)",r)
        if m:
            delta=(-1 if m.group(1)=="up" else 1)*int(m.group(2))
            self.active.evaluate(HANDLE_SCROLL_JS,{"delta":delta,"duration":SCROLL_DURATION}); return
        # auto scroll
        if r in {"start scroll up","start scroll down"}:
            self.active.evaluate(AUTO_SCROLL_JS,{"dir":"up" if "up" in r else "down","speed":AUTO_SCROLL_SPEED}); return
        if r=="stop scroll": self.active.evaluate(AUTO_SCROLL_JS,{"dir":"stop","speed":0}); return
        # new tab
        if r=="new tab":
            self.active=self.ctx.new_page()
            self.log_msg("Opened new tab"); return
        # close tab  (title optional)                 ★ regex updated
        m=re.fullmatch(r"close\s+tab(?:\s+(.+))?",r)
        if m:
            txt=(m.group(1) or "").strip().lower()
            if txt:
                tgt=next((pg for pg in self.ctx.pages if txt in (pg.title() or "").lower()),None)
            else:
                tgt=self.active
            if tgt:
                tgt.close(); self.log_msg("Tab closed")
                if self.ctx.pages: self.active=self.ctx.pages[0]
                else: self.active=None
            else:self.log_msg("No tab matches")
            return
        # switch tab
        m=re.fullmatch(r"switch\s+to\s+tab\s+(.+)",r)
        if m:
            txt=m.group(1).strip().lower()
            tgt=next((pg for pg in self.ctx.pages if txt in (pg.title() or "").lower()),None)
            if tgt: self.active=tgt; tgt.bring_to_front(); self.log_msg("Switched tab")
            else: self.log_msg("No tab matches")
            return
        self.log_msg("Unrecognised command")

# ───────── bootstrap ─────────
with sync_playwright() as p:
    profile = Path(tempfile.mkdtemp(prefix="pw_profile_"))
    ctx = p.chromium.launch_persistent_context(profile, headless=False)
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto(URL, wait_until="domcontentloaded")
    page.wait_for_timeout(ANIMATION_WAIT*1000)
    gui=Panel(ctx,page)
    try: gui.mainloop()
    finally:
        ctx.close(); shutil.rmtree(profile, ignore_errors=True)
