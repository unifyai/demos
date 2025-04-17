"""
Live overlay of clickable elements with interactive terminal control.

 • The overlay and the numbered list refresh every REFRESH_INTERVAL seconds,
   so scrolling or DOM changes are reflected automatically.
 • In the terminal just type the desired number and press <Enter>;
   the element is clicked in the browser.
 • Type q (or quit / exit) to close everything.

    pip install playwright && playwright install
"""

from math import floor
import time
import sys
import threading
import queue
from playwright.sync_api import sync_playwright

URL              = "https://unify.ai"
MARGIN           = 100     # px overscan above / below viewport
ANIMATION_WAIT   = 2       # initial wait for in‑page animations (s)
REFRESH_INTERVAL = 0.5     # overlay & list refresh cadence (s)

CLICKABLE_CSS = """
button:not([disabled]):visible,
input[type=button]:not([disabled]):visible,
input[type=submit]:not([disabled]):visible,
input[type=reset]:not([disabled]):visible,
[role=button]:not([disabled]):visible,
a[href]:visible,
[onclick]:visible
"""

# ----------  JS helpers  ----------------------------------------------------

ELEMENT_INFO_JS = """
(el) => {
  function hasFixedAncestor(node){
    while (node && node !== document.body && node !== document.documentElement){
      const pos = getComputedStyle(node).position;
      if (pos === 'fixed' || pos === 'sticky') return true;
      node = node.parentElement;
    }
    return false;
  }

  const r = el.getBoundingClientRect();
  if (!r.width || !r.height) return null;     // hidden / zero‑size

  return {
    fixed : hasFixedAncestor(el),
    vleft : r.left,
    vtop  : r.top,
    left  : r.left + scrollX,
    top   : r.top  + scrollY,
    width : r.width,
    height: r.height,
    label : (el.innerText.trim()           ||
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

  if (!rootPage) {
    rootPage  = Object.assign(document.createElement('div'), {
      id:"__pw_rootPage__",  style:"position:absolute;left:0;top:0;pointer-events:none;z-index:2147483646"
    });
    rootFixed = Object.assign(document.createElement('div'), {
      id:"__pw_rootFixed__", style:"position:fixed;inset:0;pointer-events:none;z-index:2147483647"
    });
    document.body.append(rootPage, rootFixed);
  } else {
    rootPage.replaceChildren();
    rootFixed.replaceChildren();
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

# ----------  Python helpers  -------------------------------------------------

def collect_elements(page):
    """Return a list of dicts each containing element handle + meta info."""
    vp = page.evaluate("() => ({w:innerWidth, h:innerHeight})")
    vL, vT = -MARGIN, -MARGIN
    vR, vB = vp["w"] + MARGIN, vp["h"] + MARGIN

    elems = []
    for handle in page.locator(CLICKABLE_CSS).element_handles():
        info = page.evaluate(ELEMENT_INFO_JS, handle)
        if not info:
            continue
        vl, vt, w, hgt = info["vleft"], info["vtop"], info["width"], info["height"]
        if (vl + w) < vL or vl > vR or (vt + hgt) < vT or vt > vB:
            continue
        info["handle"] = handle            # keep handle for later click
        elems.append(info)
    return elems


def build_boxes(elements):
    return [
        dict(i=idx,
             fixed=e['fixed'],
             px=floor(e['vleft']), py=floor(e['vtop']),
             x =floor(e['left']),  y =floor(e['top']),
             w =floor(e['width']), h =floor(e['height']))
        for idx, e in enumerate(elements, 1)
    ]


def serialise(boxes):
    """Tiny serialisation to detect visual changes quickly."""
    return tuple((b['fixed'], b['x'], b['y'], b['w'], b['h']) for b in boxes)


# ----------  Input thread  ---------------------------------------------------

def start_input_listener(q):
    """Background thread reading from stdin and pushing complete lines to q."""
    def _reader():
        for line in sys.stdin:
            q.put(line.strip())
    t = threading.Thread(target=_reader, daemon=True)
    t.start()


# ----------  Main  -----------------------------------------------------------

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page    = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded")
    page.wait_for_timeout(ANIMATION_WAIT * 1000)

    print("\nScroll the page freely. Numbers update live.")
    print("Type a number + <Enter> to click it, or q to quit.\n")

    last_snapshot = None
    input_q = queue.Queue()
    start_input_listener(input_q)

    try:
        while True:
            elements = collect_elements(page)
            boxes    = build_boxes(elements)
            snap     = serialise(boxes)

            if snap != last_snapshot:
                # print live list (no screen clearing to avoid fighting with user typing)
                print("\n" + "-"*40)
                for b, e in zip(boxes, elements):
                    print(f"{b['i']:>2}. {e['label']}")
                print("-"*40)
                page.evaluate(UPDATE_OVERLAY_JS, boxes)
                last_snapshot = snap

            # handle any user inputs
            while not input_q.empty():
                cmd = input_q.get()
                if cmd.lower() in {"q", "quit", "exit"}:
                    raise KeyboardInterrupt
                if not cmd.isdigit():
                    print(f"! Not a number: {cmd}")
                    continue
                idx = int(cmd)
                if 1 <= idx <= len(elements):
                    try:
                        elements[idx - 1]["handle"].click()
                        print(f"✓ Clicked element #{idx}\n")
                    except Exception as e:
                        print(f"! Click failed: {e}\n")
                else:
                    print(f"! Index {idx} out of range\n")

            time.sleep(REFRESH_INTERVAL)

    except KeyboardInterrupt:
        pass
    finally:
        browser.close()
