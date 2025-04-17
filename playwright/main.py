"""
Overlay numbered boxes on clickable elements in/near the viewport.
The overlay is refreshed every REFRESH_INTERVAL seconds so that
scrolling, dynamic DOM changes,   etc. are reflected automatically.

    pip install playwright && playwright install
"""

from math import floor
import time
from playwright.sync_api import sync_playwright

URL              = "https://unify.ai"
MARGIN           = 100     # px overscan above / below viewport
ANIMATION_WAIT   = 2       # initial wait for in‑page animations (s)
REFRESH_INTERVAL = 0.5     # HOW OFTEN TO RE-COMPUTE (s)

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
el => {
  function hasFixedAncestor(node){
    while (node && node !== document.body && node !== document.documentElement){
      const pos = getComputedStyle(node).position;
      if (pos === 'fixed' || pos === 'sticky') return true;
      node = node.parentElement;
    }
    return false;
  }

  const r = el.getBoundingClientRect();
  if (!r.width || !r.height) return null;          // hidden / zero‑size

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

# One‑shot script that *keeps* the two root containers and just clears / repaints.
UPDATE_OVERLAY_JS = """
(boxes) => {
  let rootPage  = document.getElementById("__pw_rootPage__");
  let rootFixed = document.getElementById("__pw_rootFixed__");

  // First run – create roots
  if (!rootPage) {
    rootPage  = Object.assign(document.createElement('div'), {
      id:"__pw_rootPage__",  style:"position:absolute;left:0;top:0;pointer-events:none;z-index:2147483646"
    });
    rootFixed = Object.assign(document.createElement('div'), {
      id:"__pw_rootFixed__", style:"position:fixed;inset:0;pointer-events:none;z-index:2147483647"
    });
    document.body.append(rootPage, rootFixed);
  } else {
    // Subsequent runs – purge previous children
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
        elems.append(info)
    return elems


def build_boxes(elements):
    boxes = []
    for idx, e in enumerate(elements, 1):
        boxes.append(
            dict(i=idx,
                 fixed=e['fixed'],
                 px=floor(e['vleft']), py=floor(e['vtop']),
                 x =floor(e['left']),  y =floor(e['top']),
                 w =floor(e['width']), h =floor(e['height']))
        )
    return boxes


def serialise(boxes):
    """Lightweight tuple serialisation so we can detect changes quickly."""
    return tuple((b['fixed'], b['x'], b['y'], b['w'], b['h']) for b in boxes)

# ----------  Main  ----------------------------------------------------------

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page    = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded")
    page.wait_for_timeout(ANIMATION_WAIT * 1000)

    print("\n👉  Overlays are live.  Scroll the page; they’ll refresh automatically.  Ctrl‑C to exit.")

    last_snapshot = None
    try:
        while True:
            elements = collect_elements(page)
            boxes     = build_boxes(elements)
            snap      = serialise(boxes)

            if snap != last_snapshot:
                # Optional: console listing (comment out if noisy)
                print("\x1b[2J\x1b[H", end="")   # clear screen
                for b, e in zip(boxes, elements):
                    print(f"{b['i']:>2}. {e['label']}")
                page.evaluate(UPDATE_OVERLAY_JS, boxes)
                last_snapshot = snap

            time.sleep(REFRESH_INTERVAL)

    except KeyboardInterrupt:
        pass
    finally:
        browser.close()
