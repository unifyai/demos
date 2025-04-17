"""
Overlay numbered boxes on clickable elements that are inside (viewport ±MARGIN),
wait ANIMATION_WAIT s, and keep anything inside a fixed/sticky ancestor
pinned while normal content scrolls.

pip install playwright && playwright install
"""

from math import floor
from playwright.sync_api import sync_playwright

URL            = "https://unify.ai"
MARGIN         = 100   # px overscan
ANIMATION_WAIT = 2     # s

CLICKABLE_CSS = """
button:not([disabled]):visible,
input[type=button]:not([disabled]):visible,
input[type=submit]:not([disabled]):visible,
input[type=reset]:not([disabled]):visible,
[role=button]:not([disabled]):visible,
a[href]:visible,
[onclick]:visible
"""

# ---------------------------------------------------------------------------
#  NEW: fixed‑like detection walks up through ancestors
ELEMENT_INFO_JS = """
el => {
  function hasFixedAncestor(node){
    while (node && node !== document.body && node !== document.documentElement){
      const p = getComputedStyle(node).position;
      if (p === 'fixed' || p === 'sticky') return true;
      node = node.parentElement;
    }
    return false;
  }

  const r = el.getBoundingClientRect();
  if (!r.width || !r.height) return null;     // hidden / 0‑size

  const fixedLike = hasFixedAncestor(el);

  return {
    fixed : fixedLike,
    vleft : r.left,          // viewport coords
    vtop  : r.top,
    left  : r.left + scrollX,  // page coords
    top   : r.top  + scrollY,
    width : r.width,
    height: r.height,
    label : (el.innerText.trim() ||
             el.getAttribute('aria-label') ||
             el.getAttribute('alt')        ||
             el.getAttribute('title')      ||
             el.getAttribute('href')       ||
             '<no label>')
  };
}
"""

# ---------------------------------------------------------------------------

def collect_elements(page):
    vp = page.evaluate("""() => ({w:innerWidth,h:innerHeight})""")
    vL, vT = -MARGIN, -MARGIN
    vR, vB = vp["w"] + MARGIN, vp["h"] + MARGIN

    elems = []
    for h in page.locator(CLICKABLE_CSS).element_handles():
        info = page.evaluate(ELEMENT_INFO_JS, h)
        if not info:
            continue
        vl, vt, w, hgt = info["vleft"], info["vtop"], info["width"], info["height"]
        if (vl + w) < vL or vl > vR or (vt + hgt) < vT or vt > vB:
            continue
        elems.append(info)
    return elems

# ---------------------------------------------------------------------------

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page    = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded")
    page.wait_for_timeout(ANIMATION_WAIT * 1000)

    elements = collect_elements(page)

    # Console list + payload
    print()
    boxes = []
    for idx, e in enumerate(elements, 1):
        print(f"{idx}. {e['label']}")
        boxes.append(
            dict(i=idx,
                 fixed=e['fixed'],
                 px=floor(e['vleft']), py=floor(e['vtop']),
                 x =floor(e['left']),  y =floor(e['top']),
                 w =floor(e['width']), h =floor(e['height']))
        )

    # Inject overlay (two layers)
    page.evaluate(
        """boxes => {
          document.getElementById("__pw_rootFixed__")?.remove();
          document.getElementById("__pw_rootPage__") ?.remove();

          const rootPage  = Object.assign(document.createElement('div'),{
            id:"__pw_rootPage__",  style:"position:absolute;left:0;top:0;pointer-events:none;z-index:2147483646"});
          const rootFixed = Object.assign(document.createElement('div'),{
            id:"__pw_rootFixed__", style:"position:fixed;inset:0;pointer-events:none;z-index:2147483647"});
          document.body.append(rootPage, rootFixed);

          boxes.forEach(({i,fixed,x,y,px,py,w,h}, n,{length})=>{
            const hue = 360*n/length;
            const div = document.createElement('div');
            div.style.cssText = `
              position:${fixed?'fixed':'absolute'};
              left:${fixed?px:x}px; top:${fixed?py:y}px;
              width:${w}px; height:${h}px;
              outline:2px solid hsl(${hue} 100% 50%);
              background:hsl(${hue} 100% 50% / .12);
              font:700 12px/1 sans-serif;
              color:hsl(${hue} 100% 30%);
            `;
            const tag=document.createElement('span');
            tag.textContent=i;
            tag.style.cssText="position:absolute;left:0;top:0;background:#fff;padding:0 2px";
            div.append(tag);
            (fixed?rootFixed:rootPage).append(div);
          });
        }""",
        boxes,
    )

    print(f"\n👉  Overlaid {len(boxes)} clickable element(s).  Scroll to verify; Ctrl‑C to exit.")
    try:
        page.wait_for_timeout(1e9)
    except KeyboardInterrupt:
        pass
    breakpoint()
    browser.close()
