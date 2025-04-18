"""
Live overlay of clickable elements with interactive terminal control,
including smooth one‑off scrolls, continuous auto‑scroll, and tab management.

    pip install playwright && playwright install
"""

from math import floor
import time, sys, threading, queue, re, textwrap
from playwright.sync_api import sync_playwright

URL              = "https://unify.ai"
MARGIN           = 100
ANIMATION_WAIT   = 2
REFRESH_INTERVAL = 0.5      # s

SCROLL_DURATION  = 400      # ms  (one‑off smooth scroll over this time)
AUTO_SCROLL_SPEED = 50 / SCROLL_DURATION   # px / ms  → 0.25 → 250 px / s

CLICKABLE_CSS = """
button:not([disabled]):visible,
input[type=button]:not([disabled]):visible,
input[type=submit]:not([disabled]):visible,
input[type=reset]:not([disabled]):visible,
[role=button]:not([disabled]):visible,
a[href]:visible,
[onclick]:visible
"""

# -----------------------------  JS snippets  --------------------------------

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
  if (!r.width || !r.height) return null;

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

# one‑off smooth scroll ------------------------------------------------------
HANDLE_SCROLL_JS = """
({delta, duration}) => {
  const startY   = window.scrollY;
  const targetY  = startY + delta;
  const startTs  = performance.now();

  function ease(p){ return p < .5 ? 2*p*p : -1 + (4 - 2*p)*p }

  function step(ts){
    const p = Math.min(1, (ts - startTs) / duration);
    window.scrollTo(0, startY + (targetY - startY) * ease(p));
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}
"""

# continuous auto‑scroll -----------------------------------------------------
AUTO_SCROLL_JS = """
({dir, speed}) => {
  // helper to stop any existing auto‑scroll
  if (!window.__pw_stopAutoScroll){
    window.__pw_stopAutoScroll = () => {
      if (window.__pw_autoScrollId){
        cancelAnimationFrame(window.__pw_autoScrollId);
        window.__pw_autoScrollId = null;
      }
    };
  }

  // always stop previous loop first
  window.__pw_stopAutoScroll();

  // if 'stop' requested, we’re done.
  if (dir === 'stop') return;

  const sign = dir === 'down' ? 1 : -1;
  let last   = performance.now();

  function step(ts){
    const dt = ts - last;
    last = ts;
    window.scrollBy(0, sign * speed * dt);
    window.__pw_autoScrollId = requestAnimationFrame(step);
  }
  window.__pw_autoScrollId = requestAnimationFrame(step);
}
"""

# -------------------------  Python helpers  ---------------------------------

def collect_elements(page):
    vp = page.evaluate("() => ({w:innerWidth, h:innerHeight})")
    vL, vT = -MARGIN, -MARGIN
    vR, vB = vp['w'] + MARGIN, vp['h'] + MARGIN

    elems = []
    for h in page.locator(CLICKABLE_CSS).element_handles():
        info = page.evaluate(ELEMENT_INFO_JS, h)
        if not info: continue
        vl, vt, w, hgt = info['vleft'], info['vtop'], info['width'], info['height']
        if (vl + w) < vL or vl > vR or (vt + hgt) < vT or vt > vB: continue
        info['handle'] = h
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
    return tuple((b['fixed'], b['x'], b['y'], b['w'], b['h']) for b in boxes)


def list_tabs(pages, active):
    res = []
    for p in pages:
        try:
            title = p.title()
        except Exception:
            title = "<unavailable>"
        marker = "*" if p is active else " "
        res.append(f"{marker} {title}")
    return res


def background_stdin(q):
    for line in sys.stdin:
        q.put(line.rstrip("\n"))


# -------------------------  Main  -------------------------------------------

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    pages       = [browser.new_page()]
    active_page = pages[0]
    active_page.goto(URL, wait_until="domcontentloaded")
    active_page.wait_for_timeout(ANIMATION_WAIT * 1000)

    cmd_q = queue.Queue()
    threading.Thread(target=background_stdin, args=(cmd_q,), daemon=True).start()

    last_snapshot = None

    HELP_TXT = textwrap.dedent(f"""
        Commands:
          <num>                     – click numbered element
          scroll up|down <px>       – smooth one‑off scroll
          start scroll up|down      – begin continuous auto‑scroll
          stop scroll               – halt auto‑scroll
          new tab                   – open about:blank
          close tab <text>          – close first tab whose title contains text
          switch to tab <text>      – activate matching tab
          q                         – quit
    """).strip()

    try:
        while True:
            if active_page:
                elements = collect_elements(active_page)
                boxes    = build_boxes(elements)
                snap     = serialise(boxes)
                if snap != last_snapshot:
                    print("\x1b[2J\x1b[H", end="")   # clear TTY
                    print("Open tabs:")
                    print("\n".join(list_tabs(pages, active_page)))
                    print("\n" + HELP_TXT + "\n")
                    for b, e in zip(boxes, elements):
                        print(f"{b['i']:>2}. {e['label']}")
                    active_page.evaluate(UPDATE_OVERLAY_JS, boxes)
                    last_snapshot = snap

            # process commands
            while not cmd_q.empty():
                raw  = cmd_q.get().strip()
                if not raw:
                    continue
                rlow = raw.lower()
                if rlow in {"q", "quit", "exit"}:
                    raise KeyboardInterrupt

                # numbered click
                if raw.isdigit() and active_page:
                    idx = int(raw)
                    if 1 <= idx <= len(elements):
                        try:
                            elements[idx-1]['handle'].click()
                            print(f"✓ Clicked element #{idx}")
                        except Exception as e:
                            print(f"! Click failed: {e}")
                    else:
                        print(f"! Index {idx} out of range")
                    continue

                # one‑off smooth scroll
                m = re.fullmatch(r"scroll\s+(up|down)\s+(\d+)", rlow)
                if m and active_page:
                    direction, px = m.group(1), int(m.group(2))
                    delta = -px if direction == "up" else px
                    active_page.evaluate(HANDLE_SCROLL_JS,
                                         {"delta": delta, "duration": SCROLL_DURATION})
                    last_snapshot = None
                    continue

                # start auto‑scroll
                if rlow in {"start scroll up", "start scroll down"} and active_page:
                    dir_ = "up" if "up" in rlow else "down"
                    active_page.evaluate(
                        AUTO_SCROLL_JS,
                        {"dir": dir_, "speed": AUTO_SCROLL_SPEED}
                    )
                    print(f"▶ Auto‑scroll {dir_} started")
                    last_snapshot = None
                    continue

                # stop auto‑scroll
                if rlow == "stop scroll" and active_page:
                    active_page.evaluate(AUTO_SCROLL_JS,
                                         {"dir": "stop", "speed": 0})
                    print("■ Auto‑scroll stopped")
                    last_snapshot = None
                    continue

                # new tab
                if rlow == "new tab":
                    newp = browser.new_page()
                    pages.append(newp)
                    active_page = newp
                    last_snapshot = None
                    continue

                # close tab
                m = re.fullmatch(r"close\s+tab\s+(.+)", rlow)
                if m:
                    txt = m.group(1).strip().lower()
                    tgt = next((pg for pg in pages if txt in pg.title().lower()), None)
                    if tgt:
                        tgt.close()
                        pages.remove(tgt)
                        active_page = pages[-1] if pages else None
                        last_snapshot = None
                    else:
                        print(f"! No tab matches '{txt}'")
                    continue

                # switch tab
                m = re.fullmatch(r"switch\s+to\s+tab\s+(.+)", rlow)
                if m:
                    txt = m.group(1).strip().lower()
                    tgt = next((pg for pg in pages if txt in pg.title().lower()), None)
                    if tgt:
                        active_page = tgt
                        active_page.bring_to_front()
                        last_snapshot = None
                    else:
                        print(f"! No tab matches '{txt}'")
                    continue

                print(f"! Unrecognised command: {raw}")

            time.sleep(REFRESH_INTERVAL)

    except KeyboardInterrupt:
        pass
    finally:
        browser.close()
