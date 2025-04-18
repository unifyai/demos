"""
Entry‑point.  Launches a persistent Chromium context, opens the GUI.
"""

import shutil
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

from browser_utils import launch_persistent
from commands import CommandRunner
from gui import ControlPanel


START_URL = "https://unify.ai"
ANIMATION_WAIT = 2  # seconds


def main() -> None:
    with sync_playwright() as pw:
        profile_dir = Path(tempfile.mkdtemp(prefix="pw_profile_"))
        ctx = launch_persistent(pw)

        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(START_URL, wait_until="domcontentloaded")
        page.wait_for_timeout(ANIMATION_WAIT * 1000)

        gui = ControlPanel(CommandRunner(ctx, log_fn=lambda m: gui.log_msg(m)))
        try:
            gui.mainloop()
        finally:
            ctx.close()
            shutil.rmtree(profile_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
