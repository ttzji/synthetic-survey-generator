"""
Visits the Streamlit app in a real (headless) browser, the way a person would.

Why: Streamlit Community Cloud puts apps to sleep after 12 hours without traffic, and its docs say the way to
keep an app awake is to visit it. A plain `curl` only downloads the page shell; it never runs the app in a
browser. This script loads the page, clicks the "get this app back up" button if the app is asleep, waits until
the app's own title is visible, and exits with an ERROR if that never happens -- so a broken keep-alive shows up
as a red run in the GitHub Actions tab instead of passing silently.

Settings (environment variables):
    APP_URL          required. The app's address, e.g. https://your-app.streamlit.app  (a repository secret)
    EXPECT_TEXT      text that must be visible once the app is up (default: the app's title)
    TOTAL_WAIT_SECONDS   how long to wait in total (default 300; a sleeping app can take a few minutes to start)
"""
import os
import re
import sys
import time

from playwright.sync_api import sync_playwright

URL = os.environ.get("APP_URL", "").strip()
EXPECT = os.environ.get("EXPECT_TEXT", "Synthetic Survey Data Generator")
TOTAL_WAIT = int(os.environ.get("TOTAL_WAIT_SECONDS", "300"))
WAKE_BUTTON = re.compile(r"get this app back up", re.I)

if not URL:
    sys.exit("APP_URL is empty. Add it in the repo: Settings > Secrets and variables > Actions > New repository secret.")
if not URL.startswith("http"):
    URL = "https://" + URL


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        print("Opening the app")
        response = page.goto(URL, wait_until="domcontentloaded", timeout=90_000)
        print(f"HTTP status: {response.status if response else 'unknown'}")      # the address is deliberately not printed: Actions logs are public in a public repo

        clicked = False
        deadline = time.time() + TOTAL_WAIT
        while time.time() < deadline:
            # Streamlit Cloud shows the app inside an iframe, so check every frame on the page.
            for frame in page.frames:
                try:
                    if frame.get_by_text(EXPECT, exact=False).count() > 0:
                        page.wait_for_timeout(5_000)      # let the app's live connection settle: this is the "traffic"
                        print(f"App is awake and rendered (found: {EXPECT!r}).")
                        browser.close()
                        return 0
                    button = frame.get_by_role("button", name=WAKE_BUTTON)
                    if button.count() == 0:
                        button = frame.get_by_text(WAKE_BUTTON)     # in case it is styled text, not a real <button>
                    if not clicked and button.count() > 0:
                        button.first.click()
                        clicked = True
                        print("App was asleep: clicked the wake-up button; waiting for it to start.")
                except Exception:
                    pass                                    # a frame that is mid-load; try again on the next pass
            time.sleep(3)

        print(f"ERROR: after {TOTAL_WAIT}s the app never showed {EXPECT!r}.")
        print("First 500 characters of the page text, to help diagnose:")
        try:
            print(page.inner_text("body")[:500])
        except Exception:
            pass
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
