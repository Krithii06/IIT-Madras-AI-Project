"""Capture the application screenshots used in the report.

Drives a real browser against the locally running stack, so the images in the
report show the actual application responding to a real prediction rather than a
mock-up. Start both servers first:

    cd backend && uvicorn app.main:app --port 8000
    cd frontend && npm run dev

    python -m src.report.screenshots

Uses the Chrome already installed on the machine rather than downloading a browser.
"""

import argparse

from playwright.sync_api import sync_playwright

from src import config

FRONTEND = "http://localhost:5173"
BACKEND = "http://localhost:8000"


def pick_sample():
    """A visibly diseased leaf makes a more useful screenshot than a random one."""
    for cls in ("Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___Apple_scab"):
        folder = config.RAW_DIR / cls
        if folder.exists():
            files = sorted(folder.glob("*.JPG"))
            if files:
                return files[0]
    raise SystemExit("no dataset images found; run src.data.prepare first")


def capture(frontend=FRONTEND, backend=BACKEND, width=1180, height=820):
    # full_page grows to fit the content but never shrinks below the viewport, so the
    # viewport is kept short to avoid a band of empty page under each screenshot.
    out = config.FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    sample = pick_sample()

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": width, "height": height},
                                device_scale_factor=2)

        # Viewport-scoped, not full_page: the page runs to about 4,000px once the
        # reference sections are included, which shrinks to an illegible strip when
        # fitted to a report page.
        page.goto(frontend, wait_until="networkidle")
        # The model panel only renders once /model-info has come back.
        page.wait_for_selector("text=Architecture", timeout=30_000)
        page.screenshot(path=str(out / "screenshot_upload.png"))
        print("wrote screenshot_upload.png")

        page.set_input_files("input[type=file]", str(sample))
        page.wait_for_selector(".preview img", timeout=15_000)
        page.screenshot(path=str(out / "screenshot_preview.png"))
        print("wrote screenshot_preview.png")

        page.click("button.primary")
        page.wait_for_selector(".result-label", timeout=60_000)
        page.wait_for_timeout(400)
        # Element-scoped rather than viewport: the page is long enough now that a
        # viewport shot of the result would be mostly the usage guide above it.
        # ":has(.result-label)" picks the result card without matching the guide,
        # which also contains the word "result".
        page.locator(".card:has(.result-label)").screenshot(
            path=str(out / "screenshot_result.png"))
        print("wrote screenshot_result.png")

        # Crop to the reference sections rather than another full page, so the
        # report shows them at readable size.
        page.locator(".card", has_text="Tools and techniques").screenshot(
            path=str(out / "screenshot_about.png"))
        print("wrote screenshot_about.png")

        # The generated OpenAPI page is the clearest evidence the API is real.
        api = browser.new_page(viewport={"width": width, "height": 1000},
                               device_scale_factor=2)
        api.goto(f"{backend}/docs", wait_until="networkidle")
        api.wait_for_selector(".opblock", timeout=30_000)
        api.screenshot(path=str(out / "screenshot_api.png"), full_page=True)
        print("wrote screenshot_api.png")

        # A phone viewport, to check the layout actually reflows rather than
        # just trusting the media query.
        mobile = browser.new_page(viewport={"width": 390, "height": 780},
                                  device_scale_factor=2, is_mobile=True,
                                  has_touch=True)
        mobile.goto(frontend, wait_until="networkidle")
        mobile.wait_for_selector("text=Architecture", timeout=30_000)
        # Viewport only, not full_page: the whole page is over 6,000px tall on a
        # phone, which is unusable as a figure.
        mobile.screenshot(path=str(out / "screenshot_mobile.png"))
        print("wrote screenshot_mobile.png")

        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend", default=FRONTEND)
    parser.add_argument("--backend", default=BACKEND)
    args = parser.parse_args()
    capture(args.frontend, args.backend)
