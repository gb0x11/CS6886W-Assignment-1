"""Export the panels of the W&B report (Question 3) as PNG images for the PDF.

The report is rendered by W&B itself in a headless Chromium; each panel is
saved as report/figures/wandb/<panel>.png, which report/report.tex picks up
automatically instead of the local matplotlib renderings.

    pip install playwright && python -m playwright install chromium
    python scripts/wandb_screenshots.py --report_url <W&B report URL>

Authentication uses the API key that `wandb login` stored in ~/.netrc (it is
sent only to the W&B server and never printed).
"""
import argparse
import base64
import netrc
import os
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

REPORT = ("https://forge.coreweave.com/wandb/cs25m508-indian-institute-of-technology/"
          "cs6886w-vgg6-cifar10/reports/CS6886W-A1:-VGG6-on-CIFAR-10-(Question-3-plots)"
          "--VmlldzoxODE3NjQ4Mw==")

# Panel title in the W&B report -> output file name used by report.tex
NAMES = {"Configuration vs. validation accuracy": "parallel_coordinates",
         "val_acc v. created": "valacc_scatter",
         "train_acc": "train_acc", "train_loss": "train_loss",
         "val_acc": "val_acc", "val_loss": "val_loss"}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report_url", default=REPORT)
    p.add_argument("--api_host", default="api.forge.coreweave.com",
                   help="host of the W&B API entry in ~/.netrc (api.wandb.ai for wandb.ai)")
    p.add_argument("--out_dir", default="report/figures/wandb")
    p.add_argument("--wait_ms", type=int, default=6000, help="render time per panel")
    args = p.parse_args()

    _, _, key = netrc.netrc().authenticators(args.api_host)
    auth = "Basic " + base64.b64encode(f"api:{key}".encode()).decode()
    app = urlparse(args.report_url)

    def add_auth(route, request):
        headers = dict(request.headers)
        headers["authorization"] = auth
        route.continue_(headers=headers)

    os.makedirs(args.out_dir, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1100}, device_scale_factor=2)
        # The W&B web app talks to <app>/api/...; authenticate those calls with the API key.
        page.route(f"{app.scheme}://{app.netloc}/api/**", add_auth)
        page.goto(args.report_url, wait_until="domcontentloaded", timeout=90000)
        page.wait_for_timeout(15000)
        # Drop the cookie banner so it does not cover the panels.
        page.evaluate("""() => { for (const e of document.querySelectorAll('div,section,aside')) {
            if (getComputedStyle(e).position === 'fixed' && e.textContent.includes('We use cookies'))
                e.remove(); } }""")
        panels = page.locator("div.panel-bank__panel")
        for i in range(panels.count()):
            panel = panels.nth(i)
            panel.scroll_into_view_if_needed()  # panels render lazily
            page.wait_for_timeout(args.wait_ms)
            page.mouse.move(5, 5)  # keep hover controls out of the image
            titles = panel.locator(".PanelTitle")
            title = titles.first.inner_text().strip() if titles.count() else ""
            name = NAMES.get(title, "param_importance" if "importance" in panel.inner_text().lower()
                             else f"panel{i}")
            panel.screenshot(path=os.path.join(args.out_dir, f"{name}.png"))
            print(f"{title or name!r} -> {name}.png")
        browser.close()


if __name__ == "__main__":
    main()
