#!/usr/bin/env python3
"""Render figure pages (fig_*.html) to vector PDF (for LaTeX) and PNG (for the README) with headless Chromium.

  python3 render.py [fig_name ...]        -> out/<name>.pdf, out/<name>.png

Each page draws into <div id="fig"> and sets window.__READY__ = true when done. The PDF page is cropped to the
figure's bounding box, so \\includegraphics[width=\\linewidth] keeps the type at its designed size when the figure
was designed 624 px wide.
"""
import glob
import os
import sys

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
names = sys.argv[1:] or [os.path.basename(p)[:-5] for p in sorted(glob.glob(os.path.join(HERE, "fig_*.html")))]

with sync_playwright() as p:
    browser = p.chromium.launch()
    for name in names:
        page = browser.new_page(device_scale_factor=3)
        page.goto("file://" + os.path.join(HERE, name + ".html"))
        page.wait_for_function("window.__READY__ === true", timeout=30000)
        page.evaluate("document.fonts.ready")
        box = page.eval_on_selector("#fig", "e => { const r = e.getBoundingClientRect(); return [r.width, r.height]; }")
        w, h = int(round(box[0])), int(round(box[1]))
        page.set_viewport_size({"width": w, "height": h})
        page.pdf(path=os.path.join(OUT, name + ".pdf"), width=f"{w}px", height=f"{h + 1}px", print_background=True,
                 margin=dict(top="0", right="0", bottom="0", left="0"), page_ranges="1")
        page.locator("#fig").screenshot(path=os.path.join(OUT, name + ".png"))
        print(f"{name}: {w}x{h}px -> out/{name}.pdf, .png")
        page.close()
    browser.close()
