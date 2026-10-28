"""Export manuscript figures at 1200 dpi and at least 8000 x 3000 pixels.

The one-point font increase and tick/layout handling follow the supplied
LaTeX package. This helper writes the PNG only; it creates no audit records.
"""
from pathlib import Path
import math
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator
from matplotlib.transforms import Bbox

MINIMUM_PIXELS = (8000, 3000)
PNG_DPI = 1200
FONT_INCREMENT_PT = 1.0
# PNG stores integer pixels per metre; round upward so its stored density
# reaches the nominal 1200 dpi rather than falling just below it.
_SAVE_DPI = math.ceil(PNG_DPI / 0.0254) * 0.0254


def export_figure(fig, stem, padding=0.08):
    stem = Path(stem)
    fig.canvas.draw()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator(axis.get_majorticklocs()))
            axis.set_minor_locator(FixedLocator(axis.get_minorticklocs()))
    for text in fig.findobj(match=Text):
        text.set_fontsize(float(text.get_fontsize()) + FONT_INCREMENT_PT)
    if stem.name == "Pic4":
        for ax in fig.axes[:1]:
            ax.set_xticklabels([
                label.get_text().replace(" ", "\n", 1)
                for label in ax.get_xticklabels()
            ])
    fig.canvas.draw()
    fig.set_layout_engine("none")
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer()).padded(padding)
    width = max(box.width, (MINIMUM_PIXELS[0] + 2) / _SAVE_DPI)
    height = max(box.height, (MINIMUM_PIXELS[1] + 2) / _SAVE_DPI)
    cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
    box = Bbox.from_bounds(cx - width / 2, cy - height / 2, width, height)
    fig.savefig(stem.with_suffix(".png"), format="png", dpi=_SAVE_DPI,
                bbox_inches=box, pad_inches=0)
