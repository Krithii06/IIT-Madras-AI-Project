"""Copy the figures the LaTeX report uses into docs/report/latex/figures.

Only the five figures the document actually includes are copied, and oversized ones
are downscaled - the raw sample grids are ~1.7 MB each, which makes an Overleaf
project slow to sync for no visible gain at print size.

    python -m src.report.latex_assets
"""

import shutil

from PIL import Image

from src import config

LATEX_DIR = config.PROJECT_ROOT / "docs" / "report" / "latex"
FIG_DIR = LATEX_DIR / "figures"

# source name -> (max width in pixels or None, output format)
#
# Grids of leaf photographs go to JPEG: re-encoding them as PNG is larger than the
# original, because PNG is the wrong codec for photographic content. Charts and UI
# screenshots stay PNG, where flat colour and text edges compress well and JPEG
# would add ringing around the type.
# The confusion matrix is deliberately absent: at two classes it is four numbers,
# and the report prints it as a table, which is more precise and costs no page space.
#
# The class distribution chart is gone too: the selected-subset table lists the same
# four counts exactly, and the workflow diagram earns that page space instead.
WANTED = {
    "leaf_groups.png": (1400, "jpg"),
    "mobilenet_leaf_test_errors.png": (1400, "jpg"),
    "screenshot_upload.png": (1500, "png"),
    "screenshot_result.png": (None, "png"),
}


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, (max_width, fmt) in WANTED.items():
        source = config.FIGURES_DIR / name
        if not source.exists():
            print(f"  missing (skipped): {name}")
            continue

        target = FIG_DIR / (name if fmt == "png" else name.replace(".png", ".jpg"))
        if max_width is None and fmt == "png":
            shutil.copy(source, target)
        else:
            with Image.open(source) as img:
                img = img.convert("RGB") if fmt == "jpg" else img
                if max_width and img.width > max_width:
                    height = round(img.height * max_width / img.width)
                    img = img.resize((max_width, height), Image.LANCZOS)
                if fmt == "jpg":
                    img.save(target, quality=88, optimize=True, progressive=True)
                else:
                    img.save(target, optimize=True)

        size_kb = target.stat().st_size / 1024
        total += size_kb
        print(f"  {target.name:<44} {size_kb:>7.0f} KB")

    print(f"\n{FIG_DIR}  ({total / 1024:.1f} MB total)")


if __name__ == "__main__":
    main()
