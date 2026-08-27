"""Sanity-check the LaTeX report without a TeX installation.

There is no LaTeX toolchain on the machine this was written on, so the document
cannot be compiled here. This does the checks that catch the mistakes which would
otherwise only surface on Overleaf, plus the one that matters most: that the numbers
written into the prose still match what the pipeline actually measured.

    python -m src.report.check_latex
"""

import json
import re
import sys

from src import config

TEX = config.PROJECT_ROOT / "docs" / "report" / "latex" / "main.tex"
FIG_DIR = TEX.parent / "figures"


def strip_comments(text):
    # Remove % comments but keep escaped \%
    return re.sub(r"(?<!\\)%.*", "", text)


def check_environments(body, problems):
    opened = re.findall(r"\\begin\{([^}]+)\}", body)
    closed = re.findall(r"\\end\{([^}]+)\}", body)
    for name in set(opened) | set(closed):
        if opened.count(name) != closed.count(name):
            problems.append(
                f"environment '{name}' unbalanced: "
                f"{opened.count(name)} begin vs {closed.count(name)} end")


def check_braces(body, problems):
    depth = 0
    for i, ch in enumerate(body):
        if ch in "{}" and (i == 0 or body[i - 1] != "\\"):
            depth += 1 if ch == "{" else -1
            if depth < 0:
                problems.append(f"unmatched closing brace near offset {i}")
                return
    if depth != 0:
        problems.append(f"unbalanced braces: depth {depth} at end of file")


A4_W_CM, A4_H_CM = 21.0, 29.7

# Rough prose density by base font size, after allowing for headings and the
# inter-paragraph skip. Used only for the page estimate.
WORDS_PER_PAGE = {10: 650, 11: 560, 12: 480}


def page_geometry(body):
    """Read the margin and font size out of the document instead of assuming them."""
    margin = re.search(r"margin\s*=\s*([\d.]+)\s*cm", body)
    margin_cm = float(margin.group(1)) if margin else 2.5

    size = re.search(r"\\documentclass\[([^\]]*)\]", body)
    pt = 10
    if size:
        found = re.search(r"(\d+)pt", size.group(1))
        if found:
            pt = int(found.group(1))

    return (A4_W_CM - 2 * margin_cm, A4_H_CM - 2 * margin_cm,
            WORDS_PER_PAGE.get(pt, 560), pt, margin_cm)


def check_figures(body, problems, text_w, text_h):
    """Check each figure exists, and estimate the page space it will occupy."""
    entries = re.findall(r"\\includegraphics(?:\[([^\]]*)\])?\{([^}]+)\}", body)
    used, page_fraction = [], 0.0

    for options, name in entries:
        used.append(name)
        path = FIG_DIR / name
        if not path.exists():
            problems.append(f"missing figure file: figures/{name}")
            continue

        match = re.search(r"width\s*=\s*([\d.]+)\s*\\textwidth", options or "")
        fraction = float(match.group(1)) if match else 1.0
        try:
            from PIL import Image

            with Image.open(path) as img:
                aspect = img.height / img.width
        except Exception:
            aspect = 0.6
        rendered_cm = fraction * text_w * aspect
        page_fraction += rendered_cm / text_h

    present = {p.name for p in FIG_DIR.iterdir()} if FIG_DIR.exists() else set()
    for extra in sorted(present - set(used)):
        problems.append(f"note: figures/{extra} is present but never included")
    return used, page_fraction


# Arguments that are never typeset, so an underscore inside them is harmless:
# file paths, URLs and cross-reference keys.
_LITERAL_ARGS = re.compile(
    r"\\(?:includegraphics(?:\[[^\]]*\])?|graphicspath|url|href|label|ref|cite|bibitem)"
    r"(?:\{[^{}]*\})+"
)


def check_raw_underscores(body, problems):
    """An unescaped _ outside maths is the classic Overleaf compile failure."""
    scanned = _LITERAL_ARGS.sub(" ", body)
    scanned = re.sub(r"\$[^$]*\$", " ", scanned)
    for match in re.finditer(r"(?<!\\)_", scanned):
        start = max(0, match.start() - 45)
        problems.append(
            f"unescaped underscore: ...{scanned[start:match.start() + 15]!r}")


def check_numbers(body, problems):
    """Cross-check quoted results against results/metrics."""
    metrics_path = config.METRICS_DIR / "mobilenet_leaf_test.json"
    errors_path = config.METRICS_DIR / "mobilenet_leaf_test_error_analysis.json"
    leak_path = config.METRICS_DIR / "leakage_analysis.json"
    if not metrics_path.exists():
        problems.append("no test metrics to check the prose against")
        return

    m = json.load(open(metrics_path, encoding="utf-8"))
    e = json.load(open(errors_path, encoding="utf-8")) if errors_path.exists() else {}
    leak = json.load(open(leak_path, encoding="utf-8")) if leak_path.exists() else {}

    expected = {
        "accuracy": f"{m['accuracy']:.4f}",
        "macro F1": f"{m['macro_f1']:.4f}",
        "accuracy as a percentage": f"{m['accuracy'] * 100:.2f}",
        "test image count": f"{m['images']}",
        "test leaf count": f"{m['leaves']}",
        "precision (diseased)": f"{m['precision_positive']:.4f}",
        "recall (diseased)": f"{m['recall_positive']:.4f}",
    }
    if e:
        expected["image errors"] = str(e["image_errors"])
        expected["leaf-level accuracy"] = f"{e['leaf_level_accuracy']:.4f}"
        expected["mean confidence when wrong"] = f"{e['mean_confidence_when_wrong']:.4f}"

    flat = " ".join(body.split())
    for label, value in expected.items():
        if value not in flat:
            problems.append(f"{label} '{value}' from metrics does not appear in the report")

    if leak:
        naive = leak["random_image_split"]["val_vs_train"]
        grouped = leak["leaf_grouped_split"]["val_vs_train"]
        for label, block in (("naive", naive), ("grouped", grouped)):
            pct = f"{block['leaked_fraction'] * 100:.1f}"
            if pct not in flat:
                problems.append(f"{label}-split leak share '{pct}' does not appear")

    cm = m["confusion_matrix"]
    for value in (cm[0][0], cm[1][0], cm[1][1]):
        if str(value) not in flat:
            problems.append(f"confusion matrix entry {value} does not appear")


def main():
    if not TEX.exists():
        sys.exit(f"not found: {TEX}")

    raw = TEX.read_text(encoding="utf-8")
    body = strip_comments(raw)
    problems = []

    text_w, text_h, words_per_page, pt, margin_cm = page_geometry(body)

    check_environments(body, problems)
    check_braces(body, problems)
    used, figure_pages = check_figures(body, problems, text_w, text_h)
    check_raw_underscores(body, problems)
    check_numbers(body, problems)

    if "\\end{document}" not in body:
        problems.append("no \\end{document}")

    # Every \ref needs its \label, or the PDF ships with "??" in the text.
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    for target in set(re.findall(r"\\ref\{([^}]+)\}", body)):
        if target not in labels:
            problems.append(f"\\ref{{{target}}} has no matching \\label")

    for line_no, line in enumerate(body.splitlines(), 1):
        non_ascii = [c for c in line if ord(c) > 127]
        if non_ascii:
            problems.append(
                f"line {line_no}: non-ASCII {non_ascii!r} - fine under modern pdfLaTeX "
                f"but safer replaced with a LaTeX escape")

    # Diagram bodies are markup, not prose; strip them before counting words.
    prose = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", body,
                   flags=re.DOTALL)
    diagrams = len(re.findall(r"\\begin\{tikzpicture\}", body))
    words = len(re.findall(r"[A-Za-z']+", re.sub(r"\\[a-zA-Z]+", " ", prose)))

    text_pages = words / words_per_page
    diagram_pages = diagrams * 0.24
    total = text_pages + figure_pages + diagram_pages

    print(f"{TEX.relative_to(config.PROJECT_ROOT)}")
    print(f"  layout             : {pt}pt, {margin_cm}cm margins, "
          f"{text_w:.1f}x{text_h:.1f}cm text block")
    print(f"  words (approx)     : {words}")
    print(f"  figures included   : {len(used)}  ({figure_pages:.2f} pages of space)")
    print(f"  tikz diagrams      : {diagrams}  ({diagram_pages:.2f} pages)")
    print(f"  estimated pages    : {total:.1f}   "
          f"(text {text_pages:.1f} + art {figure_pages + diagram_pages:.1f})")
    if total > 7.6:
        problems.append(f"note: estimated {total:.1f} pages, over the 6-7 page target")

    hard = [p for p in problems if not p.startswith("note:")]
    notes = [p for p in problems if p.startswith("note:")]
    for note in notes:
        print(f"  {note}")
    if hard:
        print(f"\n{len(hard)} problem(s):")
        for problem in hard:
            print(f"  - {problem}")
        sys.exit(1)
    print("\nchecks passed")


if __name__ == "__main__":
    main()
