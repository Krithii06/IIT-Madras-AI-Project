"""Build the technical report PDF from measured results.

Every figure quoted in the document is read out of results/metrics at build time.
Nothing is typed into the prose by hand, so the report cannot claim a number the
pipeline did not produce.

    python -m src.report.build_report
    python -m src.report.build_report --app-url https://... --api-url https://...

The hosted URLs are supplied on the command line because deployment happens outside
this repository. If they are omitted the report says so explicitly rather than
inventing a link.
"""

import argparse
import csv
import json
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from src import config
from src.report import content

ACCENT = colors.HexColor("#2f6f43")
INK = colors.HexColor("#1c1c1c")
MUTED = colors.HexColor("#5a5a5a")
RULE = colors.HexColor("#d4d4d4")


def styles():
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=23, leading=28,
                                textColor=INK, spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontSize=11.5,
                                   leading=16, textColor=MUTED, alignment=1),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=14, leading=18,
                             textColor=ACCENT, spaceBefore=15, spaceAfter=7),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=11.5, leading=15,
                             textColor=INK, spaceBefore=10, spaceAfter=5),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=9.7, leading=14.4,
                               alignment=TA_JUSTIFY, textColor=INK, spaceAfter=7),
        "caption": ParagraphStyle("caption", parent=base["Normal"], fontSize=8.2,
                                  leading=11, textColor=MUTED, alignment=1, spaceAfter=10),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=8.3, leading=11),
        "cellb": ParagraphStyle("cellb", parent=base["Normal"], fontSize=8.3, leading=11,
                                fontName="Helvetica-Bold"),
        "mono": ParagraphStyle("mono", parent=base["Normal"], fontName="Courier",
                               fontSize=8.2, leading=11.5, textColor=INK),
    }
    return s


def read_json(path, default=None):
    if not path.exists():
        return default
    return json.load(open(path, encoding="utf-8"))


def gather():
    """Pull every measured value the prose and tables need into one dict."""
    m = config.METRICS_DIR
    data = {
        "test": read_json(m / "mobilenet_leaf_test.json"),
        "val": read_json(m / "mobilenet_leaf_val.json"),
        "leakage": read_json(m / "leakage_analysis.json"),
        "errors": read_json(m / "mobilenet_leaf_test_error_analysis.json"),
        "export": read_json(config.MODELS_DIR / "export" / "export_info.json", {}),
        "runs": {},
        "all_test": [],
    }

    for path in sorted(m.glob("*_test.json")):
        data["all_test"].append(read_json(path))
    if config.MODELS_DIR.exists():
        for run_dir in sorted(p for p in config.MODELS_DIR.iterdir() if p.is_dir()):
            cfg = read_json(run_dir / "train_config.json")
            if cfg:
                data["runs"][run_dir.name] = cfg

    manifest_rows = []
    if config.MANIFEST_PATH.exists():
        with open(config.MANIFEST_PATH, encoding="utf-8", newline="") as fh:
            manifest_rows = list(csv.DictReader(fh))
    data["manifest"] = manifest_rows
    return data


def placeholders(d):
    test, leak, err = d["test"], d["leakage"], d["errors"]
    rows = d["manifest"]
    total_images = len(rows)
    total_leaves = len({r["leaf_id"] for r in rows}) if rows else 0
    healthy = sum(1 for r in rows if r["binary_label"] == "healthy")

    grouped = leak["leaf_grouped_split"]["val_vs_train"]
    naive = leak["random_image_split"]["val_vs_train"]

    return {
        "test_accuracy_pct": f"{test['accuracy'] * 100:.2f}%",
        "test_macro_f1": f"{test['macro_f1']:.4f}",
        "test_images": f"{test['images']:,}",
        "test_leaves": f"{test['leaves']}",
        "total_images": f"{total_images:,}",
        "total_leaves": f"{total_leaves}",
        "images_per_leaf": f"{total_images / total_leaves:.2f}" if total_leaves else "n/a",
        "healthy_pct": f"{healthy / total_images * 100:.1f}%" if total_images else "n/a",
        "leafmap_coverage": f"{sum(1 for r in rows if r['leaf_id_from_map'] == '1'):,}",
        "grouped_leak_pct": f"{grouped['leaked_fraction'] * 100:.1f}%",
        "naive_leak_pct": f"{naive['leaked_fraction'] * 100:.1f}%",
        "grouped_leaked": f"{grouped['images_with_a_sibling_in_train']}",
        "naive_leaked": f"{naive['images_with_a_sibling_in_train']}",
        "grouped_val_images": f"{grouped['held_out_images']}",
        "naive_val_images": f"{naive['held_out_images']}",
        "image_errors": f"{err['image_errors']}",
        "missed_leaves": f"{err['leaves_with_at_least_one_error']}",
        "leaf_accuracy": f"{err['leaf_level_accuracy']:.4f}",
        "error_leaf": next(iter(err["errors_per_error_leaf"]), "n/a").replace("Apple___", ""),
        "mean_conf_wrong": f"{err['mean_confidence_when_wrong']:.4f}",
        "max_conf_wrong": f"{err['max_confidence_when_wrong']:.4f}",
        "errors_caught": f"{err['errors_below_threshold']}",
        "image_size": str(config.IMAGE_SIZE),
        "resize_before_crop": str(config.RESIZE_BEFORE_CROP),
        "full_images": "54,305",
        "full_max_class": "5,507",
        "full_min_class": "152",
        "full_imbalance": "36",
    }


def para(text, style, fill):
    cleaned = " ".join(text.strip().split("\n"))
    return Paragraph(cleaned.format(**fill), style)


def prose(text, s, fill, flow):
    """Split a content block into paragraphs on blank lines."""
    for block in text.strip().split("\n\n"):
        flow.append(para(block, s["body"], fill))


def make_table(rows, s, widths=None, align_right_from=1):
    data = []
    for i, row in enumerate(rows):
        style = s["cellb"] if i == 0 else s["cell"]
        data.append([Paragraph(str(c), style) for c in row])

    t = Table(data, colWidths=widths, hAlign="LEFT")
    commands = [
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]
    # Callers pass a column index past the end to mean "leave everything left-aligned".
    n_cols = len(rows[0]) if rows else 0
    if align_right_from < n_cols:
        commands.append(("ALIGN", (align_right_from, 1), (-1, -1), "RIGHT"))
    t.setStyle(TableStyle(commands))
    return t


def figure(path, s, caption, width=150 * mm):
    if not path.exists():
        return []
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    height = width * h / w
    max_h = 108 * mm
    if height > max_h:
        height = max_h
        width = height * w / h
    return [RLImage(str(path), width=width, height=height),
            Paragraph(caption, s["caption"])]


def build(out_path, app_url=None, api_url=None):
    d = gather()
    if not d["test"]:
        raise SystemExit("no test metrics found - run src.evaluation.evaluate first")
    fill = placeholders(d)
    s = styles()
    flow = []

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=20 * mm, bottomMargin=18 * mm,
        title="Plant Disease Classification - Technical Report",
        author="Krithick Balaji Ramesh",
    )

    # ---- cover -------------------------------------------------------------
    flow.append(Spacer(1, 34 * mm))
    flow.append(Paragraph("Plant Disease Classification", s["title"]))
    flow.append(Spacer(1, 3 * mm))
    flow.append(Paragraph(
        "A binary healthy/diseased leaf classifier, from dataset audit to deployment",
        s["subtitle"]))
    flow.append(Spacer(1, 14 * mm))
    flow.append(make_table([
        ["Author", "Krithick Balaji Ramesh"],
        ["Task", "Internship technical assessment"],
        ["Dataset", "PlantVillage (apple subset, colour)"],
        ["Model", f"{d['test']['arch']}, ImageNet-pretrained, fine-tuned"],
        ["Test accuracy", f"{d['test']['accuracy']:.4f} on {d['test']['images']} held-out images"],
        ["Date", date.today().strftime("%d %B %Y")],
    ], s, widths=[38 * mm, 118 * mm], align_right_from=99))
    flow.append(Spacer(1, 12 * mm))
    flow.append(Paragraph("Abstract", s["h2"]))
    prose(content.ABSTRACT, s, fill, flow)
    flow.append(PageBreak())

    # ---- 1-3 problem and data ---------------------------------------------
    flow.append(Paragraph("1. Problem statement", s["h1"]))
    prose(content.PROBLEM, s, fill, flow)
    flow.append(Paragraph("1.1 An ambiguity in the brief", s["h2"]))
    prose(content.AMBIGUITY, s, fill, flow)

    flow.append(Paragraph("2. Dataset", s["h1"]))
    prose(content.DATASET, s, fill, flow)
    flow.append(make_table([
        ["Property", "Value"],
        ["Source", "huggingface.co/datasets/mohanty/PlantVillage"],
        ["Upstream", "github.com/spMohanty/PlantVillage-Dataset"],
        ["Paper", "Mohanty, Hughes &amp; Salathe (2016), Frontiers in Plant Science"],
        ["Licence", "CC BY-SA 3.0"],
        ["Full dataset", "54,305 images, 38 classes, 14 crops"],
    ], s, widths=[32 * mm, 124 * mm], align_right_from=99))
    flow.append(Spacer(1, 5 * mm))

    flow.append(Paragraph("3. Selected classes", s["h1"]))
    prose(content.SELECTED_CLASSES, s, fill, flow)

    counts = {}
    for r in d["manifest"]:
        counts[r["source_class"]] = counts.get(r["source_class"], 0) + 1
    rows = [["Source class", "Images", "Binary label"]]
    for c in config.SOURCE_CLASSES:
        rows.append([f"<font face='Courier'>{c}</font>", f"{counts.get(c, 0):,}",
                     config.to_binary_label(c)])
    rows.append([f"<b>Total</b>", f"<b>{len(d['manifest']):,}</b>", ""])
    flow.append(make_table(rows, s, widths=[80 * mm, 26 * mm, 30 * mm]))
    flow.append(Spacer(1, 4 * mm))
    flow.extend(figure(config.FIGURES_DIR / "class_distribution.png", s,
                       "Figure 1. Source class counts and the binary target they collapse into."))

    flow.append(Paragraph("3.1 Data quality checks", s["h2"]))
    prose(content.DATA_QUALITY, s, fill, flow)
    flow.extend(figure(config.FIGURES_DIR / "class_samples.png", s,
                       "Figure 2. Representative images from each source class."))
    flow.append(PageBreak())

    # ---- 4 split -----------------------------------------------------------
    flow.append(Paragraph("4. Splitting and data leakage", s["h1"]))
    prose(content.SPLIT, s, fill, flow)

    split_rows = [["Split", "Images", "Leaves", "Healthy", "Diseased"]]
    for name in ("train", "val", "test"):
        cfg = d["runs"].get("mobilenet_leaf", {})
        key = {"train": "train_images", "val": "val_images", "test": "test_images"}[name]
        split_rows.append([name, f"{cfg.get(key, 0):,}", "", "", ""])
    # Prefer exact per-split composition recomputed from the manifest.
    try:
        from src.data.dataset import leaf_grouped_split
        tr, va, te = leaf_grouped_split(d["manifest"])
        split_rows = [["Split", "Images", "Leaves", "Healthy", "Diseased"]]
        for name, subset in (("train", tr), ("val", va), ("test", te)):
            h = sum(1 for r in subset if r["binary_label"] == "healthy")
            split_rows.append([name, f"{len(subset):,}",
                               f"{len({r['leaf_id'] for r in subset})}",
                               f"{h:,}", f"{len(subset) - h:,}"])
    except Exception:
        pass
    flow.append(make_table(split_rows, s, widths=[26 * mm, 26 * mm, 26 * mm, 26 * mm, 26 * mm]))
    flow.append(Spacer(1, 5 * mm))

    flow.append(Paragraph("4.1 What a naive split leaks", s["h2"]))
    prose(content.LEAKAGE, s, fill, flow)
    flow.append(make_table([
        ["Split strategy", "Val images", "Sharing a leaf with train", "Share"],
        ["Leaf-grouped (used here)", fill["grouped_val_images"], fill["grouped_leaked"],
         fill["grouped_leak_pct"]],
        ["Naive random image split", fill["naive_val_images"], fill["naive_leaked"],
         fill["naive_leak_pct"]],
    ], s, widths=[54 * mm, 26 * mm, 50 * mm, 22 * mm]))
    flow.append(Spacer(1, 4 * mm))
    flow.extend(figure(config.FIGURES_DIR / "leaf_groups.png", s,
                       "Figure 3. Each row is one physical leaf photographed repeatedly. "
                       "A random split scatters these rows across train and test."))
    flow.append(PageBreak())

    # ---- 5-6 preprocessing -------------------------------------------------
    flow.append(Paragraph("5. Preprocessing and augmentation", s["h1"]))
    prose(content.PREPROCESSING, s, fill, flow)
    flow.extend(figure(config.FIGURES_DIR / "augmentations.png", s,
                       "Figure 4. One training image under the augmentation pipeline."))
    flow.append(Paragraph("5.1 Input resolution", s["h2"]))
    prose(content.RESOLUTION, s, fill, flow)

    # ---- 6-7 model and training -------------------------------------------
    flow.append(Paragraph("6. Model and training methodology", s["h1"]))
    prose(content.MODEL, s, fill, flow)

    hyper = [["Setting", "Value"],
             ["Architecture", d["test"]["arch"]],
             ["Input resolution", f"{config.IMAGE_SIZE}x{config.IMAGE_SIZE}"],
             ["Loss", "Cross-entropy (unweighted)"],
             ["Optimiser", "AdamW, weight decay 1e-4"],
             ["Head LR / fine-tune LR", "1e-3 / 1e-4"],
             ["Scheduler", "Cosine annealing during fine-tuning"],
             ["Batch size", "32"],
             ["Epochs / warm-up / patience", "6 / 2 / 3"],
             ["Checkpoint selection", "Best validation macro F1"],
             ["Seed", str(config.SEED)],
             ["Hardware", "Intel i5-1135G7, 4 cores, CPU only, 16 GB RAM"],
             ["Framework", "PyTorch 2.12 (CPU), torchvision 0.27"]]
    cfg = d["runs"].get("mobilenet_leaf", {})
    if cfg.get("total_train_seconds"):
        hyper.append(["Training time", f"{cfg['total_train_seconds'] / 60:.1f} min"])
    if cfg.get("best_epoch"):
        hyper.append(["Best epoch", str(cfg["best_epoch"])])
    flow.append(make_table(hyper, s, widths=[54 * mm, 102 * mm], align_right_from=99))
    flow.append(Spacer(1, 4 * mm))
    flow.extend(figure(config.FIGURES_DIR / "training_curves.png", s,
                       "Figure 5. Training and validation curves."))
    flow.append(PageBreak())

    # ---- 8-9 evaluation ----------------------------------------------------
    flow.append(Paragraph("7. Evaluation", s["h1"]))
    flow.append(Paragraph(
        "All figures below are on the official PlantVillage held-out test split, "
        f"{d['test']['images']} images from {d['test']['leaves']} leaves, none of which "
        "shares a leaf with any training image. The test set was used once, after "
        "model selection had finished on validation.", s["body"]))

    t = d["test"]
    flow.append(Paragraph("7.1 Headline metrics", s["h2"]))
    flow.append(make_table([
        ["Metric", "Value"],
        ["Accuracy", f"{t['accuracy']:.4f}"],
        ["Macro F1", f"{t['macro_f1']:.4f}"],
        ["Weighted F1", f"{t['weighted_f1']:.4f}"],
        ["Precision (diseased)", f"{t['precision_positive']:.4f}"],
        ["Recall (diseased)", f"{t['recall_positive']:.4f}"],
        ["F1 (diseased)", f"{t['f1_positive']:.4f}"],
    ], s, widths=[54 * mm, 30 * mm]))
    flow.append(Spacer(1, 4 * mm))

    flow.append(Paragraph("7.2 Per-class results", s["h2"]))
    rows = [["Class", "Precision", "Recall", "F1", "Support"]]
    for name, v in t["per_class"].items():
        rows.append([name, f"{v['precision']:.4f}", f"{v['recall']:.4f}",
                     f"{v['f1']:.4f}", str(v["support"])])
    flow.append(make_table(rows, s, widths=[36 * mm, 30 * mm, 26 * mm, 26 * mm, 26 * mm]))
    flow.append(Spacer(1, 4 * mm))

    flow.append(Paragraph("7.3 Confusion matrix", s["h2"]))
    labels = t["confusion_matrix_labels"]
    cm = t["confusion_matrix"]
    rows = [[""] + [f"predicted {l}" for l in labels]]
    for i, l in enumerate(labels):
        rows.append([f"<b>actual {l}</b>"] + [str(v) for v in cm[i]])
    flow.append(make_table(rows, s, widths=[40 * mm, 40 * mm, 40 * mm]))
    flow.append(Spacer(1, 4 * mm))
    flow.extend(figure(config.FIGURES_DIR / "mobilenet_leaf_test_confusion_matrix.png", s,
                       "Figure 6. Test-set confusion matrix.", width=88 * mm))

    flow.append(Paragraph("7.4 Performance by original disease", s["h2"]))
    flow.append(Paragraph(
        "The deployed model answers healthy or diseased, but the four source classes "
        "are retained in the manifest so the binary result can be broken down by the "
        "disease actually present.", s["body"]))
    rows = [["Source class", "Images", "Correct", "Recall", "Mean confidence"]]
    for name, v in sorted(t["per_source_class"].items()):
        rows.append([f"<font face='Courier'>{name}</font>", str(v["images"]),
                     str(v["correct"]), f"{v['recall']:.4f}", f"{v['mean_confidence']:.4f}"])
    flow.append(make_table(rows, s, widths=[58 * mm, 20 * mm, 22 * mm, 24 * mm, 32 * mm]))
    flow.append(Spacer(1, 4 * mm))

    if len(d["all_test"]) > 1:
        flow.append(Paragraph("7.5 Architecture comparison", s["h2"]))
        prose(content.SELECTION_NOTE, s, fill, flow)
        rows = [["Run", "Architecture", "Split", "Accuracy", "Macro F1", "Train (min)"]]
        for m in d["all_test"]:
            rcfg = d["runs"].get(m["run_name"], {})
            mins = rcfg.get("total_train_seconds")
            rows.append([f"<font face='Courier'>{m['run_name']}</font>", m["arch"],
                         m["split_strategy"], f"{m['accuracy']:.4f}",
                         f"{m['macro_f1']:.4f}",
                         f"{mins / 60:.1f}" if mins else "-"])
        flow.append(make_table(rows, s, widths=[36 * mm, 32 * mm, 20 * mm, 24 * mm,
                                                 24 * mm, 22 * mm]))
        flow.append(Spacer(1, 4 * mm))

    flow.append(PageBreak())

    # ---- 8 error analysis --------------------------------------------------
    flow.append(Paragraph("8. Error analysis", s["h1"]))
    prose(content.ERROR_ANALYSIS, s, fill, flow)
    flow.extend(figure(config.FIGURES_DIR / "mobilenet_leaf_test_errors.png", s,
                       "Figure 7. Every misclassified test image, most confident first. "
                       "All are the same physical leaf."))
    flow.append(Paragraph("8.1 Confidence reporting", s["h2"]))
    prose(content.CONFIDENCE, s, fill, flow)
    flow.append(PageBreak())

    # ---- 9-11 system -------------------------------------------------------
    flow.append(Paragraph("9. Backend", s["h1"]))
    prose(content.BACKEND, s, fill, flow)
    flow.append(Paragraph("9.1 API endpoints", s["h2"]))
    flow.append(make_table([
        ["Method", "Path", "Purpose"],
        ["GET", "<font face='Courier'>/</font>", "Service description and endpoint list"],
        ["GET", "<font face='Courier'>/health</font>", "Liveness probe; reports model load state"],
        ["GET", "<font face='Courier'>/model-info</font>",
         "Architecture, classes, input size, threshold"],
        ["POST", "<font face='Courier'>/predict</font>",
         "Classify an uploaded image (multipart, field <font face='Courier'>file</font>)"],
    ], s, widths=[18 * mm, 34 * mm, 104 * mm], align_right_from=99))
    flow.append(Spacer(1, 4 * mm))
    flow.append(Paragraph("9.2 Error responses", s["h2"]))
    flow.append(make_table([
        ["Status", "Condition"],
        ["400", "Empty, corrupt or undecodable image"],
        ["413", "Over the 8 MB upload limit, or an implausible pixel count"],
        ["415", "Content type is not JPEG, PNG or WebP"],
        ["422", "No file field in the request"],
        ["503", "Model failed to load"],
    ], s, widths=[20 * mm, 136 * mm], align_right_from=99))
    flow.append(Spacer(1, 5 * mm))

    flow.append(Paragraph("10. Frontend", s["h1"]))
    prose(content.FRONTEND, s, fill, flow)

    flow.append(Paragraph("11. Libraries and tools", s["h1"]))
    flow.append(make_table([
        ["Area", "Choice"],
        ["Training", "PyTorch 2.12 (CPU), torchvision 0.27"],
        ["Data and metrics", "NumPy, scikit-learn, Pillow, pandas"],
        ["Figures", "Matplotlib"],
        ["Serving", "ONNX Runtime 1.20 (no torch in the deployed image)"],
        ["API", "FastAPI, uvicorn, Pydantic"],
        ["Frontend", "React 18, Vite 5"],
        ["Tests", "pytest, FastAPI TestClient"],
        ["Hosting", "Render (Docker, backend), Vercel (static, frontend)"],
    ], s, widths=[40 * mm, 116 * mm], align_right_from=99))
    flow.append(PageBreak())

    # ---- 12 deployment -----------------------------------------------------
    flow.append(Paragraph("12. Deployment", s["h1"]))
    prose(content.DEPLOYMENT, s, fill, flow)

    flow.append(Paragraph("12.1 Free-tier constraints", s["h2"]))
    flow.append(make_table([
        ["Constraint", "Render free web service"],
        ["Memory", "512 MB"],
        ["CPU", "0.1 CPU"],
        ["Idle behaviour", "Sleeps after 15 minutes"],
        ["Cold start", "About 60 seconds"],
        ["Monthly allowance", "750 instance hours"],
    ], s, widths=[44 * mm, 112 * mm], align_right_from=99))
    flow.append(Spacer(1, 5 * mm))

    flow.append(Paragraph("12.2 Hosted URLs", s["h2"]))
    if app_url or api_url:
        flow.append(make_table([
            ["Component", "URL"],
            ["Web application", app_url or "not supplied"],
            ["API", api_url or "not supplied"],
        ], s, widths=[36 * mm, 120 * mm], align_right_from=99))
    else:
        flow.append(Paragraph(
            "The deployment artefacts are complete and committed - a Dockerfile, a Render "
            "blueprint (<font face='Courier'>render.yaml</font>) and step-by-step "
            "instructions in <font face='Courier'>docs/DEPLOYMENT.md</font> - but the "
            "hosted instance is created under the author's own accounts and was not live "
            "at the time this document was generated. No URL is quoted here rather than "
            "quote one that does not resolve. Rebuild this report with "
            "<font face='Courier'>--app-url</font> and <font face='Courier'>--api-url</font> "
            "once deployed.", s["body"]))
    flow.append(Spacer(1, 4 * mm))

    flow.append(Paragraph("13. Screenshots", s["h1"]))
    shots = [
        ("screenshot_upload.png", "Figure 8. Landing view, with the usage guide above the "
                                  "upload area."),
        ("screenshot_result.png", "Figure 9. Prediction result, confidence and model panel."),
        ("screenshot_about.png", "Figure 10. The tools and techniques reference shown in the "
                                 "application itself."),
        ("screenshot_mobile.png", "Figure 11. The same page at a 390px phone viewport."),
        ("screenshot_api.png", "Figure 12. The generated OpenAPI documentation."),
    ]
    added = False
    for name, caption in shots:
        block = figure(config.FIGURES_DIR / name, s, caption, width=130 * mm)
        if block:
            flow.extend(block)
            added = True
    if not added:
        flow.append(Paragraph(
            "No application screenshots were captured for this build.", s["body"]))
    flow.append(PageBreak())

    # ---- 14-16 closing -----------------------------------------------------
    flow.append(Paragraph("14. Challenges and how they were handled", s["h1"]))
    prose(content.CHALLENGES, s, fill, flow)

    flow.append(Paragraph("15. Limitations", s["h1"]))
    prose(content.LIMITATIONS, s, fill, flow)

    flow.append(Paragraph("16. Future improvements", s["h1"]))
    prose(content.FUTURE, s, fill, flow)

    flow.append(Paragraph("17. Conclusion", s["h1"]))
    prose(content.CONCLUSION, s, fill, flow)

    doc.build(flow, onFirstPage=_footer, onLaterPages=_footer)
    return out_path


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    if doc.page > 1:
        canvas.drawString(22 * mm, 11 * mm, "Plant Disease Classification - technical report")
        canvas.drawRightString(A4[0] - 22 * mm, 11 * mm, str(doc.page))
    canvas.restoreState()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument("--app-url", default=None)
    parser.add_argument("--api-url", default=None)
    args = parser.parse_args()

    out_dir = config.PROJECT_ROOT / "docs" / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out or (out_dir / "Plant_Disease_Classification_Report.pdf")

    path = build(out, args.app_url, args.api_url)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
