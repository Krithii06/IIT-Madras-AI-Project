# LaTeX report (Overleaf)

Source for the technical report. Roughly 7 pages, compiles with pdfLaTeX.

```
latex/
├── main.tex
├── figures/
│   ├── class_distribution.png
│   ├── leaf_groups.jpg
│   ├── mobilenet_leaf_test_errors.jpg
│   ├── screenshot_upload.png
│   └── screenshot_result.png
└── README.md
```

## Putting it on Overleaf

1. Zip the `latex/` folder — `main.tex` and `figures/` must stay in the same relative
   positions, since `\graphicspath{{figures/}}` expects it.
2. Overleaf → **New Project → Upload Project** → select the zip.
3. Overleaf picks `main.tex` as the root document automatically. If it does not, use
   **Menu → Main document**.
4. Compiler should be **pdfLaTeX** (Menu → Compiler). That is the default.

Only standard packages are used — `geometry`, `graphicx`, `booktabs`, `xcolor`,
`caption`, `subcaption`, `enumitem`, `float`, `hyperref`, `lmodern` — all present in
Overleaf's default TeX Live image. Nothing needs installing.

## Regenerating the figures

The figures are produced by the project pipeline, not drawn by hand:

```bash
python -m src.evaluation.figures --dataset
python -m src.evaluation.error_analysis --run-name mobilenet_leaf --split test
python -m src.report.screenshots        # needs backend + frontend running
python -m src.report.latex_assets       # copies and compresses into figures/
```

`latex_assets` writes photographic grids as JPEG and charts and screenshots as PNG.
Re-encoding the leaf grids as PNG made them *larger* than the source, because PNG is
the wrong codec for photographs; that one change took the folder from 2.5 MB to 0.7 MB.

## Checking it before uploading

There is no TeX toolchain on the development machine, so the document is validated
rather than compiled:

```bash
python -m src.report.check_latex
```

That verifies environments and braces balance, every `\includegraphics` target exists,
no stray unescaped underscores survive outside file paths and URLs, and — the point of
it — that every result quoted in the prose still matches `results/metrics/`. If a model
is retrained and a number moves, this fails instead of letting a stale figure sit in
the report.

## Note on the numbers

Every measured value in `main.tex` is checked against the metrics JSON by the script
above. If you edit a result by hand, run the checker or it will drift.

The hosted URL is the one thing the report cannot fill in on its own. Once the app is
deployed, add it to the Deployment section.
