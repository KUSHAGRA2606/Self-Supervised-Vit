"""Build the Week 4 PDF and editable Markdown report from measured outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path


LABELS = {
    "naive_attention_week3_dino_s8": "Naive attention (Week 3 DINO ViT-S/8)",
    "linear_dinov2_s14_224": "Linear DINOv2 ViT-S/14, 224 px",
    "linear_dinov2_s14_448": "Linear DINOv2 ViT-S/14, 448 px",
    "linear_dino_s8_224": "Linear DINO ViT-S/8, 224 px",
    "plus_ms_tta": "+ms DINOv2 ViT-S/14, 448 px + TTA",
}


def _fmt(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value)*100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _read_results(root: Path):
    path = root / "results" / "metrics" / "comparison.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(root: str | Path) -> tuple[Path, Path]:
    """Create a concise PDF report and Markdown source using ReportLab."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, Image, PageBreak, KeepTogether)

    root = Path(root)
    pdf_path = root / "Week4_Report.pdf"
    md_path = root / "Week4_Report.md"
    status_path = root / "results" / "metrics" / "run_status.json"
    if not status_path.exists():
        raise RuntimeError("Final report blocked: run_status.json is missing; finish the corrected test evaluation first.")
    run_status = json.loads(status_path.read_text(encoding="utf-8"))
    if run_status.get("status") != "complete":
        raise RuntimeError(f"Final report blocked: current run status is {run_status.get('status')!r}.")
    results = _read_results(root)
    if results is None:
        raise FileNotFoundError("Run the segmentation notebook through metric export before building the final report.")
    expected_test = int(run_status.get("n_test", 0))
    if expected_test < 10 or any(int(row.get("n_test", -1)) != expected_test for row in results):
        raise RuntimeError("Final report blocked: metrics do not cover the full held-out test split.")

    table_rows = []
    for row in results:
        table_rows.append({**row, "label": LABELS.get(row.get("setting"), row.get("setting"))})
    supervised = [r for r in table_rows if r.get("setting") != "naive_attention_week3_dino_s8"]
    best_miou = max(supervised, key=lambda r: float(r.get("mIoU", -1)))
    best_target = max(supervised, key=lambda r: float(r.get("target_IoU", -1)))
    naive = next(r for r in table_rows if r.get("setting") == "naive_attention_week3_dino_s8")
    by_name = {r.get("setting"): r for r in table_rows}
    linear224 = by_name.get("linear_dinov2_s14_224", {})
    linear448 = by_name.get("linear_dinov2_s14_448", {})
    dino8 = by_name.get("linear_dino_s8_224", {})
    ms = by_name.get("plus_ms_tta", {})

    markdown = f"""# Week 4 Report: First Segmentation Models on Frozen DINOv2

## Aim and method

This experiment converts DESI g/r/z cutouts into three-class pixel masks: background (0), the central/nearest target galaxy (1), and other sources (2). Because DESI segmentation ground truth is unavailable in this assignment, labels are pseudo-labels from photutils `Background2D` and `SourceFinder`; all scores measure agreement with this pipeline, not absolute astronomical truth.

About 3,000 cutouts were normalized per band by subtracting a local `Background2D` estimate and dividing by its background RMS, then converted with the Week 1 g/r/z asinh RGB mapping. A fixed seed-42 70/10/20 split was saved to `data/splits.json`. The frozen backbones feed a trainable `BatchNorm2d + 1x1 convolution` head; only the head is optimized. Checkpoint selection uses validation foreground mean IoU. The test set is used once for final reporting.

## Test results

| Setup | Backbone/input/grid | mIoU | All-source IoU | Target-galaxy IoU |
|---|---|---:|---:|---:|
"""
    for r in table_rows:
        label = r["label"]
        grid = r.get("patch_grid", "")
        backbone = r.get("backbone", "")
        size = r.get("input_px", "")
        detail = f"{backbone}, {size}px, {grid}"
        miou = _fmt(r.get("mIoU"))
        all_iou = _fmt(r.get("all_source_iou")) if r.get("all_source_iou") is not None else "N/A"
        target = _fmt(r.get("target_IoU"))
        markdown += f"| {label} | {detail} | {miou} | {all_iou} | {target} |\n"

    q1 = f"The best supervised head reaches {_fmt(best_target.get('target_IoU'))} target IoU, compared with {_fmt(naive.get('target_IoU'))} for Week 3 attention (difference {_fmt(float(best_target.get('target_IoU',0))-float(naive.get('target_IoU',0)))}). The attention baseline has no class output; its separate all-source IoU is {_fmt(naive.get('all_source_iou'))}."
    delta_resolution = float(linear448.get("mIoU", 0)) - float(linear224.get("mIoU", 0))
    delta_ms = float(ms.get("mIoU", 0)) - float(linear448.get("mIoU", 0))
    delta_grid = float(dino8.get("mIoU", 0)) - float(linear224.get("mIoU", 0))
    q2 = (f"DINOv2 linear mIoU is {_fmt(linear224.get('mIoU'))} at 224px and {_fmt(linear448.get('mIoU'))} at 448px "
          f"(resolution change: {_fmt(delta_resolution)}); +ms/TTA is {_fmt(ms.get('mIoU'))} "
          f"(versus 448px linear: {_fmt(delta_ms)}); DINO ViT-S/8 linear is {_fmt(dino8.get('mIoU'))} "
          f"(versus DINOv2 224px: {_fmt(delta_grid)}). The +ms-versus-448 comparison combines multiscale features and TTA; "
          "the smaller-patch comparison also changes the backbone, so these are informative ablations rather than perfectly isolated causal tests.")
    note_file = root / "results" / "metrics" / "visual_review_notes.json"
    review_notes = json.loads(note_file.read_text(encoding="utf-8")) if note_file.exists() else {}
    observed = "; ".join(str(review_notes[k]).strip() for k in
                          ("fragmentation", "faint_outskirts", "blends_stars") if str(review_notes.get(k, "")).strip())
    q3 = (f"Visual review of the hard set found: {observed}. " if observed else
          "Visual review notes have not yet been entered; inspect the hard-set gallery before making a claim about correction. ")
    q3 += "Because training targets are photutils labels, a head can learn label fragmentation as well as real source structure; agreement metrics are not ground-truth validation."
    q4 = "Pseudo-labels inherit missed faint outskirts, deblending choices, fragments, and merged neighbours from classical source finding. Better labels would come from expert-reviewed masks or adjudicated multi-annotator galaxy boundaries, with explicit target identity and blend boundaries."
    next_note = str(review_notes.get("recommended_next_step", "")).strip()
    q5 = (next_note + " " if next_note else "Visual review is still needed to choose between label and model changes. ")
    q5 += f"The best measured mIoU is {_fmt(best_miou.get('mIoU'))} and best target IoU is {_fmt(best_target.get('target_IoU'))}."
    markdown += f"""
## Interpretation and required questions

1. **How much better is a trained linear head than naive attention at isolating the target?** {q1}
2. **Which matters more: multiscale features or a finer patch grid?** {q2}
3. **Does the head fix fragmentation or reproduce pseudo-label errors?** {q3}
4. **What are the pseudo-label limits, and what labels would help?** {q4}
5. **What should improve next: labels, features, or head?** {q5}

## Limitations

All metrics are computed against photutils pseudo-labels rather than human ground truth. The center/nearest-source rule can assign the wrong object when the target is offset, the cutout center is blank, or a blend is unresolved. A model that agrees with the labels can therefore still be astronomically wrong. The hard-set examples are included for qualitative review.
"""
    md_path.write_text(markdown, encoding="utf-8")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER,
                              textColor=colors.HexColor("#15334A"), spaceAfter=14))
    styles.add(ParagraphStyle(name="SmallBody", parent=styles["BodyText"], fontSize=8.5,
                              leading=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="Question", parent=styles["BodyText"], fontSize=8.5,
                              leading=12, leftIndent=8, spaceAfter=7))
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=0.6*inch,
                            leftMargin=0.6*inch, topMargin=0.55*inch, bottomMargin=0.55*inch)
    story = [Paragraph("Week 4 Report: First Segmentation Models on Frozen DINOv2", styles["ReportTitle"]),
             Paragraph("Aim and method", styles["Heading2"]),
             Paragraph("This experiment predicts three classes per pixel: background (0), target galaxy (1), and other sources (2). With no DESI segmentation ground truth supplied, photutils masks are pseudo-labels and the reported scores measure agreement with the classical pipeline. About 3,000 cutouts are background/RMS-normalized per band, converted with the Week 1 g/r/z asinh mapping, and split 70/10/20 with a saved seed-42 split. Frozen DINO/DINOv2 features feed a trainable BatchNorm plus 1x1 convolution head. Validation foreground mean IoU selects checkpoints; the held-out test split supplies final scores.", styles["SmallBody"]),
             Paragraph("Test results", styles["Heading2"])]
    data = [["Setup", "Input / grid", "mIoU", "All-source IoU", "Target IoU"]]
    for r in table_rows:
        data.append([r["label"], f"{r.get('input_px','')}px / {r.get('patch_grid','')}",
                     _fmt(r.get("mIoU")), _fmt(r.get("all_source_iou")) if r.get("all_source_iou") is not None else "N/A",
                     _fmt(r.get("target_IoU"))])
    tbl = Table(data, colWidths=[2.2*inch,1.1*inch,0.75*inch,1.05*inch,1.05*inch], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#15334A")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white), ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),7.3), ("LEADING",(0,0),(-1,-1),9),
        ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#AAB7C2")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#EEF3F6")]),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"), ("TOPPADDING",(0,0),(-1,-1),5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story.extend([tbl, Spacer(1,10), Paragraph("Answers to the report questions", styles["Heading2"])])
    for i, (question, answer) in enumerate([
        ("1. Trained head versus naive attention", q1),
        ("2. Multiscale versus finer grid", q2),
        ("3. Fragmentation", q3),
        ("4. Pseudo-label limits", q4),
        ("5. Next improvement", q5),
    ]):
        story.append(KeepTogether([Paragraph(f"<b>{question}</b> {answer}", styles["Question"])]))
    story.extend([Spacer(1,4), Paragraph("Limitations", styles["Heading2"]),
                  Paragraph("Scores are agreement with photutils pseudo-labels, not human-ground-truth segmentation. The center/nearest-source rule may choose the wrong object in offset or unresolved blends. Review the hard-set gallery before drawing scientific conclusions.", styles["SmallBody"])])
    gallery = root / "results" / "predictions" / "hard_set_gallery_01.png"
    if gallery.exists():
        gallery_flowable = Image(str(gallery))
        gallery_flowable._restrictSize(7.25*inch, 6.8*inch)
        story.extend([PageBreak(), Paragraph("Hard-set gallery", styles["Heading2"]),
                      Paragraph("Rows show DESI RGB, pseudo-label, Week 3 attention, linear DINOv2, and +ms with TTA.", styles["SmallBody"]),
                      gallery_flowable])
    doc.build(story)
    return pdf_path, md_path

