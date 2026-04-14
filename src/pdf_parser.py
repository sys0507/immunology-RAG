# =============================================================================
# ImmunoBiology RAG — PDF Parser & Image Extractor
# =============================================================================
# Adapted from Tesla RAG: src/parser/pdf_parse.py + src/parser/image_handler.py
# Key differences from Tesla system:
#   - Multi-PDF support: accepts List[Path], auto-discovers data/raw/*.pdf
#   - Richer metadata: source_file, doc_type, chapter, chunk_id per chunk
#   - Recalibrated parameters for Janeway's Immunobiology 10e layout:
#       * Page size: ~595 × 842 pt (A4)
#       * Header crop: top 50 pt (contains running chapter title + page number)
#       * Footer crop: bottom 40 pt (contains book title + page number)
#       * Body text font: ~10-11 pt
#       * Chapter headings: bold, font >= 14 pt, text starts with "Chapter" or digit
#   - Image thresholds:
#       * Min width/height: 100 × 100 px (Tesla used 34 px — too small for textbook)
#       * PNG icons (logos, bullets) are skipped as in Tesla
#       * Figure caption detection: regex "Figure X.X" or "FIGURE"
#   - Layout detection: single-column (textbook) vs double-column (paper)
#     based on median x-coordinate of text blocks
#
# Usage:
#   python src/pdf_parser.py --inspect          # layout inspection only
#   python src/pdf_parser.py                    # full parse all PDFs in data/raw/
#   python src/pdf_parser.py --pdf path/to.pdf  # parse single PDF

# %% [Cell 1: Imports and configuration]
import os
import re
import json
import copy
import hashlib
import argparse
from pathlib import Path
from glob import glob
from typing import List, Optional, Tuple
from datetime import datetime

import io

import fitz  # PyMuPDF
from tqdm import tqdm
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Optional OCR support (required for scanned / image-based PDFs)
# Install: apt-get install -y tesseract-ocr tesseract-ocr-eng
#          pip install pytesseract Pillow
# ---------------------------------------------------------------------------
try:
    import pytesseract
    from PIL import Image as _PILImage
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

from src import constant
from src.fields.immuno_images import ImmunoImages
from src.fields.chunk_info_mongo import ChunkInfo
from src.client.mongodb_config import MongoConfig

# ---------------------------------------------------------------------------
# Parameter documentation (calibrated for Janeway's Immunobiology 10e)
# Run --inspect first to verify these for your specific PDF.
# ---------------------------------------------------------------------------

# Page layout parameters (all in PDF points, 1 pt = 1/72 inch)
# Janeway 10e: 595 × 842 pt (A4). Header occupies top ~50 pt (running title).
# Footer occupies bottom ~40 pt (page number + book title line).
PAGE_HEADER_CLIP = 50    # points from top to crop (removes running chapter title)
PAGE_FOOTER_CLIP = 40    # points from bottom to crop (removes page number line)

# Page range filter: skip front matter (cover, TOC, preface) and back matter (index)
# Janeway 10e: ~35 front pages before Chapter 1; adjust after --inspect
MIN_CONTENT_PAGE = 35    # 0-indexed; pages before this are skipped
MAX_CONTENT_PAGE = 9999  # set to actual last page after --inspect

# Image extraction thresholds
IMAGE_MIN_WIDTH   = 100   # px — skip small inline icons/arrows (Tesla used 34)
IMAGE_MIN_HEIGHT  = 100   # px
IMAGE_MIN_BYTES   = 5000  # bytes — skip tiny images that are decorative elements

# Figure caption detection
FIGURE_CAPTION_PATTERN = re.compile(
    r'(Figure|FIGURE|Fig\.?)\s+\d+[\.\-]\d*',
    re.IGNORECASE
)

# Chapter heading detection
# Janeway uses "Chapter N Title" on its own line with large bold font
CHAPTER_HEADING_PATTERN = re.compile(
    r'^(Chapter|CHAPTER)\s+(\d+)',
    re.IGNORECASE
)
# Also detect numeric chapter numbers as standalone headings (e.g., "3\nBasic Concepts")
CHAPTER_NUMBER_PATTERN = re.compile(r'^(\d{1,2})\s*$')

# Image title candidate scoring (adapted from Tesla image_handler.py)
TITLE_CONFIG = {
    "min_font_size": 9,      # pts — figure captions in Janeway are ~9-10pt
    "max_lines": 5,          # max lines a caption block can span
    "max_length": 200,       # max characters in a caption block
    "bold_weight": 0.7,      # bold font detection threshold
    "above_image_bonus": 2,  # scoring bonus for text above image
    "below_image_penalty": -1,# scoring penalty for text below image
    "min_score": 3,          # minimum score to be considered a caption
}

# MongoDB collection for storing chunks
chunk_collection = MongoConfig.get_collection(constant.mongo_collection)


# %% [Cell 2: Layout inspection]

def inspect_pdf_layout(pdf_path, sample_pages: List[int] = None) -> dict:
    """
    Inspect PDF layout and record key parameters.
    Run this FIRST before setting parsing parameters.

    Args:
        pdf_path: Path to the PDF file (str or Path — both accepted)
        sample_pages: 0-indexed page numbers to inspect (default: [0, 49, 99, 199])

    Returns:
        dict with layout statistics
    """
    pdf_path = Path(pdf_path)   # normalise: accept both str and Path
    if sample_pages is None:
        sample_pages = [0, 49, 99, 199]

    pdf = fitz.open(str(pdf_path))
    total_pages = len(pdf)
    report_lines = [
        f"PDF Layout Inspection Report",
        f"File: {pdf_path.name}",
        f"Total pages: {total_pages}",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 70,
    ]

    image_widths, image_heights, image_sizes = [], [], []
    font_sizes = []

    for page_idx in sample_pages:
        if page_idx >= total_pages:
            continue
        page = pdf.load_page(page_idx)
        w, h = page.rect.width, page.rect.height
        report_lines.append(f"\n--- Page {page_idx + 1} (0-indexed: {page_idx}) ---")
        report_lines.append(f"  Dimensions: {w:.1f} x {h:.1f} pt")

        # Analyze text blocks
        blocks = page.get_text("blocks")
        report_lines.append(f"  Text blocks: {len(blocks)}")
        if blocks:
            y_coords = [(b[1], b[3]) for b in blocks if b[6] == 0]
            if y_coords:
                min_y = min(y[0] for y in y_coords)
                max_y = max(y[1] for y in y_coords)
                report_lines.append(f"  Text y-range: {min_y:.1f} – {max_y:.1f} pt")
                report_lines.append(f"  Suggested header clip: {min_y:.0f} pt from top")
                report_lines.append(f"  Suggested footer clip: {h - max_y:.0f} pt from bottom")

        # Analyze font sizes
        dict_data = page.get_text("dict")
        for blk in dict_data.get("blocks", []):
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    font_sizes.append(span.get("size", 0))

        # Analyze images
        images = page.get_images(full=True)
        report_lines.append(f"  Images: {len(images)}")
        for img in images:
            xref = img[0]
            base = pdf.extract_image(xref)
            w_px, h_px = base["width"], base["height"]
            sz = len(base["image"])
            image_widths.append(w_px)
            image_heights.append(h_px)
            image_sizes.append(sz)
            report_lines.append(f"    {base['ext']} {w_px}x{h_px}px {sz//1024}KB")

    report_lines.append("\n" + "=" * 70)
    report_lines.append("SUMMARY")
    report_lines.append(f"  Total pages: {total_pages}")
    if image_widths:
        report_lines.append(
            f"  Image width  min/median/max: "
            f"{min(image_widths)} / {sorted(image_widths)[len(image_widths)//2]} / {max(image_widths)} px"
        )
        report_lines.append(
            f"  Image height min/median/max: "
            f"{min(image_heights)} / {sorted(image_heights)[len(image_heights)//2]} / {max(image_heights)} px"
        )
        report_lines.append(
            f"  Image size   min/median/max: "
            f"{min(image_sizes)//1024} / {sorted(image_sizes)[len(image_sizes)//2]//1024} / {max(image_sizes)//1024} KB"
        )
    if font_sizes:
        report_lines.append(
            f"  Font sizes min/median/max: "
            f"{min(font_sizes):.1f} / {sorted(font_sizes)[len(font_sizes)//2]:.1f} / {max(font_sizes):.1f} pt"
        )

    # Detect whether PDF is image-based (scanned) — important for OCR decision
    # Only count pages that are actually within range (short PDFs skip high page indices)
    checked_pages = [idx for idx in sample_pages if idx < total_pages]
    scanned_count = sum(
        1 for idx in checked_pages
        if len([b for b in pdf.load_page(idx).get_text("blocks") if b[6] == 0 and b[4].strip()]) == 0
        and len(pdf.load_page(idx).get_images(full=True)) > 0
    )
    # Majority rule on actually-checked pages
    pdf_type = "SCANNED (image-based) — OCR will be used" \
               if checked_pages and scanned_count > len(checked_pages) / 2 \
               else "digital (selectable text)"
    report_lines.append(f"\nPDF type detected : {pdf_type}")
    if "SCANNED" in pdf_type:
        ocr_status = "pytesseract available ✓" if _OCR_AVAILABLE else \
                     "pytesseract NOT installed — install before running Step 2:\n" \
                     "  apt-get install -y tesseract-ocr tesseract-ocr-eng\n" \
                     "  pip install pytesseract Pillow"
        report_lines.append(f"OCR status        : {ocr_status}")

    report_text = "\n".join(report_lines)

    # Save report
    os.makedirs(constant.diagnostics_dir, exist_ok=True)
    with open(constant.pdf_layout_report, "a", encoding="utf-8") as f:
        f.write(report_text + "\n\n")
    print(f"[Parser] Layout report saved to {constant.pdf_layout_report}")
    print(report_text)

    pdf.close()
    return {
        "total_pages": total_pages,
        "image_widths": image_widths,
        "image_heights": image_heights,
        "image_sizes": image_sizes,
        "font_sizes": font_sizes,
    }


# %% [Cell 3: Layout detection — single vs double column]

def detect_layout(page: fitz.Page) -> str:
    """
    Detect whether a page is single-column (textbook) or double-column (paper).

    Strategy: examine x-coordinates of text blocks.
    - Single-column: most blocks span the full page width
    - Double-column: blocks cluster into left (x < page_width/2) and right halves

    Returns: "single" or "double"
    """
    blocks = page.get_text("blocks")
    if not blocks:
        return "single"

    page_mid = page.rect.width / 2
    left_blocks = [b for b in blocks if b[6] == 0 and b[2] < page_mid * 1.1]
    right_blocks = [b for b in blocks if b[6] == 0 and b[0] > page_mid * 0.9]
    full_blocks = [
        b for b in blocks
        if b[6] == 0 and (b[2] - b[0]) > page.rect.width * 0.6
    ]

    # If most text blocks span >60% of page width → single column
    if len(full_blocks) > len(blocks) * 0.4:
        return "single"

    # If both left and right clusters have content → double column
    if len(left_blocks) >= 2 and len(right_blocks) >= 2:
        return "double"

    return "single"


# %% [Cell 4: Chapter heading detection]

def detect_chapter(text: str, current_chapter: str) -> str:
    """
    Attempt to detect a chapter heading in page text.
    Returns the detected chapter string, or the existing current_chapter if none found.
    """
    for line in text.split("\n")[:10]:  # Check first 10 lines only
        line = line.strip()
        m = CHAPTER_HEADING_PATTERN.match(line)
        if m:
            return f"Chapter {m.group(2)}"
    return current_chapter


# %% [Cell 5: Image extraction and caption detection]

def _is_caption_candidate(
    page: fitz.Page,
    block: tuple,
    img_y_top: float,
    above: bool,
) -> bool:
    """
    Score a text block as a potential figure caption using the Tesla scoring system,
    recalibrated for Janeway's Immunobiology layout.

    Scoring:
      +2  font size >= TITLE_CONFIG["min_font_size"]
      +1  bold font
      +0.5 <= max_lines lines
      +0.5 <= max_length characters
      +2  block is ABOVE the image (captions often appear below in textbooks)
      -1  block is BELOW the image

    Figure captions in textbooks typically appear BELOW images; for them the
    FIGURE_CAPTION_PATTERN match overrides the scoring system entirely.
    """
    # Filter non-text blocks
    if block[6] != 0 or not block[4].strip():
        return False

    # Direct figure caption pattern match → always include
    if FIGURE_CAPTION_PATTERN.search(block[4]):
        return True

    # Score-based heuristic for title-like blocks
    try:
        dict_page = page.get_text("dict")
        span = dict_page["blocks"][block[5]]["lines"][0]["spans"][0]
    except (IndexError, KeyError):
        return False

    text = block[4].strip()
    font_size = span.get("size", 0)
    is_bold = "bold" in span.get("font", "").lower()

    # Exclude sentence-ending text (body text, not captions)
    if text.endswith((".", "!", "?")):
        return False

    score = 0
    score += 2 if font_size >= TITLE_CONFIG["min_font_size"] else 0
    score += 1 if is_bold else 0
    score += 0.5 if (text.count("\n") + 1) <= TITLE_CONFIG["max_lines"] else 0
    score += 0.5 if len(text) <= TITLE_CONFIG["max_length"] else 0
    score += TITLE_CONFIG["above_image_bonus"] if above else TITLE_CONFIG["below_image_penalty"]

    return score >= TITLE_CONFIG["min_score"]


def handle_image(
    img: tuple,
    img_index: int,
    page: fitz.Page,
    source_file: str,
    image_save_dir: Path,
) -> Optional[ImmunoImages]:
    """
    Extract a single image from a page and find its associated caption.

    Adapted from Tesla image_handler.handle_image(), with:
    - Raised thresholds: 100×100px minimum (vs 34px in Tesla)
    - Added minimum file size filter
    - Figure caption regex detection
    - source_file metadata

    Returns ImmunoImages or None if image should be skipped.
    """
    xref = img[0]
    base_image = page.parent.extract_image(xref)

    width = base_image["width"]
    height = base_image["height"]
    size_bytes = len(base_image["image"])

    # Skip small decorative images and icons
    if width < IMAGE_MIN_WIDTH or height < IMAGE_MIN_HEIGHT:
        return None
    if size_bytes < IMAGE_MIN_BYTES:
        return None

    # Save image to disk
    image_name = f"page{page.number + 1}_img{img_index + 1}.{base_image['ext']}"
    image_path = image_save_dir / image_name
    image_save_dir.mkdir(parents=True, exist_ok=True)
    with open(image_path, "wb") as f:
        f.write(base_image["image"])

    # Get image bounding box on the page
    try:
        img_rect = page.get_image_bbox(img)
    except Exception:
        img_rect = fitz.Rect(0, 0, width, height)

    # Expand search area around the image to find nearby text
    expanded = img_rect + (0, -200, 0, img_rect.height * 3)
    expanded[3] = min(expanded[3], page.rect[3] - PAGE_FOOTER_CLIP)
    expanded = expanded.intersect(page.rect)

    # Find related text blocks and identify captions
    title_blocks = []
    has_caption = False
    for block in page.get_text("blocks"):
        block_rect = fitz.Rect(block[:4])
        if not block_rect.intersects(expanded):
            continue
        above = block_rect.y1 < img_rect.y0
        if _is_caption_candidate(page, block, img_rect.y0, above):
            title_blocks.append(block[4].strip())
            if FIGURE_CAPTION_PATTERN.search(block[4]):
                has_caption = True

    return ImmunoImages(
        page=page.number + 1,
        image_path=str(image_path),
        title="\n".join(title_blocks) if title_blocks else None,
        has_caption=has_caption,
        source_file=source_file,
    )


# %% [Cell 6: OCR helpers for image-based (scanned) PDFs]

def _is_image_based_page(page: fitz.Page) -> bool:
    """
    Return True if the page has no selectable text but contains images.
    This identifies scanned PDFs where each page is stored as a JPEG/PNG.
    """
    text_blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    images = page.get_images(full=True)
    return len(text_blocks) == 0 and len(images) > 0


def _ocr_page(page: fitz.Page, clip: fitz.Rect = None, dpi: int = 200) -> str:
    """
    Render a page (or clip region) to a raster image and extract text via OCR.

    Args:
        page:  PyMuPDF page object
        clip:  optional crop rectangle in PDF points (crops header/footer)
        dpi:   render resolution — 200 dpi is a good balance of speed vs accuracy

    Returns:
        OCR-extracted text string, or "" if pytesseract is not available.

    Note: requires  apt-get install -y tesseract-ocr tesseract-ocr-eng
    """
    if not _OCR_AVAILABLE:
        print("[Parser] WARNING: pytesseract not installed — cannot OCR image-based page.")
        print("  Fix: apt-get install -y tesseract-ocr tesseract-ocr-eng")
        print("       pip install pytesseract Pillow")
        return ""

    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pixmap = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
    img = _PILImage.open(io.BytesIO(pixmap.tobytes("png")))
    # --psm 6 = assume a single uniform block of text (good for textbook pages)
    text = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
    return text.strip()


# %% [Cell 7: Single-PDF parsing]

def parse_pdf(pdf_path: Path) -> List[Document]:
    """
    Extract text and images from a single PDF file.

    Returns a list of LangChain Documents, one per page (after filtering),
    each with full immunology-specific metadata.

    Metadata schema per Document:
    {
      "unique_id":     MD5 hash of page text content,
      "source_file":   filename (e.g. "JanewaysImmunobiologyBiology10thEdition.pdf"),
      "doc_type":      "textbook" or "paper" (detected from layout),
      "chapter":       "Chapter N" (detected from headings),
      "page":          1-indexed page number,
      "has_figure_caption": bool,
      "images_info":   [list of ImmunoImages dicts],
    }
    """
    pdf_path = Path(pdf_path)
    source_file = pdf_path.name

    # Determine doc_type from layout of first content page
    pdf = fitz.open(str(pdf_path))
    total_pages = len(pdf)

    # Setup image output directory
    doc_stem = pdf_path.stem.replace(" ", "_")[:50]
    image_dir = Path(constant.processed_dir) / doc_stem / "images"
    text_dir  = Path(constant.processed_dir) / doc_stem / "text"
    image_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    # Detect doc_type from a sample page
    sample_page_idx = min(50, total_pages - 1)
    layout_type = detect_layout(pdf.load_page(sample_page_idx))
    doc_type = "paper" if layout_type == "double" else "textbook"

    raw_docs = []
    current_chapter = "Chapter 1"  # default until detected
    empty_pages = 0

    print(f"[Parser] Parsing '{source_file}' ({total_pages} pages, type={doc_type})...")

    for page_idx in tqdm(range(total_pages), desc=f"Parsing {source_file[:30]}"):
        # Skip front/back matter
        if page_idx < MIN_CONTENT_PAGE or page_idx > MAX_CONTENT_PAGE:
            continue

        page = pdf.load_page(page_idx)

        # Crop out header and footer
        crop = fitz.Rect(
            0,
            PAGE_HEADER_CLIP,
            page.rect.width,
            page.rect.height - PAGE_FOOTER_CLIP,
        )
        text = page.get_text(clip=crop).strip()

        # OCR fallback: if no selectable text but page is image-based (scanned PDF)
        if not text and _is_image_based_page(page):
            text = _ocr_page(page, clip=crop)

        if not text:
            empty_pages += 1
            continue

        # Try to detect chapter from this page's text
        current_chapter = detect_chapter(text, current_chapter)

        # Extract images
        images_on_page = page.get_images(full=True)
        images_info = []
        has_figure_caption = False
        for img_idx, img in enumerate(images_on_page):
            immuno_image = handle_image(img, img_idx, page, source_file, image_dir)
            if immuno_image:
                img_dict = json.loads(immuno_image.model_dump_json())
                images_info.append(img_dict)
                if immuno_image.has_caption:
                    has_figure_caption = True

        # Build metadata
        unique_id = hashlib.md5(text.encode("utf-8")).hexdigest()
        metadata = {
            "unique_id":          unique_id,
            "source_file":        source_file,
            "doc_type":           doc_type,
            "chapter":            current_chapter,
            "page":               page_idx + 1,
            "has_figure_caption": has_figure_caption,
            "images_info":        images_info,
        }

        doc = Document(page_content=text, metadata=metadata)
        raw_docs.append(doc)

        # Also save the raw page text to disk as JSON
        page_json_path = text_dir / f"page_{page_idx + 1:04d}.json"
        with open(page_json_path, "w", encoding="utf-8") as f:
            json.dump({"page_content": text, "metadata": metadata}, f, ensure_ascii=False, indent=2)

    pdf.close()
    print(f"[Parser] '{source_file}': {len(raw_docs)} content pages, "
          f"{empty_pages} empty pages, {sum(len(d.metadata['images_info']) for d in raw_docs)} images")
    return raw_docs


# %% [Cell 7: Multi-PDF batch parsing]

def load_all_pdfs(raw_dir=None) -> Tuple[List[Document], dict]:
    """
    Discover and parse all PDFs.

    Args:
        raw_dir: either
          - a directory path (str/Path) → discovers all *.pdf inside it, OR
          - a list of PDF paths         → parses exactly those files
          - None                        → uses constant.raw_dir

    Returns:
        (all_docs, extraction_report) where extraction_report is a dict
        containing per-document statistics.

    This is the primary entry point for build_index.py.
    Multi-document design: single-file paths are NEVER hardcoded.
    """
    # Accept a pre-discovered list (e.g. from build_index.py --pdf flag)
    if isinstance(raw_dir, list):
        pdf_paths = [str(p) for p in raw_dir]
    else:
        if raw_dir is None:
            raw_dir = constant.raw_dir
        pdf_paths = sorted(glob(os.path.join(str(raw_dir), "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {raw_dir}")

    print(f"[Parser] Found {len(pdf_paths)} PDF(s) in {raw_dir}:")
    for p in pdf_paths:
        print(f"  - {Path(p).name}")

    all_docs = []
    extraction_report = {
        "generated_at": datetime.now().isoformat(),
        "documents": {},
    }

    for pdf_path in pdf_paths:
        docs = parse_pdf(Path(pdf_path))
        all_docs.extend(docs)

        doc_stem = Path(pdf_path).stem.replace(" ", "_")[:50]
        extraction_report["documents"][Path(pdf_path).name] = {
            "page_count":  len(docs),
            "image_count": sum(len(d.metadata["images_info"]) for d in docs),
            "doc_type":    docs[0].metadata["doc_type"] if docs else "unknown",
            "chapters":    list({d.metadata["chapter"] for d in docs}),
        }

    # Save extraction report
    report_path = Path(constant.processed_dir) / "extraction_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(extraction_report, f, ensure_ascii=False, indent=2)
    print(f"[Parser] Extraction report saved to {report_path}")

    return all_docs, extraction_report


# %% [Cell 8: CLI entry point]

def main():
    parser = argparse.ArgumentParser(description="ImmunoBiology RAG PDF Parser")
    parser.add_argument(
        "--inspect", action="store_true",
        help="Run layout inspection only (sample pages 1, 50, 100, 200)"
    )
    parser.add_argument(
        "--pdf", type=str, default=None,
        help="Parse a specific PDF file (default: all PDFs in data/raw/)"
    )
    args = parser.parse_args()

    if args.inspect:
        raw_dir = constant.raw_dir
        pdf_paths = sorted(glob(os.path.join(raw_dir, "*.pdf")))
        if not pdf_paths:
            print(f"No PDFs found in {raw_dir}")
            return
        for pdf_path in pdf_paths:
            print(f"\nInspecting {pdf_path}...")
            inspect_pdf_layout(Path(pdf_path))
    elif args.pdf:
        docs = parse_pdf(Path(args.pdf))
        print(f"Parsed {len(docs)} pages from {args.pdf}")
    else:
        all_docs, report = load_all_pdfs()
        print(f"\nTotal documents parsed: {len(all_docs)}")
        print(f"Extraction report: {json.dumps(report, indent=2)}")


if __name__ == "__main__":
    main()
