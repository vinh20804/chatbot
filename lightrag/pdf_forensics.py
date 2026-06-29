"""PDF forensics: detect decorative images (logos, templates) via XObject reuse.

Strategy:
  1. Parse PDF with pypdf to find reused Image XObjects across pages.
  2. Extract those XObjects' pixel data, compute perceptual hash.
  3. Return set of blacklist hashes used by MarkerPictureSerializer
     at Docling extraction time to skip matching images.

Fail-open: any exception returns empty blacklist; ingestion continues normally.
"""
from __future__ import annotations

import logging
import os
from io import BytesIO

logger = logging.getLogger(__name__)


def _xobject_to_pil(obj):
    """Convert PDF Image XObject to PIL.Image. Handle raw pixel fallback."""
    from PIL import Image

    data = obj.get_data()
    w = obj.get("/Width")
    h = obj.get("/Height")
    cs = obj.get("/ColorSpace")

    # Try standard (PNG/JPEG embedded)
    try:
        return Image.open(BytesIO(data))
    except Exception:
        pass

    # Fallback: raw pixel bytes
    if not (w and h):
        raise ValueError("No width/height for raw pixel data")

    cs_str = str(cs) if cs else ""
    if "/DeviceRGB" in cs_str:
        mode = "RGB"
    elif "/DeviceGray" in cs_str:
        mode = "L"
    elif "/DeviceCMYK" in cs_str:
        mode = "CMYK"
    else:
        if len(data) == w * h * 3:
            mode = "RGB"
        elif len(data) == w * h:
            mode = "L"
        elif len(data) == w * h * 4:
            mode = "CMYK"
        else:
            raise ValueError(f"Cannot infer mode for raw data (len={len(data)})")

    return Image.frombytes(mode, (w, h), data)


def get_decorative_image_hashes(
    pdf_path: str,
    reuse_threshold: float = 0.5,
    min_pages: int = 3,
) -> set[str]:
    """Detect decorative image XObjects (logos, templates) in PDF.

    Args:
        pdf_path: path to PDF file
        reuse_threshold: fraction of pages an XObject must appear on to count as decorative
        min_pages: minimum total pages required before filter activates (guard small PDFs)

    Returns:
        Set of perceptual hash hex strings. Empty on any error or small PDF.
    """
    if os.getenv("FILTER_DECORATIVE_IMAGES", "true").lower() in ("false", "0", "no"):
        logger.info("Decorative image filter disabled via env")
        return set()

    try:
        import pypdf
        import imagehash
    except ImportError as e:
        logger.warning(f"Missing dep for forensics: {e}. Filter disabled.")
        return set()

    try:
        reader = pypdf.PdfReader(pdf_path)
    except Exception as e:
        logger.warning(f"pypdf read failed for {pdf_path}: {e}")
        return set()

    total_pages = len(reader.pages)
    if total_pages < min_pages:
        logger.info(f"PDF has {total_pages} pages < min_pages={min_pages}, skip filter")
        return set()

    # Track XObjects by indirect reference idnum across pages
    obj_pages: dict[int, set[int]] = {}
    for page_idx, page in enumerate(reader.pages):
        try:
            resources = page.get("/Resources", {})
            if not resources:
                continue
            xobjects = resources.get("/XObject", {})
            if not xobjects:
                continue
            for _name, ref in xobjects.items():
                try:
                    idnum = ref.indirect_reference.idnum
                except AttributeError:
                    continue
                obj_pages.setdefault(idnum, set()).add(page_idx)
        except Exception as e:
            logger.debug(f"Skip page {page_idx} resource scan: {e}")
            continue

    decorative_idnums = {
        idnum
        for idnum, pages in obj_pages.items()
        if len(pages) / total_pages >= reuse_threshold
    }

    if not decorative_idnums:
        return set()

    # Extract hashes for each decorative XObject (one per idnum)
    hashes: set[str] = set()
    extracted_idnums: set[int] = set()

    for page in reader.pages:
        if len(extracted_idnums) == len(decorative_idnums):
            break
        try:
            xobjects = page.get("/Resources", {}).get("/XObject", {})
        except Exception:
            continue
        for _name, ref in xobjects.items():
            try:
                idnum = ref.indirect_reference.idnum
            except AttributeError:
                continue
            if idnum not in decorative_idnums or idnum in extracted_idnums:
                continue
            try:
                obj = ref.get_object()
                if str(obj.get("/Subtype", "")) != "/Image":
                    extracted_idnums.add(idnum)
                    continue
                pil_img = _xobject_to_pil(obj)
                h = str(imagehash.phash(pil_img))
                hashes.add(h)
                extracted_idnums.add(idnum)
            except Exception as e:
                logger.debug(f"Extract obj {idnum} failed: {e}")
                extracted_idnums.add(idnum)
                continue

    logger.info(
        f"PDF forensics {pdf_path}: {len(decorative_idnums)} decorative XObjects, "
        f"{len(hashes)} blacklist hashes"
    )
    return hashes


def is_decorative(pil_img, blacklist: set[str], distance_threshold: int = 5) -> bool:
    """Check if image matches any blacklisted decorative hash within threshold."""
    if not blacklist:
        return False
    try:
        import imagehash
    except ImportError:
        return False
    try:
        img_hash = imagehash.phash(pil_img)
        for blacklist_hex in blacklist:
            try:
                bl_hash = imagehash.hex_to_hash(blacklist_hex)
                if (img_hash - bl_hash) <= distance_threshold:
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False
