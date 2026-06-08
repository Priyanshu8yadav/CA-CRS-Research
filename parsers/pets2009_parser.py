"""
pets2009_parser.py
───────────────────
Parser for PETS 2009 crowd-tracking dataset.

Annotation format (confirmed from disk inspection)
──────────────────────────────────────────────────
XML files with the schema:

    <dataset name="PETS-S1L1-1">
      <frame number="0">
        <objectlist>
          <object id="1">
            <box h="105.14" w="51.95" xc="630.13" yc="310.69"/>
          </object>
          ...
        </objectlist>
      </frame>
      ...
    </dataset>

Each XML file corresponds to one scenario/level.
The parser emits one row per (frame × person) with:
  - frame_id       : frame number
  - trajectory_id  : object id (person track)
  - gt_count       : total persons visible in that frame
  - bbox_x/y/w/h   : bounding box (converted from xc,yc,w,h centre format)
  - image_id       : derived from XML filename stem, e.g. "PETS2009-S1L1-1"

The image_id and split follow the dataset filename convention.
All PETS2009 files are treated as a single "test" split
(no labelled train set exists for PETS2009).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree

from parsers.base_parser import BaseParser

logger = logging.getLogger(__name__)


class PETS2009Parser(BaseParser):
    """
    Parse all PETS2009 XML annotation files found under <root>/annotations/.

    Parameters
    ----------
    root : str | Path
        Path to the PETS-2009 data directory (contains annotations/ subfolder).
    exclude_cropped : bool
        If True (default), skip the *-cropped.xml variants and parse only
        the full-resolution annotation files.
    """

    def __init__(
        self,
        root: str | Path,
        exclude_cropped: bool = True,
    ) -> None:
        super().__init__(root, dataset_name="pets2009")
        self.exclude_cropped = exclude_cropped

    # ── Public API ──────────────────────────────────────────────────────────

    def parse(self) -> pd.DataFrame:
        ann_dir = self.root / "annotations"
        if not ann_dir.exists():
            raise FileNotFoundError(
                f"[{self.dataset_name}] Annotation directory not found: {ann_dir}"
            )

        xml_files = sorted(ann_dir.glob("*.xml"))
        if self.exclude_cropped:
            xml_files = [f for f in xml_files if "cropped" not in f.stem.lower()]

        if not xml_files:
            raise FileNotFoundError(
                f"[{self.dataset_name}] No XML files found in {ann_dir}"
            )

        logger.info(
            "[%s] Found %d XML annotation files", self.dataset_name, len(xml_files)
        )

        all_frames: list[pd.DataFrame] = []
        for xml_path in xml_files:
            df = self._parse_xml(xml_path)
            if df is not None:
                all_frames.append(df)

        if not all_frames:
            raise RuntimeError(
                f"[{self.dataset_name}] All XML files failed to parse."
            )

        combined = pd.concat(all_frames, ignore_index=True)
        combined = self.validate_schema(combined)
        self.sanity_check(combined)
        return combined

    # ── Private helpers ─────────────────────────────────────────────────────

    def _parse_xml(self, xml_path: Path) -> pd.DataFrame | None:
        """Parse a single PETS2009 XML annotation file."""
        try:
            tree = etree.parse(str(xml_path))
        except Exception as exc:
            logger.error(
                "[%s] XML parse error in %s: %s",
                self.dataset_name, xml_path.name, exc,
            )
            return None

        root_el   = tree.getroot()
        image_id  = xml_path.stem  # e.g. "PETS2009-S1L1-1"
        rows: list[dict] = []

        for frame_el in root_el.findall("frame"):
            frame_id = int(frame_el.get("number", -1))

            obj_list = frame_el.find("objectlist")
            if obj_list is None:
                continue

            objects = obj_list.findall("object")
            gt_count = len(objects)

            for obj_el in objects:
                traj_id = str(obj_el.get("id", ""))
                box_el  = obj_el.find("box")

                if box_el is None:
                    continue

                # Box is stored as centre (xc, yc) + half-dimensions
                xc = float(box_el.get("xc", "nan"))
                yc = float(box_el.get("yc", "nan"))
                bw = float(box_el.get("w",  "nan"))
                bh = float(box_el.get("h",  "nan"))

                # Convert to top-left corner
                bbox_x = xc - bw / 2.0
                bbox_y = yc - bh / 2.0

                rows.append({
                    "dataset":           self.dataset_name,
                    "split":             "test",   # PETS2009 has no train split
                    "image_id":          image_id,
                    "frame_id":          frame_id,
                    "gt_count":          gt_count,
                    "trajectory_id":     traj_id,
                    "cx":                xc,
                    "cy":                yc,
                    "bbox_x":            bbox_x,
                    "bbox_y":            bbox_y,
                    "bbox_w":            bw,
                    "bbox_h":            bh,
                    "density_available": False,
                })

        if not rows:
            logger.warning(
                "[%s] No valid frame annotations in %s",
                self.dataset_name, xml_path.name,
            )
            return None

        logger.info(
            "[%s] %s → %d frames, %d rows",
            self.dataset_name, xml_path.name,
            len({r["frame_id"] for r in rows}), len(rows),
        )
        return pd.DataFrame(rows)
