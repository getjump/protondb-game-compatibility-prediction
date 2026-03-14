"""Relabeling heuristics for noisy tinkering/works_oob boundary.

Phase 8 experiment showed Strict relabeling of 51% tinkering→oob
(reports without effort markers) gives +0.010 F1.

Applied only at training time, never modifies the database.
"""

from __future__ import annotations

import logging
import re
import sqlite3

import numpy as np

logger = logging.getLogger(__name__)

# Effort markers: if ANY match, keep as tinkering.
# If NONE match AND no customization/launch-flag notes → relabel to works_oob.
EFFORT_STRICT = re.compile(
    r"(protontricks|winetricks|winedlloverrides|gamescope|mangohud|lutris"
    r"|launch\s*option|STEAM_COMPAT|PROTON_|DXVK_|VKD3D_|MESA_"
    r"|flatpak|\.ini\b|\.cfg\b|\.conf\b|config\s*file"
    r"|[A-Z_]{3,}=[^\s]+"  # env vars
    r"|--[a-z]"  # CLI flags
    r"|workaround|trick|tweak|hack"
    r"|wined3d|d3d11|d3d9|dxvk|vkd3d|vulkan\s*render"
    r"|disabl|delet|remov|renam|mov[ei]"  # actions
    r"|controller\s*layout|remap|rebind"
    r"|terminal|command\s*line|bash|shell"
    r"|install\s*script|eac|easy\s*anti|battleye"
    r"|fix|patch|mod\b|mods\b"
    r"|swap|switch.*(?:render|mode|layout)"
    r")", re.IGNORECASE
)


def get_relabel_ids(conn: sqlite3.Connection) -> set[str]:
    """Get IDs of tinkering reports to relabel as works_oob.

    A tinkering report (verdict='yes', verdict_oob='no') is relabeled if:
    1. notes_customizations is empty
    2. notes_launch_flags (notes from launch options) is empty
    3. No effort markers found in concluding_notes + notes_verdict + notes_extra
    """
    rows = conn.execute("""
        SELECT id, concluding_notes, notes_verdict, notes_customizations,
               notes_extra, notes_launch_flags
        FROM reports
        WHERE verdict='yes' AND verdict_oob='no'
    """).fetchall()

    relabel = []
    for r in rows:
        # Has customization notes → keep as tinkering
        if r["notes_customizations"] and r["notes_customizations"].strip():
            continue
        # Has launch flags notes → keep as tinkering
        if r["notes_launch_flags"] and r["notes_launch_flags"].strip():
            continue

        all_text = " ".join(
            t for t in [r["concluding_notes"], r["notes_verdict"], r["notes_extra"]] if t
        ) or ""

        if not EFFORT_STRICT.search(all_text):
            relabel.append(r["id"])

    logger.info(
        "Relabeling: %d/%d tinkering reports → works_oob (%.1f%%)",
        len(relabel), len(rows), 100 * len(relabel) / len(rows) if rows else 0,
    )
    return set(relabel)


def apply_relabeling(
    y: np.ndarray,
    report_ids: list[str],
    relabel_ids: set[str],
) -> tuple[np.ndarray, int]:
    """Apply relabeling: tinkering(1) → works_oob(2) for specified report IDs.

    Returns:
        y_relabeled: copy of y with relabeled targets
        n_relabeled: number of samples actually relabeled
    """
    y_new = y.copy()
    n_relabeled = 0
    for i, rid in enumerate(report_ids):
        if rid in relabel_ids and y_new[i] == 1:
            y_new[i] = 2
            n_relabeled += 1
    logger.info("Applied relabeling: %d samples tinkering → works_oob", n_relabeled)
    return y_new, n_relabeled
