"""Live status emit for the prepare-dataset node's on-canvas widget.

Pushes per-character counts + running total + chunk-pool capacity to
the frontend over PromptServer. Guarded so the node still loads if
PromptServer isn't available.
"""

import logging

logger = logging.getLogger(__name__)

# Guard the import so the node still loads if the API moves around.
try:
    from server import PromptServer  # type: ignore
except Exception:
    PromptServer = None  # type: ignore


def emit_prepper_status(
    node_id,
    char_counts: dict[str, int],
    total_clips: int,
    max_samples: int,
    pool_remaining: int = 0,
    pool_total: int = 0,
) -> None:
    """Send the current character counts + running total to the frontend so
    the node UI can update live. No-op if PromptServer isn't available or
    node_id wasn't provided.

    `total` on the panel tracks APPEARANCES (sum of per-character counts),
    not clip count — that's what max_samples caps. `clips` and
    `chunk pool` show separately so the operator can see capacity usage,
    dataset size, and how much of the source pool remains to search."""
    if PromptServer is None or node_id is None:
        return
    lines = []
    for name in sorted(char_counts):
        lines.append(f"{name}: {char_counts[name]}")
    total_appearances = sum(char_counts.values())
    if max_samples > 0:
        lines.append(f"total: {total_appearances}/{max_samples}")
    else:
        lines.append(f"total: {total_appearances}")
    # Blank line visually separates the per-char + total block from the
    # dataset/pool counters below.
    lines.append("")
    lines.append(f"clips: {total_clips}")
    if pool_total > 0:
        lines.append(f"chunk pool: {pool_remaining}/{pool_total}")
    try:
        PromptServer.instance.send_sync("rs.prepper.status", {
            "node_id": str(node_id),
            "text": "\n".join(lines),
        })
    except Exception:
        pass
