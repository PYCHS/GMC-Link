"""Per-track Z time-series cache: track_id → frame_id → metric depth (meters)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def save_depth_cache(data: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        str(t): {str(f): float(z) for f, z in fmap.items()}
        for t, fmap in data.items()
    }
    path.write_text(json.dumps(serialisable))


@dataclass
class DepthCache:
    table: dict[str, dict[str, float]]

    @classmethod
    def load(cls, path: Path | str) -> "DepthCache":
        return cls(json.loads(Path(path).read_text()))

    def lookup(self, track_id, frame_id) -> Optional[float]:
        per_track = self.table.get(str(track_id))
        if per_track is None:
            return None
        return per_track.get(str(frame_id))
