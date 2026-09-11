from __future__ import annotations

from pathlib import Path


def summarize_blast_outfmt6(path: str | Path) -> dict[str, int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    queries = set()
    for line in lines:
        queries.add(line.split("\t")[0])
    return {
        "num_hits": len(lines),
        "num_queries_with_hits": len(queries),
    }

