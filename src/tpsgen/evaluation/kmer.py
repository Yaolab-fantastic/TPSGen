from __future__ import annotations

from collections import Counter


def kmer_frequencies(sequence: str, k: int = 4) -> dict[str, float]:
    if len(sequence) < k:
        return {}
    counts = Counter(sequence[i : i + k] for i in range(len(sequence) - k + 1))
    total = sum(counts.values())
    return {kmer: count / total for kmer, count in counts.items()}

