from __future__ import annotations


def seq_to_kmers(sequence: str, k: int = 6) -> list[str]:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(sequence) < k:
        return []
    return [sequence[i : i + k] for i in range(len(sequence) - k + 1)]

