from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio import Align

from tomato_promoter_designer.io.csv import write_dict_rows
from tomato_promoter_designer.io.schema import LegacyLabeledSequence, LegacyMotifRecord


@dataclass(slots=True)
class DNABERTDataset:
    sequences: list[str]
    labels: list[int]


def kmer2seq(kmers: str) -> str:
    kmer_list = kmers.split()
    if not kmer_list:
        return ""
    bases = [kmer[0] for kmer in kmer_list[:-1]]
    bases.append(kmer_list[-1])
    return "".join(bases)


def read_dnabert_dev_tsv(path: str | Path) -> DNABERTDataset:
    sequences: list[str] = []
    labels: list[int] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            kmer_sequence = row.get("sequence")
            label = row.get("label")
            if kmer_sequence is None or label is None:
                raise ValueError("DNABERT dev.tsv must contain 'sequence' and 'label' columns.")
            sequences.append(kmer2seq(kmer_sequence.strip()))
            labels.append(int(label))
    if not sequences:
        raise ValueError(f"No rows found in {path}")
    return DNABERTDataset(sequences=sequences, labels=labels)


def validate_dnabert_resources(
    dev_tsv_path: str | Path,
    attention_scores_path: str | Path,
    prediction_path: str | Path | None = None,
    expected_length: int = 165,
) -> dict[str, object]:
    """Validate that DNABERT result arrays match one k-mer dataset.

    This checks file compatibility only. It does not establish that the
    upstream dataset is tomato-specific or that the arrays are independent
    evaluation data.
    """
    dataset = read_dnabert_dev_tsv(dev_tsv_path)
    num_sequences = len(dataset.sequences)
    attention = np.load(Path(attention_scores_path), mmap_mode="r")
    if attention.ndim != 2 or attention.shape[0] != num_sequences:
        raise ValueError(
            "DNABERT attention rows do not match dev.tsv: "
            f"{attention.shape} vs {num_sequences}"
        )
    lengths = {len(sequence) for sequence in dataset.sequences}
    if lengths != {expected_length}:
        raise ValueError(
            f"DNABERT dev.tsv must reconstruct {expected_length}-bp sequences; found {sorted(lengths)}"
        )
    prediction_shape = None
    if prediction_path is not None:
        predictions = np.load(Path(prediction_path), mmap_mode="r")
        if predictions.ndim != 1 or predictions.shape[0] != num_sequences:
            raise ValueError(
                "DNABERT prediction rows do not match dev.tsv: "
                f"{predictions.shape} vs {num_sequences}"
            )
        prediction_shape = list(predictions.shape)
    return {
        "dev_tsv": str(dev_tsv_path),
        "attention_shape": list(attention.shape),
        "prediction_shape": prediction_shape,
        "num_sequences": num_sequences,
        "sequence_lengths": sorted(lengths),
        "positive_sequences": sum(label == 1 for label in dataset.labels),
        "negative_sequences": sum(label == 0 for label in dataset.labels),
        "tomato_specificity_verified": False,
    }


def contiguous_regions(condition: np.ndarray, len_threshold: int = 5) -> list[tuple[int, int]]:
    if condition.size == 0:
        return []
    delta = np.diff(condition.astype(int))
    indices = np.nonzero(delta)[0] + 1
    if condition[0]:
        indices = np.r_[0, indices]
    if condition[-1]:
        indices = np.r_[indices, condition.size]
    pairs = indices.reshape(-1, 2)
    return [
        (int(start), int(end))
        for start, end in pairs
        if int(end) - int(start) >= len_threshold
    ]


def find_high_attention(score: np.ndarray, min_len: int = 5) -> list[tuple[int, int]]:
    cond1 = score > float(np.mean(score))
    cond2 = score > 10 * float(np.min(score))
    cond = np.logical_and(cond1, cond2)
    return contiguous_regions(cond, len_threshold=min_len)


def _count_motif_instances(
    sequences: list[str],
    motif: str,
    allow_multi_match: bool = False,
) -> int:
    count = 0
    for sequence in sequences:
        if allow_multi_match:
            start = 0
            while True:
                index = sequence.find(motif, start)
                if index == -1:
                    break
                count += 1
                start = index + 1
        else:
            count += int(motif in sequence)
    return count


def _count_presence_by_length(
    sequences: list[str],
    motifs: list[str],
) -> dict[str, int]:
    motif_sets_by_length: dict[int, set[str]] = {}
    counts = {motif: 0 for motif in motifs}
    for motif in motifs:
        motif_sets_by_length.setdefault(len(motif), set()).add(motif)

    for sequence in sequences:
        seen: set[str] = set()
        for motif_length, motif_set in motif_sets_by_length.items():
            if motif_length > len(sequence):
                continue
            for start in range(len(sequence) - motif_length + 1):
                substring = sequence[start : start + motif_length]
                if substring in motif_set:
                    seen.add(substring)
        for motif in seen:
            counts[motif] += 1
    return counts


def _merge_similar_motifs(
    motif_map: dict[str, dict[str, list[object]]],
    min_len: int,
) -> dict[str, dict[str, list[object]]]:
    aligner = Align.PairwiseAligner()
    aligner.internal_gap_score = -10000.0
    merged: dict[str, dict[str, list[object]]] = {}
    for motif in sorted(motif_map, key=len):
        matching_groups: list[tuple[str, object]] = []
        for canonical in merged:
            threshold = max(min_len - 1, 0.5 * min(len(motif), len(canonical)))
            alignment = aligner.align(motif, canonical)[0]
            if alignment.score >= threshold:
                matching_groups.append((canonical, alignment))

        if not matching_groups:
            merged[motif] = {
                "member_motifs": [motif],
                "seq_idx": list(motif_map[motif]["seq_idx"]),
                "atten_region_pos": list(motif_map[motif]["atten_region_pos"]),
            }
            continue

        best_alignment = max(
            (group[1] for group in matching_groups),
            key=lambda alignment: alignment.score,
        )
        # Preserve the retained implementation's object-equality comparison.
        # Despite its align_all_ties name, it did not compare numeric scores.
        for canonical, alignment in matching_groups:
            if alignment != best_alignment:
                continue
            merged[canonical]["member_motifs"].append(motif)
            query_start, query_end = alignment.aligned[0][0]
            key_start, key_end = alignment.aligned[1][0]
            left_offset = int(query_start - key_start)
            if query_end <= len(motif) and key_end == len(canonical):
                right_offset = len(motif) - int(query_end)
            elif query_end == len(motif) and key_end < len(canonical):
                right_offset = int(key_end) - len(canonical)
            else:
                right_offset = len(motif) - int(query_end)
            merged[canonical]["seq_idx"].extend(motif_map[motif]["seq_idx"])
            for start, end in motif_map[motif]["atten_region_pos"]:
                merged[canonical]["atten_region_pos"].append(
                    (int(start) + left_offset, int(end) - right_offset)
                )
    return merged


def _count_group_presence_by_length(
    sequences: list[str],
    motif_groups: dict[str, dict[str, list[object]]],
) -> dict[str, int]:
    members_by_length: dict[int, dict[str, str]] = {}
    counts = {canonical: 0 for canonical in motif_groups}
    for canonical, group in motif_groups.items():
        for member in group["member_motifs"]:
            members_by_length.setdefault(len(member), {})[member] = canonical

    for sequence in sequences:
        seen_groups: set[str] = set()
        for motif_length, member_to_group in members_by_length.items():
            if motif_length > len(sequence):
                continue
            for start in range(len(sequence) - motif_length + 1):
                substring = sequence[start : start + motif_length]
                canonical = member_to_group.get(substring)
                if canonical is not None:
                    seen_groups.add(canonical)
        for canonical in seen_groups:
            counts[canonical] += 1
    return counts


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _hypergeom_sf(x_minus_one: int, population: int, success_population: int, draws: int) -> float:
    lower = max(0, success_population + draws - population)
    upper = min(draws, success_population)
    start = max(lower, x_minus_one + 1)
    if start > upper:
        return 1.0

    log_denom = _log_comb(population, draws)
    log_terms = []
    for observed in range(start, upper + 1):
        log_num = _log_comb(success_population, observed) + _log_comb(
            population - success_population,
            draws - observed,
        )
        log_terms.append(log_num - log_denom)

    max_log = max(log_terms)
    total = sum(math.exp(term - max_log) for term in log_terms)
    return min(1.0, math.exp(max_log) * total)


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running = 1.0
    total = len(p_values)
    for rank, (index, p_value) in enumerate(reversed(indexed), start=1):
        bh_value = min(1.0, p_value * total / (total - rank + 1))
        running = min(running, bh_value)
        adjusted[index] = running
    return adjusted


def _window_region(start: int, end: int, window_size: int, sequence_length: int) -> tuple[int, int] | None:
    attention_length = end - start
    if attention_length > window_size:
        return None
    spare = window_size - attention_length
    left = spare // 2
    right = spare - left
    new_start = start - left
    new_end = end + right
    if new_start < 0 or new_end >= sequence_length:
        return None
    return new_start, new_end


def analyze_attention_dataset(
    dataset: DNABERTDataset,
    attention_scores: np.ndarray,
    window_size: int = 24,
    min_len: int = 5,
    p_value_cutoff: float = 0.005,
    min_n_motif: int = 3,
) -> tuple[list[LegacyMotifRecord], list[LegacyLabeledSequence], dict[str, object]]:
    positive_sequences = [seq for seq, label in zip(dataset.sequences, dataset.labels) if label == 1]
    negative_sequences = [seq for seq, label in zip(dataset.sequences, dataset.labels) if label == 0]
    positive_indices = [idx for idx, label in enumerate(dataset.labels) if label == 1]

    if attention_scores.shape[0] != len(dataset.sequences):
        raise ValueError(
            "Attention score row count does not match dev.tsv row count: "
            f"{attention_scores.shape[0]} vs {len(dataset.sequences)}. "
            "Make sure the attention export and dev.tsv come from the same DNABERT run."
        )

    motif_map: dict[str, dict[str, list[object]]] = {}
    for positive_offset, dataset_index in enumerate(positive_indices):
        sequence = dataset.sequences[dataset_index]
        score = np.asarray(attention_scores[dataset_index])[: len(sequence)]
        for start, end in find_high_attention(score, min_len=min_len):
            motif = sequence[start:end]
            motif_entry = motif_map.setdefault(
                motif,
                {"seq_idx": [], "atten_region_pos": []},
            )
            motif_entry["seq_idx"].append(positive_offset)
            motif_entry["atten_region_pos"].append((start, end))

    # Match the retained DNABERT workflow: test exact high-attention motifs for
    # enrichment first, then merge only motifs that pass the significance cutoff.
    exact_motifs = list(motif_map.keys())
    positive_presence = (
        _count_presence_by_length(positive_sequences, exact_motifs) if exact_motifs else {}
    )
    negative_presence = (
        _count_presence_by_length(negative_sequences, exact_motifs) if exact_motifs else {}
    )
    p_values: list[float] = []
    enrichment_ratios: list[float] = []
    for motif in exact_motifs:
        positive_count = positive_presence[motif]
        negative_count = negative_presence.get(motif, 0)
        total_count = positive_count + negative_count
        p_values.append(
            _hypergeom_sf(
                positive_count - 1,
                population=len(positive_sequences) + len(negative_sequences),
                success_population=len(positive_sequences),
                draws=total_count,
            )
        )
        pos_rate = positive_count / max(len(positive_sequences), 1)
        neg_rate = negative_count / max(len(negative_sequences), 1)
        enrichment_ratios.append(round(pos_rate / max(neg_rate, 1e-9), 6))

    adjusted_p_values = _benjamini_hochberg(p_values)
    exact_statistics = {
        motif: (float(adjusted_p_value), enrichment_ratio)
        for motif, adjusted_p_value, enrichment_ratio in zip(
            exact_motifs, adjusted_p_values, enrichment_ratios
        )
    }
    significant_motif_map = {
        motif: motif_map[motif]
        for motif in exact_motifs
        if exact_statistics[motif][0] < p_value_cutoff
    }
    motif_groups = _merge_similar_motifs(significant_motif_map, min_len=min_len)

    candidate_records: list[LegacyMotifRecord] = []
    for motif, entry in motif_groups.items():
        member_statistics = [exact_statistics[member] for member in entry["member_motifs"]]
        adjusted_p_value = min(statistic[0] for statistic in member_statistics)
        enrichment_ratio = max(statistic[1] for statistic in member_statistics)
        window_positions: list[tuple[int, int]] = []
        kept_indices: list[int] = []
        for positive_offset, region in zip(entry["seq_idx"], entry["atten_region_pos"]):
            sequence = positive_sequences[positive_offset]
            windowed = _window_region(region[0], region[1], window_size=window_size, sequence_length=len(sequence))
            if windowed is None:
                continue
            kept_indices.append(int(positive_offset))
            window_positions.append(windowed)

        if len(kept_indices) < min_n_motif:
            continue

        candidate_records.append(
            LegacyMotifRecord(
                motif=motif,
                num_instances=len(kept_indices),
                enrichment_ratio=enrichment_ratio,
                adjusted_p_value=round(float(adjusted_p_value), 8),
                indices=kept_indices,
                window_positions=window_positions,
            )
        )

    motif_records = candidate_records

    motif_positions_by_positive_index: dict[int, list[tuple[int, int]]] = {}
    for record in motif_records:
        for positive_offset, window in zip(record.indices, record.window_positions):
            motif_positions_by_positive_index.setdefault(int(positive_offset), []).append(window)

    labeled_sequences: list[LegacyLabeledSequence] = []
    for positive_offset, sequence in enumerate(positive_sequences):
        masked = ["M"] * len(sequence)
        for start, end in motif_positions_by_positive_index.get(positive_offset, []):
            masked[start:end] = list(sequence[start:end])
        labeled_sequences.append(
            LegacyLabeledSequence(
                sequence_id=positive_indices[positive_offset],
                label=1,
                original_sequence=sequence,
                labeled_sequence="".join(masked),
            )
        )

    metadata = {
        "num_total_sequences": len(dataset.sequences),
        "num_positive_sequences": len(positive_sequences),
        "num_negative_sequences": len(negative_sequences),
        "num_extracted_exact_motifs": len(motif_map),
        "num_tested_exact_motifs": len(exact_motifs),
        "num_significant_exact_motifs": len(significant_motif_map),
        "num_merged_motif_groups": len(motif_groups),
        "num_candidate_motifs_after_windowing": len(candidate_records),
        "num_retained_motifs": len(motif_records),
        "window_size": window_size,
        "min_len": min_len,
        "p_value_cutoff": p_value_cutoff,
        "min_n_motif": min_n_motif,
        "used_fallback_ranking": False,
    }
    return motif_records, labeled_sequences, metadata


def run_dnabert_legacy_annotation(
    dev_tsv_path: str | Path,
    attention_scores_path: str | Path,
    output_dir: str | Path,
    window_size: int = 24,
    min_len: int = 5,
    p_value_cutoff: float = 0.005,
    min_n_motif: int = 3,
) -> dict[str, object]:
    dataset = read_dnabert_dev_tsv(dev_tsv_path)
    attention_scores = np.load(Path(attention_scores_path))
    motif_records, labeled_sequences, metadata = analyze_attention_dataset(
        dataset,
        attention_scores,
        window_size=window_size,
        min_len=min_len,
        p_value_cutoff=p_value_cutoff,
        min_n_motif=min_n_motif,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "motif_summary.csv"
    labeled_path = output_dir / "processed_sequences.csv"
    metadata_path = output_dir / "run_metadata.json"

    if motif_records:
        write_dict_rows([record.to_dict() for record in motif_records], summary_path)
    else:
        write_dict_rows(
            [
                {
                    "motif": "none",
                    "num_instances": 0,
                    "enrichment_ratio": 0.0,
                    "adjusted_p_value": 1.0,
                    "indices": "[]",
                    "window_positions": "[]",
                }
            ],
            summary_path,
        )

    if labeled_sequences:
        write_dict_rows([record.to_dict() for record in labeled_sequences], labeled_path)
    else:
        write_dict_rows(
            [
                {
                    "sequence_id": -1,
                    "label": 1,
                    "original_sequence": "",
                    "labeled_sequence": "",
                }
            ],
            labeled_path,
        )

    metadata["summary_path"] = str(summary_path)
    metadata["labeled_path"] = str(labeled_path)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    metadata["metadata_path"] = str(metadata_path)
    return metadata
