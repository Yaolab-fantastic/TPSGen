from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tpsgen.io.schema import SequenceRecord


def spaced_kmers(sequence: str, k: int = 6) -> str:
    sequence = sequence.upper()
    if len(sequence) < k:
        raise ValueError(f"DNABERT requires a sequence of at least {k} bp.")
    return " ".join(sequence[index : index + k] for index in range(len(sequence) - k + 1))


class DNABERTTomatoAdapter:
    """Run the bundled tomato fine-tuned DNABERT and aggregate attention."""

    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        try:
            from transformers import BertForSequenceClassification, BertTokenizer
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "DNABERT inference requires the optional dependency extra: "
                "pip install tpsgen[dnabert]"
            ) from error
        self.model_dir = Path(model_dir)
        if not (self.model_dir / "pytorch_model.bin").exists():
            raise FileNotFoundError(f"DNABERT checkpoint not found in {self.model_dir}")
        self.device = torch.device(device)
        self.tokenizer = BertTokenizer.from_pretrained(
            str(self.model_dir), do_lower_case=False
        )
        self.model = BertForSequenceClassification.from_pretrained(
            str(self.model_dir), output_attentions=True
        ).to(self.device)
        self.model.eval()

    def predict(self, records: list[SequenceRecord]) -> list[dict[str, object]]:
        results = []
        for record in records:
            sequence = record.sequence.upper()
            if len(sequence) != 165 or set(sequence) - set("ACGT"):
                raise ValueError("DNABERT inference requires unambiguous 165-bp A/C/G/T sequences.")
            encoded = self.tokenizer(
                spaced_kmers(sequence), return_tensors="pt", add_special_tokens=True
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded)
                attention = torch.stack(output.attentions).mean(dim=(0, 1, 2))[0]
                probability = float(torch.softmax(output.logits, dim=-1)[0, 1].item())
            # The base-level evidence is the mean CLS attention over overlapping 6-mers.
            token_scores = attention[1 : 1 + len(sequence) - 5].detach().cpu().numpy()
            base_scores = np.zeros(len(sequence), dtype=float)
            counts = np.zeros(len(sequence), dtype=float)
            for index, score in enumerate(token_scores):
                base_scores[index : index + 6] += score
                counts[index : index + 6] += 1
            base_scores /= np.maximum(counts, 1)
            top = int(np.argmax(base_scores))
            results.append(
                {
                    "sequence_id": record.sequence_id,
                    "sequence_length": len(sequence),
                    "classification_probability": round(probability, 6),
                    "top_evidence_position": top,
                    "top_evidence_score": round(float(base_scores[top]), 6),
                    "base_evidence": ";".join(f"{value:.6g}" for value in base_scores),
                    "backend": "tomato_dnabert_6mer_attention_inference",
                }
            )
        return results
