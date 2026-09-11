from __future__ import annotations


def build_expression_lookup(rows: list[dict[str, str]], gene_key: str = "Gene") -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        gene_id = row.get(gene_key)
        if gene_id:
            lookup[gene_id] = row
    return lookup

