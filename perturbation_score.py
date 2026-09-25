from dataclasses import dataclass, asdict
from typing import Any, Optional, Sequence, Tuple

try:
    from .transcript_graph import TranscriptGraph
except ImportError:  # pragma: no cover - direct script execution fallback
    from transcript_graph import TranscriptGraph


ExonInterval = Tuple[str, int, int]
LabeledExonInterval = Tuple[str, int, int, str]


def _ensure_reference_indexes(graph: TranscriptGraph) -> None:
    if hasattr(graph, "_ensure_reference_indexes"):
        graph._ensure_reference_indexes()


@dataclass(frozen=True)
class PerturbationCounts:
    """Unweighted topology changes for one reconstructed fusion transcript."""

    novel_junctions: int = 0
    skipped_exons: int = 0
    inserted_exons: int = 0
    fusion_edges: int = 0

    @property
    def total(self) -> int:
        return (
            self.novel_junctions
            + self.skipped_exons
            + self.inserted_exons
            + self.fusion_edges
        )

    def weighted(self, a: float = 1.0, b: float = 1.0, c: float = 1.0, d: float = 1.0) -> float:
        return (
            a * self.novel_junctions
            + b * self.skipped_exons
            + c * self.inserted_exons
            + d * self.fusion_edges
        )

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _node_interval(graph: TranscriptGraph, node: Any) -> Optional[ExonInterval]:
    attrs = graph.nodes[node]
    try:
        return str(attrs["chrom"]), int(attrs["start"]), int(attrs["end"])
    except (KeyError, TypeError, ValueError):
        return None


def _exon_intervals(graph: TranscriptGraph, gene: str) -> set[ExonInterval]:
    _ensure_reference_indexes(graph)
    cached = graph.graph.get("_gene_exons")
    if cached is not None:
        return cached.get(gene, set())
    return {
        interval
        for node in graph.exon_nodes()
        if graph.nodes[node].get("gene") == gene
        for interval in [_node_interval(graph, node)]
        if interval is not None
    }


def _reference_junctions(graph: TranscriptGraph, gene: str) -> set[tuple[ExonInterval, ExonInterval]]:
    _ensure_reference_indexes(graph)
    cached = graph.graph.get("_gene_junctions")
    if cached is not None:
        return cached.get(gene, set())
    junctions = set()
    for source, target, attrs in graph.edges(data=True):
        if attrs.get("kind") != "splice_junction":
            continue
        if graph.nodes[source].get("gene") != gene or graph.nodes[target].get("gene") != gene:
            continue
        source_interval = _node_interval(graph, source)
        target_interval = _node_interval(graph, target)
        if source_interval is not None and target_interval is not None:
            junctions.add((source_interval, target_interval))
    return junctions


def build_fusion_transcript_graph(
    exon_intervals: Sequence[ExonInterval | LabeledExonInterval],
    left_gene: str,
    right_gene: str,
    partner_switch: Optional[int] = None,
) -> TranscriptGraph:
    """Build a graph from exon intervals ordered along one complete fusion read.

    Each interval may be ``(chrom, start, end, partner_gene)``. Alternatively,
    pass three-field intervals and ``partner_switch`` as the index of the first
    exon belonging to ``right_gene``.
    """
    if not left_gene or not right_gene or left_gene == right_gene:
        raise ValueError("A fusion must have two distinct partner genes")
    if len(exon_intervals) < 2:
        raise ValueError("A fusion transcript must contain at least two exons")

    graph = TranscriptGraph()
    graph.add_node("fusion_transcript", kind="transcript")
    if partner_switch is not None and not 1 <= partner_switch < len(exon_intervals):
        raise ValueError("partner_switch must point to an internal exon boundary")

    labels = []
    for index, interval in enumerate(exon_intervals):
        if len(interval) == 4:
            label = interval[3]
            if label not in (left_gene, right_gene):
                raise ValueError("Every exon partner label must be one of the two fusion genes")
        elif len(interval) == 3 and partner_switch is not None:
            label = left_gene if index < partner_switch else right_gene
        else:
            raise ValueError("Use labeled exon intervals or provide partner_switch")
        labels.append(label)

    if labels.count(left_gene) == 0 or labels.count(right_gene) == 0:
        raise ValueError("Fusion transcript must contain exons from both partner genes")
    if sum(first != second for first, second in zip(labels, labels[1:])) != 1:
        raise ValueError("Fusion exon intervals must contain one partner transition")

    previous = None
    previous_gene = None
    for index, interval in enumerate(exon_intervals):
        chrom, start, end = interval[:3]
        gene = labels[index]
        if start > end:
            raise ValueError(f"Invalid exon interval: {interval}")
        node = f"fusion_exon_{index}"
        graph.add_node(
            node,
            kind="exon",
            chrom=str(chrom),
            start=int(start),
            end=int(end),
            gene=gene,
            transcript="fusion_transcript",
        )
        graph.add_edge("fusion_transcript", node, kind="contains")
        if previous is not None:
            edge_kind = "fusion_edge" if gene != previous_gene else "splice_junction"
            graph.add_edge(previous, node, kind=edge_kind)
        previous = node
        previous_gene = gene

    return graph


def _observed_junctions(graph: TranscriptGraph, gene: str):
    junctions = set()
    for source, target, attrs in graph.edges(data=True):
        if attrs.get("kind") != "splice_junction":
            continue
        if graph.nodes[source].get("gene") != gene or graph.nodes[target].get("gene") != gene:
            continue
        source_interval = _node_interval(graph, source)
        target_interval = _node_interval(graph, target)
        if source_interval is not None and target_interval is not None:
            junctions.add((source_interval, target_interval))
    return junctions


def _breakpoint_transcript_path(
    reference_graph: TranscriptGraph,
    gene: str,
    chrom: str,
    position: int,
    keep_prefix: bool,
    anchor_interval: Optional[ExonInterval] = None,
) -> list[ExonInterval]:
    """Return the retained transcript side ending or starting at a breakpoint exon."""
    _ensure_reference_indexes(reference_graph)
    transcript_index = reference_graph.graph.get("_gene_transcripts")
    if transcript_index is None:
        transcript_index = {}
        for transcript, attrs in reference_graph.nodes(data=True):
            if attrs.get("kind") == "transcript":
                transcript_index.setdefault(attrs.get("gene"), []).append(transcript)
        reference_graph.graph["_gene_transcripts"] = transcript_index

    candidates = []
    for transcript in transcript_index.get(gene, []):
        exons = [
            node for node in reference_graph.successors(transcript)
            if reference_graph.nodes[node].get("kind") == "exon"
        ]
        exons.sort(key=lambda node: (reference_graph.nodes[node].get("start", 0),
                                     reference_graph.nodes[node].get("end", 0)))
        hit_index = None
        for index, exon in enumerate(exons):
            exon_attrs = reference_graph.nodes[exon]
            exon_interval = _node_interval(reference_graph, exon)
            if anchor_interval is not None and exon_interval == anchor_interval:
                hit_index = index
                break
            if (anchor_interval is None
                    and str(exon_attrs.get("chrom")) == str(chrom)
                    and int(exon_attrs.get("start", 0)) <= position <= int(exon_attrs.get("end", 0))):
                hit_index = index
                break
        if hit_index is None:
            continue
        intervals = [_node_interval(reference_graph, exon) for exon in exons]
        intervals = [interval for interval in intervals if interval is not None]
        if keep_prefix:
            retained = intervals[:hit_index + 1]
        else:
            retained = intervals[hit_index:]
        if retained:
            candidates.append(retained)

    if not candidates:
        raise ValueError(f"No transcript exon overlaps {gene} at {chrom}:{position}")
    # Prefer the transcript retaining the most exonic sequence around the breakpoint.
    return max(candidates, key=lambda path: (len(path), sum(end - start + 1 for _, start, end in path)))


def score_fusion_breakpoint(
    reference_graph: TranscriptGraph,
    left_gene: str,
    left_chrom: str,
    left_position: int,
    right_gene: str,
    right_chrom: str,
    right_position: int,
    genomic_index=None,
    **weights: float,
) -> tuple[PerturbationCounts, float]:
    """Score a two-partner fusion using only its consensus breakpoint.

    The breakpoint is interpreted as a retained prefix of the left partner and
    retained suffix of the right partner. This is a topology estimate, not a
    reconstructed transcript, and requires both breakpoints to overlap exons.
    """
    if left_gene == right_gene or not left_gene or not right_gene:
        raise ValueError("score_fusion_breakpoint requires two distinct partner genes")

    def nearest_exon(gene: str, chrom: str, position: int) -> ExonInterval:
        partner_exons = {
            interval
            for interval in _exon_intervals(reference_graph, gene)
            if interval[0] == str(chrom)
        }
        for window in (0, 500, 2000):
            candidates = [
                interval for interval in partner_exons
                if (interval[1] <= position <= interval[2])
                or min(abs(position - interval[1]), abs(position - interval[2])) <= window
            ]
            if candidates:
                return min(
                    candidates,
                    key=lambda interval: min(
                        abs(position - interval[1]),
                        abs(position - interval[2]),
                    ),
                )
        raise ValueError(f"No exon within 2000 bp of {chrom}:{position}")

    left_anchor = nearest_exon(left_gene, left_chrom, int(left_position))
    right_anchor = nearest_exon(right_gene, right_chrom, int(right_position))
    left_path = _breakpoint_transcript_path(
        reference_graph, left_gene, left_chrom, int(left_position),
        keep_prefix=True, anchor_interval=left_anchor,
    )
    right_path = _breakpoint_transcript_path(
        reference_graph, right_gene, right_chrom, int(right_position),
        keep_prefix=False, anchor_interval=right_anchor,
    )
    labeled_intervals = [
        (*interval, left_gene) for interval in left_path
    ] + [
        (*interval, right_gene) for interval in right_path
    ]
    fusion_graph = build_fusion_transcript_graph(labeled_intervals, left_gene, right_gene)
    return calculate_perturbation(
        reference_graph,
        fusion_graph,
        left_gene,
        right_gene,
        **weights,
    )

def calculate_perturbation(
    reference_graph: TranscriptGraph,
    fusion_graph: TranscriptGraph,
    left_gene: str,
    right_gene: str,
    *,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    d: float = 1.0,
) -> tuple[PerturbationCounts, float]:
    """Calculate partner-aware perturbation for one two-partner fusion graph.

    Exons are matched by ``(chromosome, start, end)`` rather than graph node ID,
    because reference exon IDs and reconstructed-read exon IDs differ.
    """
    if left_gene == right_gene or not left_gene or not right_gene:
        raise ValueError("calculate_perturbation requires two distinct partner genes")

    novel_junctions = 0
    skipped_exons = 0
    inserted_exons = 0
    for gene in (left_gene, right_gene):
        reference_exons = _exon_intervals(reference_graph, gene)
        observed_exons = {
            _node_interval(fusion_graph, node)
            for node in fusion_graph.exon_nodes()
            if fusion_graph.nodes[node].get("gene") == gene
        }
        observed_exons.discard(None)
        reference_junctions = _reference_junctions(reference_graph, gene)
        observed_junctions = _observed_junctions(fusion_graph, gene)
        novel_junctions += len(observed_junctions - reference_junctions)
        skipped_exons += len(reference_exons - observed_exons)
        inserted_exons += len(observed_exons - reference_exons)

    fusion_edges = sum(
        1
        for source, target, attrs in fusion_graph.edges(data=True)
        if attrs.get("kind") == "fusion_edge"
        and {fusion_graph.nodes[source].get("gene"), fusion_graph.nodes[target].get("gene")}
        == {left_gene, right_gene}
    )
    counts = PerturbationCounts(novel_junctions, skipped_exons, inserted_exons, fusion_edges)
    return counts, counts.weighted(a=a, b=b, c=c, d=d)


def score_fusion_transcript(
    reference_graph: TranscriptGraph,
    exon_intervals: Sequence[ExonInterval | LabeledExonInterval],
    left_gene: str,
    right_gene: str,
    partner_switch: Optional[int] = None,
    **weights: float,
) -> dict:
    """Build and score a reconstructed full-length fusion transcript."""
    fusion_graph = build_fusion_transcript_graph(
        exon_intervals,
        left_gene,
        right_gene,
        partner_switch=partner_switch,
    )
    counts, score = calculate_perturbation(
        reference_graph,
        fusion_graph,
        left_gene,
        right_gene,
        **weights,
    )
    result = counts.as_dict()
    result["score"] = score
    result["left_gene"] = left_gene
    result["right_gene"] = right_gene
    return result
