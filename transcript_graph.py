import pickle
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import networkx as nx


class TranscriptGraph(nx.DiGraph):
    """Reference transcript graph persisted per GTF/DB input.

    The graph encodes gene -> transcript -> exon relationships and exon-to-exon
    splice connections. This lets a fusion transcript graph be compared against
    the reference graph to quantify perturbations.
    """

    def __init__(self, cache_path: Optional[str] = None):
        super().__init__()
        self.cache_path = Path(cache_path) if cache_path else None

    @staticmethod
    def _gene_name_from_feature(feature) -> Optional[str]:
        attrs = getattr(feature, "attributes", {}) or {}
        for key in ("gene_name", "Name", "gene_symbol"):
            value = attrs.get(key, [None])[0]
            if value:
                return str(value)
        return None

    @staticmethod
    def _transcript_id_from_feature(feature, default: str) -> str:
        attrs = getattr(feature, "attributes", {}) or {}
        for key in ("transcript_id", "ID", "transcript_name"):
            values = attrs.get(key, [])
            if values:
                return str(values[0])
        return default

    @staticmethod
    def _exon_id_from_feature(feature, default: str) -> str:
        attrs = getattr(feature, "attributes", {}) or {}
        for key in ("exon_id", "ID", "Name"):
            values = attrs.get(key, [])
            if values:
                return str(values[0])
        return default

    @classmethod
    def from_reference_db(cls, db, cache_path: Optional[str] = None, persist: bool = True):
        """Load a persisted graph if it exists, otherwise build it once and save it."""
        db_path = getattr(db, "dbfn", None)
        if cache_path is None and db_path:
            cache_path = f"{db_path}.transcript_graph.pkl"

        graph = cls(cache_path=cache_path)
        if cache_path and Path(cache_path).exists():
            try:
                with open(cache_path, "rb") as handle:
                    loaded = pickle.load(handle)
                if isinstance(loaded, cls):
                    loaded.cache_path = Path(cache_path)
                    return loaded
            except Exception:
                pass

        graph._build_reference_graph(db)
        if persist and cache_path:
            graph.save()
        return graph

    def _build_reference_graph(self, db) -> "TranscriptGraph":
        self.clear()
        for gene in db.features_of_type("gene"):
            gene_key = self._gene_name_from_feature(gene) or getattr(gene, "id", None) or f"gene:{gene.chrom}:{gene.start}-{gene.end}"
            self.add_node(
                gene_key,
                kind="gene",
                chrom=str(gene.chrom),
                start=int(gene.start),
                end=int(gene.end),
                gene_id=getattr(gene, "id", gene_key),
            )

            for transcript in db.children(gene, featuretype="transcript", order_by="start"):
                tx_id = self._transcript_id_from_feature(transcript, f"{gene_key}:{transcript.start}-{transcript.end}")
                self.add_node(
                    tx_id,
                    kind="transcript",
                    gene=gene_key,
                    chrom=str(transcript.chrom),
                    start=int(transcript.start),
                    end=int(transcript.end),
                    tx_id=tx_id,
                )
                self.add_edge(gene_key, tx_id, kind="contains")

                exons = list(db.children(transcript, featuretype="exon", order_by="start"))
                prev_exon = None
                for exon in exons:
                    exon_key = self._exon_id_from_feature(exon, f"{tx_id}:{exon.start}-{exon.end}")
                    self.add_node(
                        exon_key,
                        kind="exon",
                        chrom=str(exon.chrom),
                        start=int(exon.start),
                        end=int(exon.end),
                        transcript=tx_id,
                        gene=gene_key,
                    )
                    self.add_edge(tx_id, exon_key, kind="contains")
                    if prev_exon is not None:
                        self.add_edge(prev_exon, exon_key, kind="splice_junction")
                    prev_exon = exon_key

        if self.cache_path:
            self.save()
        return self

    def save(self, path: Optional[str] = None) -> str:
        target = Path(path) if path else self.cache_path
        if target is None:
            raise ValueError("No cache path configured for transcript graph persistence")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.cache_path = target
        return str(target)

    def load(self, path: Optional[str] = None) -> "TranscriptGraph":
        target = Path(path) if path else self.cache_path
        if target is None or not target.exists():
            raise FileNotFoundError(f"Transcript graph cache not found: {target}")
        with open(target, "rb") as handle:
            loaded = pickle.load(handle)
        self.__dict__.update(loaded.__dict__)
        return self

    def exon_nodes(self) -> Iterable[str]:
        return [node for node, attrs in self.nodes(data=True) if attrs.get("kind") == "exon"]

    def junction_edges(self) -> set[Tuple[str, str]]:
        return {
            (src, dst)
            for src, dst, attrs in self.edges(data=True)
            if attrs.get("kind") == "splice_junction"
        }

    def fusion_edges(self) -> set[Tuple[str, str]]:
        return {
            (src, dst)
            for src, dst, attrs in self.edges(data=True)
            if attrs.get("kind") in {"splice_junction", "fusion_edge"}
        }

    def perturbation_score(
        self,
        fusion_graph: "TranscriptGraph",
        *,
        a: float = 1.0,
        b: float = 1.0,
        c: float = 1.0,
        d: float = 1.0,
    ) -> float:
        """Simple perturbation score from the MVP model: a*novel_junctions + ..."""
        ref_junctions = self.junction_edges()
        fusion_junctions = fusion_graph.junction_edges()
        novel_junctions = len(fusion_junctions - ref_junctions)

        ref_exons = set(self.exon_nodes())
        fusion_exons = set(fusion_graph.exon_nodes())
        skipped_exons = len(ref_exons - fusion_exons)
        inserted_exons = len(fusion_exons - ref_exons)

        fusion_edges = len(fusion_graph.fusion_edges())
        return a * novel_junctions + b * skipped_exons + c * inserted_exons + d * fusion_edges

    def from_fusion_exons(cls, exon_intervals: Sequence[Tuple[str, int, int]], *, gene_name: Optional[str] = None):
        """Create a minimal fusion transcript graph from exon intervals."""
        graph = cls()
        if gene_name:
            graph.add_node(gene_name, kind="gene", gene_name=gene_name)
        prev_exon = None
        for idx, (chrom, start, end) in enumerate(exon_intervals):
            exon_name = f"fusion_exon_{idx}:{chrom}:{start}-{end}"
            graph.add_node(exon_name, kind="exon", chrom=chrom, start=start, end=end)
            if prev_exon is not None:
                graph.add_edge(prev_exon, exon_name, kind="splice_junction")
            prev_exon = exon_name
        return graph
