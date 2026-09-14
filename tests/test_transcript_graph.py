import gffutils
from pathlib import Path

from genomic_interval_index import GenomicIntervalIndex


def _write_test_gtf(path: Path):
    path.write_text(
        """##gff-version 3
chr1\tmakers\tgene\t100\t500\t.\t+\t.\tID=gene1;gene_name=GENE1
chr1\tmakers\ttranscript\t100\t500\t.\t+\t.\tID=tx1;Parent=gene1;transcript_id=tx1
chr1\tmakers\texon\t100\t220\t.\t+\t.\tID=exon1;Parent=tx1
chr1\tmakers\texon\t221\t500\t.\t+\t.\tID=exon2;Parent=tx1
chr2\tmakers\tgene\t200\t800\t.\t+\t.\tID=gene2;gene_name=GENE2
chr2\tmakers\ttranscript\t200\t800\t.\t+\t.\tID=tx2;Parent=gene2;transcript_id=tx2
chr2\tmakers\texon\t200\t350\t.\t+\t.\tID=exon3;Parent=tx2
chr2\tmakers\texon\t351\t800\t.\t+\t.\tID=exon4;Parent=tx2
"""
    )


def test_reference_transcript_graph_is_built_and_persisted(tmp_path):
    gtf_path = tmp_path / "toy.gtf"
    db_path = tmp_path / "toy.db"
    _write_test_gtf(gtf_path)
    gffutils.create_db(str(gtf_path), str(db_path), force=True, disable_infer_genes=False, disable_infer_transcripts=False)

    db = gffutils.FeatureDB(str(db_path), keep_order=True)
    index = GenomicIntervalIndex(db)

    assert hasattr(index, "transcript_graph")
    assert index.transcript_graph is not None
    assert index.transcript_graph.number_of_nodes() > 0
    assert "GENE1" in index.transcript_graph.nodes

    cache_path = tmp_path / "toy.transcript_graph.pkl"
    assert cache_path.exists()
