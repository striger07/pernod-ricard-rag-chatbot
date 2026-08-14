from retrieval.mmr import MMRReranker, cosine_similarity
from retrieval.models import RetrievedChunk


class _StaticEmbedder:
    def encode(self, texts):
        mapping = {
            "query": [1.0, 0.0],
            "absolut vodka sweden": [1.0, 0.0],
            "absolut vodka sweden duplicate": [0.99, 0.01],
            "glenlivet speyside malt": [0.0, 1.0],
        }
        return [mapping[text] for text in texts]


def test_cosine_orthogonal_is_zero():
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-6


def test_mmr_promotes_diverse_chunk():
    query = "query"
    similar = RetrievedChunk(
        content="absolut vodka sweden",
        title="Absolut",
        url="https://www.absolut.com/a",
        source="Absolut",
        chunk_id="a",
        embedding=[1.0, 0.0],
        rrf_score=0.03,
    )
    duplicate = RetrievedChunk(
        content="absolut vodka sweden duplicate",
        title="Absolut copy",
        url="https://www.absolut.com/b",
        source="Absolut",
        chunk_id="b",
        embedding=[0.99, 0.01],
        rrf_score=0.02,
    )
    diverse = RetrievedChunk(
        content="glenlivet speyside malt",
        title="Glenlivet",
        url="https://www.theglenlivet.com/c",
        source="The Glenlivet",
        chunk_id="c",
        embedding=[0.0, 1.0],
        rrf_score=0.01,
    )
    reranker = MMRReranker(embedder=_StaticEmbedder(), lambda_mult=0.3)
    ranked = reranker.rerank(query, [similar, duplicate, diverse], query_embedding=[1.0, 0.0], top_k=2)
    ids = [chunk.chunk_id for chunk in ranked]
    assert ids[0] == "a"
    assert ids[1] == "c"
    assert all(chunk.mmr_score is not None for chunk in ranked)
