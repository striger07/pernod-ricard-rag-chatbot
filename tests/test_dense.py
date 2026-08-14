from types import SimpleNamespace
from unittest.mock import MagicMock

from config.settings import Settings
from retrieval.dense import DenseRetriever, point_to_chunk


def test_point_to_chunk_maps_payload_and_score():
    point = SimpleNamespace(
        id="pt-1",
        score=0.81,
        vector=[0.1, 0.2],
        payload={
            "content": "Jameson is a triple-distilled Irish whiskey.",
            "title": "Jameson",
            "url": "https://www.jamesonwhiskey.com/",
            "source": "Jameson",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "chunk_id": "c1",
            "document_id": "d1",
        },
    )
    chunk = point_to_chunk(point)
    assert chunk.dense_score == 0.81
    assert chunk.title == "Jameson"
    assert chunk.url == "https://www.jamesonwhiskey.com/"
    assert chunk.source == "Jameson"
    assert chunk.timestamp.startswith("2026")
    assert chunk.content.startswith("Jameson")
    assert chunk.embedding == [0.1, 0.2]


def test_dense_retriever_accepts_raw_vector():
    settings = Settings(dense_top_k=3, _env_file=None)
    store = MagicMock()
    store.search.return_value = [
        SimpleNamespace(
            id="pt-1",
            score=0.9,
            vector=[1.0, 0.0],
            payload={
                "content": "The Glenlivet is a Speyside single malt Scotch whisky.",
                "title": "The Glenlivet",
                "url": "https://www.theglenlivet.com/",
                "source": "The Glenlivet",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "chunk_id": "g1",
            },
        )
    ]
    retriever = DenseRetriever(settings=settings, store=store, embedder=MagicMock())
    hits = retriever.search(vector=[1.0, 0.0], top_k=1)
    store.search.assert_called_once()
    kwargs = store.search.call_args
    assert kwargs.kwargs["with_vectors"] is True
    assert hits[0].title == "The Glenlivet"
    assert hits[0].dense_score == 0.9


def test_dense_retriever_embeds_query_string():
    settings = Settings(dense_top_k=2, _env_file=None)
    store = MagicMock()
    store.search.return_value = []
    embedder = MagicMock()
    embedder.encode.return_value = [[0.2, 0.8]]
    retriever = DenseRetriever(settings=settings, store=store, embedder=embedder)
    retriever.search(query="What is Absolut vodka?")
    embedder.encode.assert_called_once()
    store.search.assert_called_once()
    assert list(store.search.call_args.args[0]) == [0.2, 0.8]
