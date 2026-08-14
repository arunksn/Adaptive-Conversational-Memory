from src.embeddings.embedding_model import EmbeddingModel


def test_embedding_dimension():

    model = EmbeddingModel()

    embedding = model.encode(
        "I prefer Python for machine learning."
    )

    assert embedding.shape == (384,)


def test_embedding_type():

    model = EmbeddingModel()

    embedding = model.encode(
        "I prefer Python for machine learning."
    )

    assert embedding.dtype.name == "float32"