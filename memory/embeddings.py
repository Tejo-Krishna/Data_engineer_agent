"""
Embedding utility.

Single async embed() function using OpenAI text-embedding-3-small.
Every tool and script that needs embeddings imports from here.
Never instantiate AsyncOpenAI directly elsewhere.
"""

from openai import AsyncOpenAI

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI()  # reads OPENAI_API_KEY from environment at first call
    return _client


async def embed(text: str) -> list[float]:
    """
    Return a 1536-dimensional embedding vector for the given text.

    Uses text-embedding-3-small. Output dimension matches the vector(1536)
    columns in transform_library and catalogue Postgres tables.
    """
    response = await _get_client().embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding
