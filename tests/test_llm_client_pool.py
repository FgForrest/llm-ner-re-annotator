"""HTTP pool nesmí být menší než max_concurrency (jinak fronta v httpx)."""
import asyncio

import pytest

from labeler.llm_client import LLMClient


def _limits(max_concurrency: int):
    async def run():
        async with LLMClient(
            base_url="http://localhost:9/v1",
            model="test",
            max_concurrency=max_concurrency,
        ) as c:
            return c._client._transport._pool._max_connections, \
                   c._client._transport._pool._max_keepalive_connections
    return asyncio.run(run())


@pytest.mark.parametrize("conc", [8, 32, 64, 128, 256])
def test_pool_scales_with_concurrency(conc):
    max_conn, max_keepalive = _limits(conc)
    assert max_conn > conc, f"pool {max_conn} <= concurrency {conc}"
    assert max_keepalive >= conc


def test_small_concurrency_has_floor():
    max_conn, max_keepalive = _limits(1)
    assert max_conn >= 16
    assert max_keepalive >= 8
