import asyncio
import os
import threading

import pytest

from pirl.harbor_client import harbor_client


@pytest.fixture(autouse=True)
def bounded_selector_wait(monkeypatch):
    # The execution sandbox blocks socket.send(), including asyncio's self-pipe
    # wakeup. A timer lets the actual wrapper process cross-thread callbacks
    # without changing production cleanup or making any network connection.
    original = asyncio.new_event_loop

    def create_loop():
        loop = original()

        def tick():
            loop.call_later(0.01, tick)

        loop.call_later(0.01, tick)
        return loop

    monkeypatch.setattr(asyncio, "new_event_loop", create_loop)


async def noop():
    return None


def test_sync_cleanup_releases_loops_threads_and_file_descriptors():
    baseline_threads = {t.ident for t in threading.enumerate()}
    baseline_fds = len(os.listdir("/proc/self/fd"))
    for _ in range(12):
        client = harbor_client("http://127.0.0.1:1")
        client._run_sync(noop)  # start the real OpenEnv wrapper without a network connection
        wrapper = client._sync_client
        assert wrapper._loop_thread.is_alive()
        client.close()
        assert wrapper._loop_thread is None and wrapper._loop is None
        assert client._http.is_closed
    assert {t.ident for t in threading.enumerate()} == baseline_threads
    assert len(os.listdir("/proc/self/fd")) <= baseline_fds + 1


def test_cleanup_exception_still_stops_sync_loop():
    client = harbor_client("http://127.0.0.1:1")
    client._run_sync(noop)
    wrapper = client._sync_client

    async def fail():
        raise RuntimeError("injected disconnect failure")

    client._disconnect_async = fail
    with pytest.raises(RuntimeError, match="injected"):
        client.close()
    assert wrapper._loop is None and wrapper._loop_thread is None
    assert client._http.is_closed


def test_close_on_owning_loop_returns_an_awaitable():
    client = harbor_client("http://127.0.0.1:1")

    async def close_on_loop():
        await client.close()

    client._run_sync(close_on_loop)
    assert client._http.is_closed
    client.close()
    assert client._sync_client._loop is None


def test_unused_client_close_does_not_create_a_loop():
    client = harbor_client("http://127.0.0.1:1")
    client.close()
    assert client._sync_client is None and client._http.is_closed


def test_async_cleanup_is_awaitable_and_closes_http_client():
    async def exercise():
        client = harbor_client("http://127.0.0.1:1")
        client._claim_execution_mode("async")
        await client.close()
        assert client._http.is_closed and client._sync_client is None

    asyncio.run(exercise())
