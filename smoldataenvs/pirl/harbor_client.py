"""Local Harbor cleanup fix: close the HTTP client and stop its synchronous loop."""

import asyncio
from functools import lru_cache


class CleanupMixin:
    async def _close_async(self):
        try:
            await super()._close_async()
        finally:
            self._http.close()

    def close(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        wrapper = getattr(self, "_sync_client", None)
        if wrapper is not None and loop is not wrapper._loop:
            # EnvClient.close() dispatches cleanup but does not stop the loop's thread.
            # SyncEnvClient.close() performs both, including on cleanup exceptions.
            return wrapper.close()
        if loop is None and getattr(self, "_execution_mode", None) is None:
            self._http.close()  # unused client: do not create a loop just to close it
            return None
        # Connection-failure cleanup runs on the owning loop and must be awaitable.
        # Bypass HarborEnv.close(), which currently drops that awaitable.
        return self._dispatch(self._close_async)


@lru_cache(maxsize=1)
def _client_type():
    from openenv.harbor.client import HarborEnv

    class ManagedHarborEnv(CleanupMixin, HarborEnv):
        pass

    return ManagedHarborEnv


def harbor_client(*args, **kwargs):
    return _client_type()(*args, **kwargs)
