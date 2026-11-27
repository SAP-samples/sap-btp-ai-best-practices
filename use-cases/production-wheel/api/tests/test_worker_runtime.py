"""Verify local API startup owns a worker and shutdown signals it cleanly."""
import threading
from types import SimpleNamespace
from app.workspace.worker_runtime import embedded_worker_enabled, workspace_lifespan
import asyncio


def test_local_default_and_production_opt_out(monkeypatch):
    """Local uvicorn starts execution; production uses the independent worker by default."""
    monkeypatch.delenv("WORKSPACE_EMBEDDED_WORKER", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    assert embedded_worker_enabled()
    monkeypatch.setenv("APP_ENV", "production")
    assert not embedded_worker_enabled()
    monkeypatch.setenv("WORKSPACE_EMBEDDED_WORKER", "true")
    assert embedded_worker_enabled()


def test_lifespan_starts_and_stops_worker(monkeypatch):
    """The app lifespan starts one worker thread and waits for graceful stop."""
    from app.workspace import worker_runtime
    monkeypatch.setenv("WORKSPACE_EMBEDDED_WORKER", "true")
    started, stopped = threading.Event(), threading.Event()

    def worker():
        """Provide a cooperative fake worker without a database or child process."""
        return object()

    def loop(service, stop):
        """Record ownership until lifespan asks this worker to stop."""
        started.set()
        stop.wait(2)
        stopped.set()

    monkeypatch.setattr(worker_runtime, "get_service", worker)
    monkeypatch.setattr(worker_runtime, "worker_loop", loop)

    async def exercise():
        """Run startup and shutdown with no external server."""
        async with workspace_lifespan(SimpleNamespace()):
            assert started.wait(1)
        assert stopped.is_set()

    asyncio.run(exercise())
