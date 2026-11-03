"""The web view: a light monitoring dashboard served by the process that runs the model.

``watch.view()`` serves it from your script (in a background thread, so a notebook keeps working);
``innards view [folder]`` serves a folder of runs, or one run folder copied from another machine.
Needs the ``view`` extra: ``pip install "innards[view]"``.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
import webbrowser
from typing import Any

from innards.view.runner import Runner, RunnerError, TurnJob

__all__ = ["Runner", "RunnerError", "TurnJob", "ViewServer", "serve"]

log = logging.getLogger("innards.view")


def _free_port(host: str, port: int, tries: int = 20) -> int:
    for candidate in range(port, port + tries):
        with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, candidate))
            except OSError:
                continue
            return candidate
    raise OSError(f"No free port between {port} and {port + tries - 1} on {host}")


class ViewServer:
    """A running view. ``url`` is where to open it; ``stop()`` shuts it down."""

    def __init__(self, server: Any, thread: threading.Thread | None, url: str) -> None:
        self._server = server
        self._thread = thread
        self.url = url

    def stop(self, timeout: float = 5.0) -> None:
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout)

    def wait(self) -> None:
        """Block until the server stops (Ctrl-C stops it)."""
        try:
            while self._thread is not None and self._thread.is_alive():
                self._thread.join(0.5)
        except KeyboardInterrupt:
            self.stop()

    def __repr__(self) -> str:
        return f"<innards view at {self.url}>"

    def _repr_html_(self) -> str:  # a clickable link in Jupyter
        return f'<a href="{self.url}" target="_blank">innards view: {self.url}</a>'


def serve(
    root: str | None = None,
    *,
    watch: Any = None,
    runner: Runner | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    token: str | None = None,
    open_browser: bool = False,
    block: bool = False,
    log_level: str = "warning",
) -> ViewServer:
    """Serve the view over the runs in ``root`` (default ``$INNARDS_RUNS`` or ``./innards_runs``).

    ``block=False`` runs the server in a background thread and returns at once; ``block=True`` (the CLI)
    serves until interrupted.
    """
    try:
        import uvicorn

        from innards.view.app import create_app
    except ImportError as exc:
        raise ImportError('The web view needs FastAPI and uvicorn: pip install "innards[view]"') from exc
    from innards.store import RunStore

    store = RunStore(root)
    app = create_app(store, watch=watch, runner=runner, token=token)
    chosen = _free_port(host, port) if not block else port
    shown = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{shown}:{chosen}/"
    config = uvicorn.Config(app, host=host, port=chosen, log_level=log_level, proxy_headers=False)
    server = uvicorn.Server(config)
    if block:
        print(f"innards view · {url} · runs in {store.url}", flush=True)
        if open_browser:
            threading.Timer(1.0, webbrowser.open, args=(url,)).start()
        server.run()
        return ViewServer(server, None, url)
    thread = threading.Thread(target=server.run, name="innards-view", daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        raise RuntimeError(f"The view did not start on {url}")
    log.info("innards view at %s", url)
    if open_browser:
        webbrowser.open(url)
    return ViewServer(server, thread, url)
