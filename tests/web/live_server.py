"""The local files server on a free port, for the browser tests -- started, then waited on."""

from __future__ import annotations

import socket
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

import uvicorn

#: How long a server may take to start before the test fails rather than waits on.
STARTUP_TIMEOUT_S = 30.0


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def serving(app: Any) -> Iterator[str]:
    """Yield the base URL once uvicorn reports it is listening, and stop it afterwards.

    The tests used to start the thread and sleep a second, which is a guess: too short on a
    loaded machine (the first request is refused) and wasted everywhere else. `Server.started`
    is set when startup has finished and the socket is bound.
    """
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while not server.started:
        assert thread.is_alive(), "the server thread exited during startup"
        assert time.monotonic() < deadline, f"the server did not start in {STARTUP_TIMEOUT_S:.0f} s"
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=STARTUP_TIMEOUT_S)
