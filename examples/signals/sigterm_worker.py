"""Worker that shuts down cleanly on SIGTERM.

Shipped into the job by ``sigterm.py`` with ``inject_files``. It runs
locally too: start it, then ``kill -TERM <pid>``.

It logs a progress line every 10 steps. On SIGTERM it logs the signal,
runs cleanup, and exits 0. If no SIGTERM arrives within
``GIVE_UP_AFTER`` seconds, it exits 1 so a broken setup ends ``FAILED``
instead of running until the job timeout.
"""

import logging
import os
import signal
import sys
import time

GIVE_UP_AFTER = 300

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("worker")

received: str | None = None


def on_sigterm(signum: int, frame: object) -> None:
    """Record the signal. Logging and cleanup run in the main loop."""
    global received
    received = signal.Signals(signum).name


def cleanup(step: int) -> None:
    """Stand in for writing a checkpoint."""
    log.info("cleanup: saving progress at step %d", step)
    time.sleep(2)
    log.info("cleanup: done")


def main() -> None:
    """Work in one-second steps until SIGTERM, then clean up and exit."""
    signal.signal(signal.SIGTERM, on_sigterm)
    started = time.monotonic()
    log.info("worker started, pid %d", os.getpid())
    step = 0
    while received is None:
        elapsed = time.monotonic() - started
        if elapsed > GIVE_UP_AFTER:
            log.error("no SIGTERM after %.0fs, giving up", elapsed)
            sys.exit(1)
        step += 1
        if step % 10 == 0:
            log.info("working: step %d, %.0fs elapsed", step, elapsed)
        time.sleep(1)
    elapsed = time.monotonic() - started
    log.warning("received %s after %.1fs at step %d", received, elapsed, step)
    cleanup(step)
    log.info("graceful shutdown complete")
    sys.exit(0)


if __name__ == "__main__":
    main()
