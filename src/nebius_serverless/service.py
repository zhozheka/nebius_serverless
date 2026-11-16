"""One cloud endpoint, and the collection addressed by id."""

import sys
import threading
import time

from .errors import Error, ValidationError, WaitTimeout, recovery_for
from .loop import reject_running_loop
from .output import (
    Follow,
    Progress,
    call_progress,
    call_progress_async,
    elapsed_ms,
    log_transport,
    report,
    show_record,
    timeout_text,
)
from .run import record_name, state_of
from .status import Status
from .transport import EndpointTransport

_LOG = "nebius_serverless.service"
_NOUN = "endpoint"

READY_STATE = "RUNNING"
STOPPED_STATES = frozenset({"ERROR", "STOPPED"})
POLL_SECONDS = 5.0


def monotonic() -> float:
    """Seconds used to decide when a local wait is over."""
    return time.monotonic()


def sleep(seconds: float, stop: threading.Event | None = None) -> None:
    """Pause between polls. Return early when ``stop`` is set.

    Tests replace this with a fake clock.
    """
    if stop is None:
        time.sleep(seconds)
        return
    stop.wait(seconds)


def url_of(record: dict[str, object] | None) -> str | None:
    """The first ``https://`` entry in ``status.public_endpoints``."""
    if record is None:
        return None
    status = record.get("status")
    if not isinstance(status, dict):
        return None
    items = status.get("public_endpoints")
    if not isinstance(items, list):
        return None
    for item in items:
        if isinstance(item, str) and item.startswith("https://"):
            return item
    return None


class Service:
    """One Nebius endpoint id.

    You get a ``Service`` from ``Endpoint.create()`` or
    ``endpoints.get()``. Property reads do not perform I/O. ``wait``
    polls until the endpoint is ready or will not become ready. Nothing
    here deletes the endpoint except ``delete``.
    """

    def __init__(
        self,
        endpoint_id: str,
        *,
        name: str,
        transport: EndpointTransport,
        create_state: str | None = None,
        record: dict[str, object] | None = None,
        token: str | None = None,
    ) -> None:
        self._id = endpoint_id
        self._name = name
        self._transport = transport
        self._create_state = create_state
        self._record = record
        self._token = token
        self._url = url_of(record)

    def __repr__(self) -> str:
        return (
            f"Service(id={self._id!r}, name={self._name!r}, "
            f"status={self.status!r}, url={self._url!r})"
        )

    @property
    def id(self) -> str:
        """Nebius endpoint id. Reading it does not perform I/O."""
        return self._id

    @property
    def name(self) -> str:
        """Display name. Lookup does not use it."""
        return self._name

    @property
    def record(self) -> dict[str, object] | None:
        """Last fetched JSON, or None before the first successful fetch."""
        return self._record

    @property
    def status(self) -> str | None:
        """Raw state from the last fetch, else the create response."""
        if self._record is not None:
            return state_of(self._record)
        return self._create_state

    @property
    def url(self) -> str | None:
        """Managed ``https://`` URL, or None until a fetch has seen one."""
        return self._url

    @property
    def token(self) -> str | None:
        """Bearer token from this create call.

        ``None`` when ``auth='none'`` and on a service loaded by
        ``endpoints.get``.
        """
        return self._token

    def get(self, *, quiet: bool = False) -> dict[str, object]:
        """Fetch the endpoint once and return its JSON.

        ``ERROR`` and ``STOPPED`` are data. ``quiet=True`` suppresses
        the snapshot line for this call only.
        """
        reject_running_loop("get_async")

        def work(progress: Progress) -> dict[str, object]:
            return self._present_get(progress)

        return call_progress(
            quiet=quiet,
            logger_name=_LOG,
            work=work,
            noun="endpoint",
        )

    async def get_async(self, *, quiet: bool = False) -> dict[str, object]:
        """Await the same fetch as ``get``."""

        def work(progress: Progress) -> dict[str, object]:
            return self._present_get(progress)

        return await call_progress_async(
            quiet=quiet,
            logger_name=_LOG,
            work=work,
            noun="endpoint",
        )

    def wait(
        self,
        timeout: float | None = None,
        *,
        quiet: bool = False,
    ) -> dict[str, object]:
        """Poll ``get`` until the endpoint is ready or has stopped.

        Ready means ``RUNNING`` with a managed URL. ``ERROR`` and
        ``STOPPED`` end the wait as data. ``timeout=0`` is one fetch. A
        local timeout raises ``WaitTimeout`` and leaves the endpoint
        billing. This does not probe the application's own route.
        """
        reject_running_loop("wait_async")

        def work(progress: Progress) -> dict[str, object]:
            return self._wait(timeout, progress)

        return call_progress(
            quiet=quiet,
            logger_name=_LOG,
            work=work,
            noun="endpoint",
        )

    async def wait_async(
        self,
        timeout: float | None = None,
        *,
        quiet: bool = False,
    ) -> dict[str, object]:
        """Await the same poll as ``wait``.

        Cancelling this await does not delete the endpoint. The worker
        stops polling once a ``get`` already in flight returns.
        """

        def work(progress: Progress) -> dict[str, object]:
            return self._wait(timeout, progress)

        return await call_progress_async(
            quiet=quiet,
            logger_name=_LOG,
            work=work,
            noun="endpoint",
        )

    def delete(self, *, quiet: bool = False) -> dict[str, object]:
        """Request deletion and return when the request is accepted.

        The payload does not claim the machine has stopped or that
        billing has ended.
        """
        reject_running_loop("delete_async")

        def work(progress: Progress) -> dict[str, object]:
            return self._present_delete(progress)

        return call_progress(
            quiet=quiet,
            logger_name=_LOG,
            work=work,
            noun="endpoint",
        )

    async def delete_async(
        self,
        *,
        quiet: bool = False,
    ) -> dict[str, object]:
        """Await the same deletion request as ``delete``."""

        def work(progress: Progress) -> dict[str, object]:
            return self._present_delete(progress)

        return await call_progress_async(
            quiet=quiet,
            logger_name=_LOG,
            work=work,
            noun="endpoint",
        )

    def display(self) -> None:
        """Print the last fetched record, including the URL.

        Does not fetch. Before the first successful ``get`` or ``wait``,
        prints the id and name.
        """
        show_record(
            self._record,
            job_id=self._id,
            name=self._name,
            stream=sys.stdout,
        )

    def _get(self) -> dict[str, object]:
        record = self._transport.get(self._id)
        self._record = record
        found = url_of(record)
        if found is not None:
            self._url = found
        return record

    def _present_get(self, progress: Progress) -> dict[str, object]:
        started = time.perf_counter()
        try:
            record = self._get()
        except Error as exc:
            log_transport(
                progress,
                job_id=self._id,
                attempt=1,
                duration_ms=elapsed_ms(started),
                request_id=exc.request_id,
            )
            raise
        report(
            Status(
                kind="snapshot",
                job_id=self._id,
                state=state_of(record),
                attempt=1,
                duration_ms=elapsed_ms(started),
                noun=_NOUN,
            ),
            progress,
        )
        return record

    def _present_delete(self, progress: Progress) -> dict[str, object]:
        started = time.perf_counter()
        try:
            payload = self._transport.delete(self._id)
        except Error as exc:
            log_transport(
                progress,
                job_id=self._id,
                attempt=1,
                duration_ms=elapsed_ms(started),
                request_id=exc.request_id,
            )
            raise
        duration_ms = elapsed_ms(started)
        log_transport(
            progress,
            job_id=self._id,
            attempt=1,
            duration_ms=duration_ms,
        )
        report(
            Status(
                kind="delete_accepted",
                job_id=self._id,
                duration_ms=duration_ms,
                noun=_NOUN,
            ),
            progress,
        )
        return payload

    def _settled(self, record: dict[str, object]) -> bool:
        state = state_of(record)
        if state in STOPPED_STATES:
            return True
        return state == READY_STATE and self._url is not None

    def _wait(
        self,
        timeout: float | None,
        progress: Progress,
    ) -> dict[str, object]:
        deadline = _deadline(timeout, self._id)
        one_fetch = timeout is not None and timeout <= 0
        progress.begin_wait()
        started = monotonic()
        follow = Follow(progress, self._id, noun=_NOUN)
        attempt = 0
        while True:
            attempt += 1
            began = time.perf_counter()
            try:
                record = self._get()
            except Error as exc:
                log_transport(
                    progress,
                    job_id=self._id,
                    attempt=attempt,
                    duration_ms=elapsed_ms(began),
                    request_id=exc.request_id,
                )
                raise
            settled = self._settled(record)
            follow.poll(
                state=state_of(record),
                elapsed=monotonic() - started,
                attempt=attempt,
                duration_ms=elapsed_ms(began),
                terminal=settled,
                detail=self._url if settled else None,
            )
            if settled:
                return record
            if one_fetch or _expired(deadline):
                _give_up(follow, timeout, self._id)
            pause = POLL_SECONDS
            if deadline is not None:
                pause = max(0.0, min(pause, deadline - monotonic()))
            sleep(pause, progress.stopped)
            # A cancelled await closes the handle: stop polling here.
            progress.raise_if_abandoned()
            if _expired(deadline):
                _give_up(follow, timeout, self._id)


class Endpoints:
    """Cloud endpoints addressed by a saved id string."""

    def get(
        self,
        endpoint_id: str,
        *,
        profile: str | None = None,
        project: str | None = None,
        transport: EndpointTransport | None = None,
        quiet: bool = False,
    ) -> Service:
        """Fetch ``endpoint_id`` once and return a ``Service``.

        Does not create. ``token`` on the result is ``None``.
        """
        reject_running_loop("endpoints.get_async")

        def work(progress: Progress) -> Service:
            return self._get(
                endpoint_id,
                profile=profile,
                project=project,
                transport=transport,
                progress=progress,
            )

        return call_progress(
            quiet=quiet,
            logger_name=_LOG,
            work=work,
            noun="endpoint",
        )

    async def get_async(
        self,
        endpoint_id: str,
        *,
        profile: str | None = None,
        project: str | None = None,
        transport: EndpointTransport | None = None,
        quiet: bool = False,
    ) -> Service:
        """Await the same fetch as ``get``."""

        def work(progress: Progress) -> Service:
            return self._get(
                endpoint_id,
                profile=profile,
                project=project,
                transport=transport,
                progress=progress,
            )

        return await call_progress_async(
            quiet=quiet,
            logger_name=_LOG,
            work=work,
            noun="endpoint",
        )

    def _get(
        self,
        endpoint_id: str,
        *,
        profile: str | None,
        project: str | None,
        transport: EndpointTransport | None,
        progress: Progress,
    ) -> Service:
        chosen = transport or _sdk_transport(profile, project)
        started = time.perf_counter()
        try:
            record = chosen.get(endpoint_id)
        except Error as exc:
            log_transport(
                progress,
                job_id=endpoint_id,
                attempt=1,
                duration_ms=elapsed_ms(started),
                request_id=exc.request_id,
            )
            raise
        report(
            Status(
                kind="snapshot",
                job_id=endpoint_id,
                state=state_of(record),
                attempt=1,
                duration_ms=elapsed_ms(started),
                noun=_NOUN,
            ),
            progress,
        )
        return Service(
            endpoint_id,
            name=record_name(record),
            transport=chosen,
            record=record,
        )


endpoints = Endpoints()


def _sdk_transport(
    profile: str | None,
    project: str | None,
) -> EndpointTransport:
    from .sdk_transport import SdkEndpointTransport

    return SdkEndpointTransport(profile=profile, project=project)


def _deadline(timeout: float | None, endpoint_id: str) -> float | None:
    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValidationError(
            "wait timeout must be a number of seconds or None.",
            operation="wait",
            run_id=endpoint_id,
        )
    if timeout < 0:
        raise ValidationError(
            "wait timeout must be zero or greater.",
            operation="wait",
            run_id=endpoint_id,
        )
    return monotonic() + float(timeout)


def _expired(deadline: float | None) -> bool:
    return deadline is not None and monotonic() >= deadline


def _give_up(
    follow: Follow,
    timeout: float | None,
    endpoint_id: str,
) -> None:
    seconds = 0.0 if timeout is None else float(timeout)
    follow.timed_out(seconds)
    raise WaitTimeout(
        timeout_text(endpoint_id, seconds, _NOUN),
        operation="wait",
        run_id=endpoint_id,
        recovery=recovery_for(WaitTimeout.code, _NOUN),
    )
