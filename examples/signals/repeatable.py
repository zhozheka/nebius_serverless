"""Submit a job again when something outside the job stopped it.

``RepeatableJob.run_async()`` is plain asyncio, so the loop runs wherever
you await it: a notebook, a script, a Prefect flow, or the optional CPU
supervisor job from ``supervisor.py`` (``launch_async``).

Every step is logged: submission, each state change of each attempt, how
the attempt ended and why, and whether it is rescheduled. Logs go to the
``repeatable`` logger, or to the logger you pass, for example Prefect's
``get_run_logger()``. Every attempt is also kept in
``RepeatableJob.attempts``.
"""

import asyncio
import copy
import logging
import time
from dataclasses import dataclass

import nebius_serverless as ns

REASONS = {
    "COMPLETED": "completed",
    "CANCELLED": "cancelled",  # someone called cancel
    "FAILED": "job",  # non-zero exit, or past the job timeout
    "ERROR": "platform",  # VM stopped, no capacity, quota, ...
}
TERMINAL = frozenset(REASONS)
# state_details.code values that mean the platform ended the attempt,
# whatever the state. A stopped spot VM ends the job FAILED + Preempted.
PLATFORM_CODES = frozenset(
    {"Preempted", "VMStopped", "NotEnoughResources", "QuotaExceeded"}
)
# compute_instance_state values of a VM that may still run the container.
VM_ALIVE = frozenset({"CREATING", "STARTING", "RUNNING", "STOPPING"})

log = logging.getLogger("repeatable")


def end_reason(run: ns.Run) -> tuple[str, str]:
    """Return (reason, state_details.code) for a finished run."""
    details = run.record.get("status", {}).get("state_details") or {}
    code = details.get("code", "")
    if code in PLATFORM_CODES:
        return "platform", code
    return REASONS.get(run.status, "unknown"), code


def instance_id(run: ns.Run) -> str:
    """Return the id of the VM under a job, or "" before it has one."""
    instances = run.record.get("status", {}).get("instances") or []
    return instances[0].get("compute_instance_id", "") if instances else ""


def vm_state(run: ns.Run) -> str:
    """Return the VM's compute_instance_state, or "" when there is none."""
    instances = run.record.get("status", {}).get("instances") or []
    return instances[0].get("compute_instance_state", "") if instances else ""


def took(seconds: float) -> str:
    """Format a duration as ``1h02m03s``, ``2m03s`` or ``3s``."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


@dataclass
class Attempt:
    """One submission of the job and how it ended."""

    number: int
    job_id: str
    submitted: float  # time.time()
    status: str = "SUBMITTED"  # latest state seen
    instance_id: str = ""  # VM it ran on
    ended: float | None = None
    reason: str = ""  # completed, cancelled, job, platform
    code: str = ""  # state_details.code, e.g. VMStopped
    decision: str = ""  # reschedule or stop

    @property
    def duration(self) -> float:
        """Seconds from submission to the end, or until now."""
        return (self.ended or time.time()) - self.submitted


class RepeatableJob:
    """Submit a job again when something outside the job stopped it.

    Args:
        job: The job to run.
        name: Name of the whole run. Attempt ``n`` is created as job
            ``{name}-{n}`` with ``REPEATABLE_NAME`` and
            ``REPEATABLE_ATTEMPT`` in its environment. Defaults to the
            job's name.
        max_attempts: Give up after this many submissions.
        reschedule: End reasons that submit again (see ``REASONS``).
        poll: Seconds between status checks of the current attempt.
        drain_timeout: Before rescheduling, wait up to this many seconds
            for the old VM to stop. The job ends before its VM does, and
            the old container may still be saving its last checkpoint.
        logger: Where to log. Defaults to the ``repeatable`` logger.
    """

    def __init__(
        self,
        job: ns.Job,
        *,
        name: str | None = None,
        max_attempts: int = 3,
        reschedule: tuple[str, ...] = ("cancelled", "platform"),
        poll: float = 15.0,
        drain_timeout: float = 600.0,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self.job = job
        self.name = name or job.name
        self.max_attempts = max_attempts
        self.reschedule = reschedule
        self.poll = poll
        self.drain_timeout = drain_timeout
        self.log = logger or log
        self.runs: list[ns.Run] = []  # one Run per attempt
        self.attempts: list[Attempt] = []  # history, same order
        self._stopping = False

    async def run_async(self) -> ns.Run:
        """Run the loop in this process; return the last attempt's run."""
        self.log.info(
            "%s: up to %d attempts, reschedule on %s",
            self.name,
            self.max_attempts,
            ", ".join(self.reschedule),
        )
        for number in range(1, self.max_attempts + 1):
            tag = f"attempt {number}/{self.max_attempts}"
            self.log.info("%s: submitting", tag)
            try:
                run = await self.attempt_job(number).create_async(quiet=True)
            except Exception:
                self.log.exception("%s: create failed", tag)
                raise
            attempt = Attempt(number, run.id, time.time())
            self.runs.append(run)
            self.attempts.append(attempt)
            self.log.info("%s: created %s", tag, run.id)
            await self._watch(run, attempt, tag)
            attempt.reason, attempt.code = end_reason(run)
            again = attempt.reason in self.reschedule and not self._stopping
            attempt.decision = "reschedule" if again else "stop"
            self.log.info(
                "%s: %s ended %s (%s) after %s",
                tag,
                run.id,
                f"{run.status} {attempt.code}".rstrip(),
                attempt.reason,
                took(attempt.duration),
            )
            if again and number == self.max_attempts:
                attempt.decision = "stop"
                self.log.warning(
                    "%s: stopped from outside (%s), no attempts left",
                    tag,
                    attempt.reason,
                )
                break
            if again:
                self.log.warning(
                    "%s: stopped from outside (%s), rescheduling",
                    tag,
                    attempt.reason,
                )
                await self._drain(run, tag)
                continue
            if attempt.reason == "completed":
                self.log.info("%s: completed, done", tag)
            elif self._stopping:
                self.log.info("%s: stopped on request, done", tag)
            else:
                self.log.error(
                    "%s: %s is not rescheduled, giving up", tag, attempt.reason
                )
            return run
        self.log.error("no attempt completed in %d tries", self.max_attempts)
        raise RuntimeError(
            f"no attempt completed in {self.max_attempts} tries"
        )

    def attempt_job(self, number: int) -> ns.Job:
        """The job for attempt ``number``: named ``{name}-{number}``."""
        job = copy.copy(self.job)
        job.name = f"{self.name}-{number}"
        job.env = (
            *self.job.env,
            ("REPEATABLE_NAME", self.name),
            ("REPEATABLE_ATTEMPT", str(number)),
        )
        return job

    async def _watch(self, run: ns.Run, attempt: Attempt, tag: str) -> None:
        """Poll the attempt until it ends, logging every state change."""
        while True:
            try:
                await run.get_async(quiet=True)
            except Exception as exc:  # keep watching through API hiccups
                self.log.warning("%s: status check failed: %s", tag, exc)
                await asyncio.sleep(self.poll)
                continue
            if run.status != attempt.status:
                attempt.status = run.status
                where = ""
                if run.status == "RUNNING":
                    attempt.instance_id = instance_id(run)
                    where = f" on {attempt.instance_id}"
                self.log.info("%s: %s %s%s", tag, run.id, run.status, where)
            if run.status in TERMINAL:
                attempt.ended = time.time()
                return
            await asyncio.sleep(self.poll)

    async def _drain(self, run: ns.Run, tag: str) -> None:
        """Wait until the attempt's VM has stopped, or ``drain_timeout``."""
        vm, state = instance_id(run), vm_state(run)
        if state not in VM_ALIVE:
            return
        self.log.info("%s: waiting for %s to stop (%s)", tag, vm, state)
        start = time.monotonic()
        while state in VM_ALIVE:
            if time.monotonic() - start > self.drain_timeout:
                self.log.warning(
                    "%s: %s still %s after %s, rescheduling anyway",
                    tag,
                    vm,
                    state,
                    took(time.monotonic() - start),
                )
                return
            await asyncio.sleep(self.poll)
            try:
                await run.get_async(quiet=True)
            except Exception as exc:  # keep waiting through API hiccups
                self.log.warning("%s: status check failed: %s", tag, exc)
                continue
            state = vm_state(run)
        self.log.info(
            "%s: %s is %s after %s",
            tag,
            vm,
            state or "gone",
            took(time.monotonic() - start),
        )

    async def cancel_async(self) -> None:
        """Stop for good: cancel the current attempt, do not reschedule."""
        self._stopping = True
        if not self.runs:
            self.log.info("cancel requested before the first attempt")
            return
        self.log.info("cancel requested: cancelling %s", self.runs[-1].id)
        await self.runs[-1].cancel_async(quiet=True)

    def summary(self) -> str:
        """One line per attempt: id, end state, reason, decision, time."""
        lines = []
        for a in self.attempts:
            state = f"{a.status} {a.code}".rstrip()
            lines.append(
                f"{a.number}. {a.job_id}  {state} ({a.reason or '-'})"
                f"  -> {a.decision or 'running'}  {took(a.duration)}"
                f"  {a.instance_id}".rstrip()
            )
        return "\n".join(lines)

    async def launch_async(self, account, **kwargs) -> ns.Run:
        """Run this loop in a CPU job instead; see ``supervisor.launch``."""
        from supervisor import launch  # optional, only for this mode

        return await launch(self, account, **kwargs)
