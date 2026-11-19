"""Awaited twins and the sync-in-loop guard."""

import asyncio
import threading

import pytest
from fakes import FakeTransport

import nebius_serverless as ns


def _job() -> ns.Job:
    return ns.Job(
        ["nvidia-smi"],
        image="python:3.12-slim",
        gpu="h100",
        timeout="1h",
    )


def test_sync_call_inside_a_loop_names_the_twin() -> None:
    job = _job()

    async def create() -> None:
        job.create(dry_run=True, region="eu-north1", project="p")

    with pytest.raises(ns.ValidationError, match="create_async"):
        asyncio.run(create())


def test_awaited_twins_match_the_sync_calls() -> None:
    job = _job()
    sync = job.create(dry_run=True, region="eu-north1", project="p")

    async def once() -> dict[str, object]:
        described = await job.create_async(
            dry_run=True,
            region="eu-north1",
            project="p",
        )
        assert isinstance(described, dict)
        return described

    assert asyncio.run(once()) == sync


def test_async_get_wait_and_cancel() -> None:
    fake = FakeTransport()
    fake.states = ["RUNNING", "COMPLETED"]
    run = _job().create(
        region="eu-north1",
        project="p",
        transport=fake,
    )
    assert isinstance(run, ns.Run)

    async def follow() -> None:
        first = await run.get_async()
        assert first["status"] == {"state": "RUNNING"}
        finished = await run.wait_async(timeout=0)
        assert finished["status"] == {"state": "COMPLETED"}
        payload = await run.cancel_async()
        assert payload["accepted"] is True
        loaded = await ns.jobs.get_async(run.id, transport=fake)
        assert loaded.id == run.id

    asyncio.run(follow())
    assert fake.cancels == [run.id]


def test_cancelling_wait_does_not_cancel_the_job() -> None:
    fake = FakeTransport()
    started = threading.Event()
    release = threading.Event()

    def blocking_get(job_id: str) -> dict[str, object]:
        started.set()
        release.wait(5)
        return {
            "metadata": {"id": job_id},
            "status": {"state": "RUNNING"},
        }

    fake.get = blocking_get  # type: ignore[method-assign]
    run = _job().create(
        region="eu-north1",
        project="p",
        transport=fake,
    )
    assert isinstance(run, ns.Run)

    async def cancel_wait() -> None:
        task = asyncio.create_task(run.wait_async(timeout=0))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_wait())
    release.set()
    assert fake.cancels == []


_CANCEL_SCRIPT = r"""
import asyncio, sys, time
sys.path.insert(0, "tests")
from fakes import FakeEndpointTransport, FakeTransport
import nebius_serverless as ns

kind = sys.argv[1]
if kind == "job":
    fake = FakeTransport()
    fake.states = ["RUNNING"]
    handle = ns.Job(["x"], image="a:b", timeout="1h").create(
        region="eu-north1", project="p", transport=fake, quiet=True)
else:
    fake = FakeEndpointTransport()
    fake.states = ["PROVISIONING"]
    handle = ns.Endpoint(image="a:b", port=80).create(
        region="eu-north1", project="p", transport=fake, quiet=True)

async def main():
    task = asyncio.create_task(handle.wait_async(quiet=True))
    while not fake.gets:
        await asyncio.sleep(0.01)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

started = time.monotonic()
asyncio.run(main())
print(len(fake.gets), round(time.monotonic() - started, 1))
"""


@pytest.mark.parametrize("kind", ["job", "endpoint"])
def test_cancelled_wait_async_stops_polling_and_lets_the_process_exit(
    kind: str,
) -> None:
    import subprocess
    import sys

    # The real poll interval (10s / 5s) applies: only the stop signal can
    # end this quickly. Before the fix the child polled until killed.
    out = subprocess.run(
        [sys.executable, "-c", _CANCEL_SCRIPT, kind],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert out.returncode == 0, out.stderr
    gets, seconds = out.stdout.split()
    assert int(gets) == 1
    assert float(seconds) < 2.0
