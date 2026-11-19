"""A local job recipe. Constructing it does not contact Nebius."""

import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import overload

from .aliases import resolve_machine
from .errors import Error, UnsupportedFeature
from .loop import reject_running_loop
from .output import (
    Progress,
    call_progress,
    call_progress_async,
    elapsed_ms,
    log_transport,
    report,
)
from .profile import resolve_location
from .recipe import (
    check_identity,
    describe,
    load_inject_files,
    reject_spot_on_cpu,
    request_body,
    timeout_seconds,
    validate_command,
    validate_disk,
    validate_env,
    validate_image,
    validate_inject_files,
    validate_optional_text,
    validate_pricing,
    validate_registry_secret,
    validate_retries,
    validate_secrets,
    validate_spot_policy,
    validate_volumes,
    validate_working_dir,
)
from .run import Run
from .status import Status
from .transport import Transport, call_transport
from .yaml_job import load_yaml_fields

_LOG = "nebius_serverless.job"


class Job:
    """Recipe for one ``nebius ai job create``.

    Constructing a ``Job``, loading YAML, or importing the package does
    not authenticate, upload, or create. The Nebius SDK is imported on
    the call that needs a profile or a live create.

    Args:
        command: Nonempty list of strings, e.g. ``["python", "train.py",
            "--epochs", "3"]``. The first item is the container command;
            the rest are POSIX-quoted into one ``args`` string. A shell
            string is rejected; wrap shell syntax as
            ``["bash", "-lc", "a && b"]``.
        image: ``name:tag`` or ``registry/path@sha256:<digest>``, e.g.
            ``"pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime"``. Sent
            unchanged.
        gpu: GPU alias such as ``"h100"``, ``"h200"``, ``"b200"``,
            ``"b300"``, ``"l40s"`` or ``"rtx"``. Picks the smallest
            one-GPU preset in the project region. Omit it for CPU.
        platform: Explicit platform id, e.g. ``"gpu-h100-sxm"``.
            Overrides ``gpu``.
        preset: Explicit preset id, e.g. ``"1gpu-16vcpu-200gb"``. When
            omitted, the smallest preset for the platform is used.
        timeout: Required job deadline: ``"1h"``, ``"24h"``, seconds
            (``3600``), or a ``timedelta``. Must be 1 to 168 hours.
        pricing: ``"on-demand"`` (default) or ``"spot"``. Spot is
            preemptible and not available on CPU.
        subnet_id: Subnet id, e.g. ``"vpcsubnet-..."``. Only needed
            when the project has more than one subnet.
        volumes: Mount strings ``source:/container/path[:ro|rw]`` or
            ``s3://bucket:/container/path[:ro|rw]`` (no ``:profile``
            suffix), e.g.
            ``["s3://my-data:/workspace/data:ro"]``. Mode defaults to
            ``rw``.
        inject_files: ``local_path:/container/path`` strings, e.g.
            ``["./config.yaml:/app/config.yaml"]``. Each file is read on
            ``create`` and must be at most 64 KiB.
        env: Plain environment variables, e.g.
            ``{"WANDB_MODE": "offline"}``. Values are hidden from dry-run
            output and error text. Use ``secrets`` for tokens.
        name: Console label. Defaults to ``job-`` plus 12 hex chars.
            It is not a lookup key.
        disk: Boot disk size, e.g. ``"500Gi"``. Defaults to ``"250Gi"``.
        profile: Nebius CLI profile name. Defaults to the active profile.
        project: Project id, e.g. ``"project-..."``. Overrides the
            profile's project.
        retries: How many times Nebius restarts the job after it fails
            (``restart_attempts``). ``0`` (default) leaves it unset, so
            the service default applies.
        working_dir: Container working directory, e.g. ``"/workspace"``.
            At most 128 characters. Defaults to the image ``WORKDIR``.
        secrets: Environment variables read from MysteryBox, e.g.
            ``{"HF_TOKEN": "mbsec-..."}``. Each value is a secret id
            (``mbsec-``, latest version) or a version id
            (``mbsecver-``). The variable gets the payload entry whose
            key matches its name, or the first entry. Names must not
            repeat ``env``.
        registry_secret: MysteryBox secret version id (``mbsecver-...``)
            for a private image registry. The payload must have
            ``REGISTRY_USERNAME`` and ``REGISTRY_PASSWORD``.
        spot_policy: Pricing policy id that caps the spot price, e.g.
            ``"pricingpolicy-..."``. Requires ``pricing="spot"``. The
            VM is preempted when the spot price goes above the cap.

    Shared memory is always 16 GiB.

    Example::

        import nebius_serverless as ns

        job = ns.Job(
            ["bash", "-lc", "axolotl train /workspace/data/config.yaml"],
            image="docker.io/axolotlai/axolotl:main-20260309-py3.11-cu128-2.9.1",
            gpu="h100",
            pricing="spot",
            retries=2,  # restart after preemption or a crash
            timeout="24h",
            disk="500Gi",
            volumes=[
                "s3://axolotl-data:/workspace/data:ro",
                "s3://axolotl-artifacts:/workspace/artifacts:rw",
            ],
            env={"WANDB_MODE": "offline"},
            secrets={"HF_TOKEN": "mbsec-..."},  # read from MysteryBox
            name="axolotl-lora",
        )
        print(job.create(dry_run=True))  # inspect the request only
        run = job.create()
        run.wait()
        run.display()
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        image: str,
        gpu: str | None = None,
        platform: str | None = None,
        preset: str | None = None,
        timeout: str | int | float | timedelta,
        pricing: str = "on-demand",
        subnet_id: str | None = None,
        volumes: Sequence[str] | None = None,
        inject_files: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        name: str | None = None,
        disk: str | None = None,
        profile: str | None = None,
        project: str | None = None,
        retries: int = 0,
        working_dir: str | None = None,
        secrets: Mapping[str, str] | None = None,
        registry_secret: str | None = None,
        spot_policy: str | None = None,
    ) -> None:
        self.command = validate_command(command)
        self.image = validate_image(image)
        self.gpu = gpu
        self.platform = validate_optional_text(platform, "platform")
        self.preset = validate_optional_text(preset, "preset")
        check_identity(self.gpu, self.platform, self.preset)
        self.timeout_seconds = timeout_seconds(timeout)
        self.pricing = validate_pricing(pricing)
        reject_spot_on_cpu(self.pricing, self.gpu, self.platform)
        self.subnet_id = validate_optional_text(subnet_id, "subnet_id")
        self.volumes = validate_volumes(volumes)
        self.inject_files = validate_inject_files(inject_files)
        self.env = validate_env(env)
        supplied = validate_optional_text(name, "name")
        self.name = supplied or _generated_name()
        self.disk = validate_disk(validate_optional_text(disk, "disk"))
        self.profile = validate_optional_text(profile, "profile")
        self.project = validate_optional_text(project, "project")
        self.retries = validate_retries(retries)
        self.working_dir = validate_working_dir(working_dir)
        self.secrets = validate_secrets(secrets, self.env)
        self.registry_secret = validate_registry_secret(registry_secret)
        self.spot_policy = validate_spot_policy(spot_policy, self.pricing)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Job":
        """Load a schema-version-1 command document.

        Function documents are rejected. Relative paths inside the
        document are not expanded.
        """
        fields = load_yaml_fields(path)
        return cls(**fields)  # type: ignore[arg-type]

    @overload
    def create(
        self,
        dry_run: bool = False,
        *,
        region: str | None = None,
        project: str | None = None,
        transport: Transport | None = None,
        quiet: bool = False,
    ) -> Run: ...

    @overload
    def create(
        self,
        dry_run: bool = True,
        *,
        region: str | None = None,
        project: str | None = None,
        transport: Transport | None = None,
        quiet: bool = False,
    ) -> dict[str, object]: ...

    def create(
        self,
        dry_run: bool = False,
        *,
        region: str | None = None,
        project: str | None = None,
        transport: Transport | None = None,
        quiet: bool = False,
    ) -> Run | dict[str, object]:
        """Create a cloud job, or return the request when ``dry_run``.

        ``dry_run=True`` resolves project and region and does not call
        create. Region comes from ``region=`` or the profile when either
        names one, otherwise from the project's ``spec.region``. A live
        call returns a ``Run`` after the id is known and does not wait
        for the container to finish. A second call starts another cloud
        job. The live call prints one line once the id is known.
        ``quiet=True`` suppresses that line for this call only.
        ``dry_run=True`` stays silent. ``quiet`` is not sent to Nebius.
        """
        reject_running_loop("create_async")

        def work(progress: Progress) -> Run | dict[str, object]:
            return self._create(
                dry_run=dry_run,
                region=region,
                project=project,
                transport=transport,
                progress=progress,
            )

        return call_progress(quiet=quiet, logger_name=_LOG, work=work)

    async def create_async(
        self,
        dry_run: bool = False,
        *,
        region: str | None = None,
        project: str | None = None,
        transport: Transport | None = None,
        quiet: bool = False,
    ) -> Run | dict[str, object]:
        """Await the same create as ``create``.

        Cancelling this await does not cancel the cloud job.
        ``quiet=True`` suppresses progress for this call only.
        """

        def work(progress: Progress) -> Run | dict[str, object]:
            return self._create(
                dry_run=dry_run,
                region=region,
                project=project,
                transport=transport,
                progress=progress,
            )

        return await call_progress_async(
            quiet=quiet,
            logger_name=_LOG,
            work=work,
        )

    def _create(
        self,
        *,
        dry_run: bool,
        region: str | None,
        project: str | None,
        transport: Transport | None,
        progress: Progress,
    ) -> Run | dict[str, object]:
        project_id, region_id = resolve_location(
            profile=self.profile,
            project=project if project is not None else self.project,
            region=region,
        )
        machine = resolve_machine(
            self.gpu,
            self.platform,
            self.preset,
            region_id,
        )
        if self.pricing == "spot" and machine.cpu:
            raise UnsupportedFeature(
                "Spot pricing is not supported on CPU.",
                operation="create",
            )
        description = describe(
            command=self.command,
            image=self.image,
            project=project_id,
            region=region_id,
            platform=machine.platform,
            preset=machine.preset,
            timeout=self.timeout_seconds,
            pricing=self.pricing,
            subnet_id=self.subnet_id,
            volumes=self.volumes,
            inject_files=self.inject_files,
            env_keys=[key for key, _value in self.env],
            name=self.name,
            disk=self.disk,
            retries=self.retries,
            working_dir=self.working_dir,
            secrets=self.secrets,
            registry_secret=self.registry_secret,
            spot_policy=self.spot_policy,
        )
        if dry_run:
            return description
        chosen = transport or _sdk_transport(self.profile, project_id)
        body = request_body(
            command=self.command,
            image=self.image,
            project=project_id,
            platform=machine.platform,
            preset=machine.preset,
            timeout=self.timeout_seconds,
            pricing=self.pricing,
            subnet_id=self.subnet_id,
            volumes=self.volumes,
            inject_files=load_inject_files(self.inject_files),
            env=self.env,
            name=self.name,
            disk=self.disk,
            retries=self.retries,
            working_dir=self.working_dir,
            secrets=self.secrets,
            registry_secret=self.registry_secret,
            spot_policy=self.spot_policy,
        )
        started = time.perf_counter()
        try:
            result = call_transport(chosen, body)
        except Error as exc:
            log_transport(
                progress,
                job_id=None,
                attempt=1,
                duration_ms=elapsed_ms(started),
                request_id=exc.request_id,
            )
            raise
        duration_ms = elapsed_ms(started)
        job_id = result.resource_id or ""
        log_transport(
            progress,
            job_id=job_id,
            attempt=1,
            duration_ms=duration_ms,
            request_id=result.request_id,
        )
        report(
            Status(
                kind="submitted",
                job_id=job_id,
                name=self.name,
                state=result.state,
                duration_ms=duration_ms,
                request_id=result.request_id,
            ),
            progress,
        )
        return Run(
            job_id,
            name=self.name,
            transport=chosen,
            create_state=result.state,
        )


def _sdk_transport(profile: str | None, project: str) -> Transport:
    from .sdk_transport import SdkTransport

    return SdkTransport(profile=profile, project=project)


def _generated_name() -> str:
    return "job-" + uuid.uuid4().hex[:12]
