"""A local endpoint recipe. Constructing it does not contact Nebius."""

import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import overload

from .aliases import resolve_machine
from .endpoint_recipe import (
    describe_endpoint,
    endpoint_request_body,
    generate_token,
    validate_auth,
    validate_optional_command,
    validate_port,
    validate_token,
    validate_token_secret,
)
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
    load_inject_files,
    reject_spot_on_cpu,
    validate_disk,
    validate_env,
    validate_image,
    validate_inject_files,
    validate_optional_text,
    validate_pricing,
    validate_registry_secret,
    validate_secrets,
    validate_spot_policy,
    validate_volumes,
    validate_working_dir,
)
from .service import Service
from .status import Status
from .transport import EndpointTransport, call_transport
from .yaml_job import load_endpoint_yaml_fields

_LOG = "nebius_serverless.endpoint"
_OPERATION = "Endpoint"


class Endpoint:
    """Recipe for one ``nebius ai endpoint create``.

    An endpoint is a container that stays up and answers on a managed
    ``https://`` URL. Constructing an ``Endpoint``, loading YAML, or
    importing the package does not authenticate, upload, or create.
    Each created endpoint bills until its own ``delete``.

    Args:
        command: Optional list of strings that overrides the image
            entrypoint, e.g. ``["bash", "-c", "vllm serve ..."]``.
            ``None`` (default) keeps the entrypoint. The first item is
            the container command; the rest are POSIX-quoted into one
            ``args`` string. A shell string is rejected.
        image: ``name:tag`` or ``registry/path@sha256:<digest>``, e.g.
            ``"vllm/vllm-openai:v0.19.1"``. Sent unchanged.
        port: Container HTTP port, 1 to 65535, e.g. ``8000``. Served on
            a managed ``https://`` URL (``Service.url``).
        gpu: GPU alias such as ``"h100"``, ``"h200"``, ``"b200"``,
            ``"b300"``, ``"l40s"`` or ``"rtx"``. Picks the smallest
            one-GPU preset in the project region. Omit it for CPU.
        platform: Explicit platform id, e.g. ``"gpu-l40s-a"``. Overrides
            ``gpu``.
        preset: Explicit preset id, e.g. ``"1gpu-8vcpu-32gb"``. When
            omitted, the smallest preset for the platform is used.
        pricing: ``"on-demand"`` (default) or ``"spot"``. Spot is
            preemptible and not available on CPU.
        auth: ``"none"`` (default, public URL) or ``"token"`` (bearer
            token required).
        token: Bearer token for ``auth="token"``. When omitted, a random
            token is generated on each ``create`` and returned as
            ``Service.token``. Setting it with ``auth="none"`` raises.
        subnet_id: Subnet id, e.g. ``"vpcsubnet-..."``. Only needed
            when the project has more than one subnet.
        volumes: Mount strings ``source:/container/path[:ro|rw]`` or
            ``s3://bucket:/container/path[:ro|rw]`` (no ``:profile``
            suffix), e.g.
            ``["s3://my-models:/models:ro"]``. Mode defaults to ``rw``.
        inject_files: ``local_path:/container/path`` strings, e.g.
            ``["./config.yaml:/app/config.yaml"]``. Each file is read on
            ``create`` and must be at most 64 KiB.
        env: Plain environment variables, e.g.
            ``{"WANDB_MODE": "offline"}``. Values are hidden from dry-run
            output and error text. Use ``secrets`` for tokens.
        name: Console label. Defaults to ``endpoint-`` plus 12 hex
            chars. It is not a lookup key.
        disk: Boot disk size, e.g. ``"500Gi"``. Defaults to ``"250Gi"``.
        profile: Nebius CLI profile name. Defaults to the active profile.
        project: Project id, e.g. ``"project-..."``. Overrides the
            profile's project.
        token_secret: MysteryBox secret id or version id whose payload
            has ``AUTH_TOKEN``. Requires ``auth="token"`` and excludes
            ``token``. No token is generated and ``Service.token`` is
            ``None``.
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

    Example::

        import nebius_serverless as ns

        endpoint = ns.Endpoint(
            command=[
                "bash",
                "-c",
                "python3 -m vllm.entrypoints.openai.api_server "
                "--model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000",
            ],
            image="vllm/vllm-openai:v0.19.1",
            port=8000,
            gpu="l40s",
            auth="token",
            disk="500Gi",
            secrets={"HF_TOKEN": "mbsec-..."},  # read from MysteryBox
            name="qwen-tiny",
        )
        print(endpoint.create(dry_run=True))  # inspect the request only
        service = endpoint.create()
        service.wait()
        print(service.url, service.token)
        service.delete()  # stops billing
    """

    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        image: str,
        port: int,
        gpu: str | None = None,
        platform: str | None = None,
        preset: str | None = None,
        pricing: str = "on-demand",
        auth: str = "none",
        token: str | None = None,
        subnet_id: str | None = None,
        volumes: Sequence[str] | None = None,
        inject_files: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        name: str | None = None,
        disk: str | None = None,
        profile: str | None = None,
        project: str | None = None,
        token_secret: str | None = None,
        working_dir: str | None = None,
        secrets: Mapping[str, str] | None = None,
        registry_secret: str | None = None,
        spot_policy: str | None = None,
    ) -> None:
        op = _OPERATION
        self.command = validate_optional_command(command)
        self.image = validate_image(image, operation=op)
        self.port = validate_port(port)
        self.gpu = gpu
        self.platform = validate_optional_text(
            platform, "platform", operation=op
        )
        self.preset = validate_optional_text(preset, "preset", operation=op)
        check_identity(self.gpu, self.platform, self.preset, operation=op)
        self.pricing = validate_pricing(pricing, operation=op)
        reject_spot_on_cpu(
            self.pricing, self.gpu, self.platform, operation=op
        )
        self.auth = validate_auth(auth)
        self.token = validate_token(token, self.auth)
        self.token_secret = validate_token_secret(
            token_secret, self.auth, self.token
        )
        self.subnet_id = validate_optional_text(
            subnet_id, "subnet_id", operation=op
        )
        self.volumes = validate_volumes(volumes, operation=op)
        self.inject_files = validate_inject_files(inject_files, operation=op)
        self.env = validate_env(env, operation=op)
        supplied = validate_optional_text(name, "name", operation=op)
        self.name = supplied or _generated_name()
        self.disk = validate_disk(
            validate_optional_text(disk, "disk", operation=op),
            operation=op,
        )
        self.profile = validate_optional_text(profile, "profile", operation=op)
        self.project = validate_optional_text(project, "project", operation=op)
        self.working_dir = validate_working_dir(working_dir, operation=op)
        self.secrets = validate_secrets(secrets, self.env, operation=op)
        self.registry_secret = validate_registry_secret(
            registry_secret, operation=op
        )
        self.spot_policy = validate_spot_policy(
            spot_policy, self.pricing, operation=op
        )

    def __repr__(self) -> str:
        token = "" if self.token is None else ", token=<hidden>"
        return (
            f"Endpoint(image={self.image!r}, port={self.port!r}, "
            f"name={self.name!r}, auth={self.auth!r}{token})"
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Endpoint":
        """Load a schema-version-1 endpoint document.

        Relative strings inside the document are not expanded.
        """
        fields = load_endpoint_yaml_fields(path)
        return cls(**fields)  # type: ignore[arg-type]

    @overload
    def create(
        self,
        dry_run: bool = False,
        *,
        region: str | None = None,
        project: str | None = None,
        transport: EndpointTransport | None = None,
        quiet: bool = False,
    ) -> Service: ...

    @overload
    def create(
        self,
        dry_run: bool = True,
        *,
        region: str | None = None,
        project: str | None = None,
        transport: EndpointTransport | None = None,
        quiet: bool = False,
    ) -> dict[str, object]: ...

    def create(
        self,
        dry_run: bool = False,
        *,
        region: str | None = None,
        project: str | None = None,
        transport: EndpointTransport | None = None,
        quiet: bool = False,
    ) -> Service | dict[str, object]:
        """Create a cloud endpoint, or return the request when ``dry_run``.

        A live call returns a ``Service`` once Nebius has accepted the
        endpoint and the id is known. It does not wait for the URL.
        Acceptance is not readiness. A second call starts another
        endpoint. ``quiet=True`` suppresses progress for this call only.
        """
        reject_running_loop("create_async")

        def work(progress: Progress) -> Service | dict[str, object]:
            return self._create(
                dry_run=dry_run,
                region=region,
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

    async def create_async(
        self,
        dry_run: bool = False,
        *,
        region: str | None = None,
        project: str | None = None,
        transport: EndpointTransport | None = None,
        quiet: bool = False,
    ) -> Service | dict[str, object]:
        """Await the same create as ``create``.

        Cancelling this await leaves the endpoint running when create
        was already accepted.
        """

        def work(progress: Progress) -> Service | dict[str, object]:
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
            noun="endpoint",
        )

    def _create(
        self,
        *,
        dry_run: bool,
        region: str | None,
        project: str | None,
        transport: EndpointTransport | None,
        progress: Progress,
    ) -> Service | dict[str, object]:
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
        description = describe_endpoint(
            command=self.command,
            image=self.image,
            port=self.port,
            project=project_id,
            region=region_id,
            platform=machine.platform,
            preset=machine.preset,
            pricing=self.pricing,
            auth=self.auth,
            token_supplied=self.token is not None,
            subnet_id=self.subnet_id,
            volumes=self.volumes,
            inject_files=self.inject_files,
            env_keys=[key for key, _value in self.env],
            name=self.name,
            disk=self.disk,
            token_secret=self.token_secret,
            working_dir=self.working_dir,
            secrets=self.secrets,
            registry_secret=self.registry_secret,
            spot_policy=self.spot_policy,
        )
        if dry_run:
            return description
        chosen = transport or _sdk_transport(self.profile, project_id)
        token = None
        if self.auth == "token" and self.token_secret is None:
            token = self.token or generate_token()
        body = endpoint_request_body(
            command=self.command,
            image=self.image,
            port=self.port,
            project=project_id,
            platform=machine.platform,
            preset=machine.preset,
            pricing=self.pricing,
            token=token,
            subnet_id=self.subnet_id,
            volumes=self.volumes,
            inject_files=load_inject_files(self.inject_files),
            env=self.env,
            name=self.name,
            disk=self.disk,
            token_secret=self.token_secret,
            working_dir=self.working_dir,
            secrets=self.secrets,
            registry_secret=self.registry_secret,
            spot_policy=self.spot_policy,
        )
        started = time.perf_counter()
        try:
            result = call_transport(chosen, body, noun="endpoint")
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
        endpoint_id = result.resource_id or ""
        log_transport(
            progress,
            job_id=endpoint_id,
            attempt=1,
            duration_ms=duration_ms,
            request_id=result.request_id,
        )
        report(
            Status(
                kind="submitted",
                job_id=endpoint_id,
                name=self.name,
                state=result.state,
                duration_ms=duration_ms,
                request_id=result.request_id,
                noun="endpoint",
            ),
            progress,
        )
        return Service(
            endpoint_id,
            name=self.name,
            transport=chosen,
            create_state=result.state,
            token=token,
        )


def _sdk_transport(profile: str | None, project: str) -> EndpointTransport:
    from .sdk_transport import SdkEndpointTransport

    return SdkEndpointTransport(profile=profile, project=project)


def _generated_name() -> str:
    return "endpoint-" + uuid.uuid4().hex[:12]
