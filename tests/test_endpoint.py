"""Endpoint recipe, create mapping, Service lifecycle, and YAML."""

import asyncio
import io
import sys
from pathlib import Path

import pytest
from fakes import FakeClock, FakeEndpointTransport

import nebius_serverless as ns
from nebius_serverless.sdk_transport import build_endpoint_request

VLLM = "vllm/vllm-openai:v0.19.1"
SECRET = "super-secret-value"
TOKEN = "bearer-token-value"
SERVE = [
    "bash",
    "-c",
    "python3 -m vllm.entrypoints.openai.api_server "
    "--model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000",
]


def _endpoint(**kwargs: object) -> ns.Endpoint:
    fields: dict[str, object] = {
        "image": VLLM,
        "port": 8000,
        "gpu": "l40s",
        "env": {"HF_TOKEN": SECRET},
    }
    fields.update(kwargs)
    command = fields.pop("command", None)
    return ns.Endpoint(command, **fields)  # type: ignore[arg-type]


def _create(fake: FakeEndpointTransport, **kwargs: object) -> ns.Service:
    service = _endpoint(**kwargs).create(
        region="eu-north1",
        project="project-1",
        transport=fake,
    )
    assert isinstance(service, ns.Service)
    return service


def _spec(body: dict[str, object]) -> dict[str, object]:
    spec = body["spec"]
    assert isinstance(spec, dict)
    return spec


def _install_clock(monkeypatch: pytest.MonkeyPatch, clock: FakeClock) -> None:
    monkeypatch.setattr(
        "nebius_serverless.service.monotonic", clock.monotonic
    )
    monkeypatch.setattr("nebius_serverless.service.sleep", clock.sleep)


# Recipe validation


def test_constructing_does_not_contact_nebius() -> None:
    endpoint = _endpoint(name="qwen-tiny")
    assert endpoint.command is None
    assert endpoint.port == 8000
    assert endpoint.auth == "none"
    assert endpoint.token is None
    assert endpoint.pricing == "on-demand"
    assert endpoint.env == (("HF_TOKEN", SECRET),)
    assert endpoint.name == "qwen-tiny"
    generated = _endpoint().name
    assert generated.startswith("endpoint-")
    assert len(generated) == len("endpoint-") + 12


@pytest.mark.parametrize("port", [0, 65536, "8000", True, 80.0, None])
def test_port_must_be_an_integer_in_range(port: object) -> None:
    with pytest.raises(ns.ValidationError, match="port") as caught:
        _endpoint(port=port)
    assert caught.value.operation == "Endpoint"


def test_auth_and_token_rules() -> None:
    with pytest.raises(ns.ValidationError, match="auth"):
        _endpoint(auth="basic")
    with pytest.raises(ns.ValidationError, match="auth='token'") as caught:
        _endpoint(token=TOKEN)
    assert TOKEN not in str(caught.value)
    assert _endpoint(auth="token", token=TOKEN).token == TOKEN
    assert TOKEN not in repr(_endpoint(auth="token", token=TOKEN))


def test_command_rules() -> None:
    with pytest.raises(ns.ValidationError, match="shell string") as caught:
        _endpoint(command="python serve.py")
    assert caught.value.operation == "Endpoint"
    with pytest.raises(ns.ValidationError):
        _endpoint(command=[])
    assert _endpoint(command=["serve"]).command == ("serve",)


def test_gpu_cpu_and_spot_on_cpu() -> None:
    with pytest.raises(ns.ValidationError, match="Omit gpu"):
        _endpoint(gpu="cpu")
    with pytest.raises(ns.UnsupportedFeature, match="Spot"):
        _endpoint(gpu=None, pricing="spot")
    with pytest.raises(ns.ValidationError, match="Unknown gpu"):
        _endpoint(gpu="a100")


# Create mapping


def test_create_body_keeps_entrypoint_when_command_is_omitted() -> None:
    fake = FakeEndpointTransport()
    service = _create(fake, name="qwen-tiny")
    assert service.id == "endpoint-1"
    assert service.name == "qwen-tiny"
    assert service.status == "PROVISIONING"
    assert service.record is None
    assert service.url is None
    assert service.token is None
    assert fake.gets == []
    body = fake.bodies[0]
    assert body["metadata"] == {"parent_id": "project-1", "name": "qwen-tiny"}
    spec = _spec(body)
    assert "container_command" not in spec
    assert "args" not in spec
    assert spec["image"] == VLLM
    assert spec["ports"] == [{"container_port": 8000, "protocol": "HTTP"}]
    assert spec["platform"] == "gpu-l40s-a"
    assert spec["preset"] == "1gpu-8vcpu-32gb"
    assert spec["disk"] == "250Gi"
    assert spec["shm_size_bytes"] == 16 * 1024**3
    assert "auth_token" not in spec
    assert "preemptible" not in spec
    assert spec["environment_variables"] == [
        {"name": "HF_TOKEN", "value": SECRET}
    ]


def test_create_body_with_command_spot_and_token() -> None:
    fake = FakeEndpointTransport()
    service = _create(
        fake,
        command=SERVE,
        auth="token",
        token=TOKEN,
        pricing="spot",
        disk="500Gi",
        volumes=["s3://models:/models:ro"],
        subnet_id="subnet-1",
    )
    assert service.token == TOKEN
    spec = _spec(fake.bodies[0])
    assert spec["container_command"] == "bash"
    assert spec["args"] == (
        "-c 'python3 -m vllm.entrypoints.openai.api_server "
        "--model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000'"
    )
    assert spec["auth_token"] == TOKEN
    assert spec["preemptible"] is True
    assert spec["follows_spot_price"] is True
    assert spec["disk"] == "500Gi"
    assert spec["subnet_id"] == "subnet-1"
    assert spec["volumes"] == [
        {
            "source": "s3://models",
            "container_path": "/models",
            "mode": "ro",
        }
    ]


def test_token_auth_without_token_generates_one_per_create() -> None:
    fake = FakeEndpointTransport()
    endpoint = _endpoint(auth="token")
    first = endpoint.create(
        region="eu-north1", project="p", transport=fake
    )
    second = endpoint.create(
        region="eu-north1", project="p", transport=fake
    )
    assert isinstance(first, ns.Service)
    assert isinstance(second, ns.Service)
    assert first.token and second.token
    assert first.token != second.token
    assert _spec(fake.bodies[0])["auth_token"] == first.token
    assert _spec(fake.bodies[1])["auth_token"] == second.token
    assert endpoint.token is None
    assert second.id == "endpoint-2"


def test_dry_run_hides_env_values_and_token() -> None:
    fake = FakeEndpointTransport()
    described = _endpoint(
        command=SERVE,
        auth="token",
        token=TOKEN,
        name="qwen-tiny",
    ).create(
        dry_run=True,
        region="eu-north1",
        project="project-1",
        transport=fake,
    )
    assert fake.bodies == []
    assert described == {
        "image": VLLM,
        "command": SERVE,
        "args": described["args"],
        "port": 8000,
        "project": "project-1",
        "region": "eu-north1",
        "platform": "gpu-l40s-a",
        "preset": "1gpu-8vcpu-32gb",
        "pricing": "on-demand",
        "auth": "token",
        "token": "provided",
        "env": ["HF_TOKEN"],
        "name": "qwen-tiny",
        "disk": "250Gi",
        "shm": "16Gi",
    }
    assert SECRET not in repr(described)
    assert TOKEN not in repr(described)
    generated = _endpoint(auth="token").create(
        dry_run=True, region="eu-north1", project="p"
    )
    assert isinstance(generated, dict)
    assert generated["token"] == "generated"
    assert "command" not in generated


def test_gpu_region_mismatch_names_endpoint_recovery() -> None:
    with pytest.raises(ns.ValidationError, match="not available") as caught:
        _endpoint().create(dry_run=True, region="us-central1", project="p")
    assert "endpoints.get" in caught.value.recovery
    assert "jobs.get" not in caught.value.recovery


def test_create_failures_map_to_endpoint_errors() -> None:
    fake = FakeEndpointTransport()
    fake.failure = "capacity"
    with pytest.raises(ns.CapacityError, match="endpoint"):
        _create(fake)
    fake.failure = None
    fake.drop_id = True
    with pytest.raises(ns.SubmissionUncertain) as caught:
        _create(fake)
    assert "endpoints.get" in caught.value.recovery
    assert len(fake.bodies) == 2


def test_build_endpoint_request_uses_sdk_types() -> None:
    fake = FakeEndpointTransport()
    _create(
        fake,
        command=SERVE,
        auth="token",
        token=TOKEN,
        pricing="spot",
        volumes=["data:/work:ro"],
        name="qwen-tiny",
    )
    request = build_endpoint_request(fake.bodies[0])
    spec = request.spec
    assert request.metadata.parent_id == "project-1"
    assert request.metadata.name == "qwen-tiny"
    assert spec.image == VLLM
    assert spec.container_command == "bash"
    assert spec.platform == "gpu-l40s-a"
    assert spec.auth_token == TOKEN
    assert spec.preemptible is True
    assert len(spec.ports) == 1
    assert spec.ports[0].container_port == 8000
    assert spec.ports[0].protocol.name == "HTTP"
    assert spec.disk.size_bytes == 250 * 1024**3
    assert spec.environment_variables[0].name == "HF_TOKEN"
    assert spec.volumes[0].container_path == "/work"
    assert spec.volumes[0].mode.name == "READ_ONLY"


def test_build_endpoint_request_without_command() -> None:
    fake = FakeEndpointTransport()
    _create(fake)
    spec = build_endpoint_request(fake.bodies[0]).spec
    assert spec.container_command == ""
    assert spec.args == ""
    assert spec.auth_token == ""


# Service lifecycle


def test_get_sets_record_status_and_url() -> None:
    fake = FakeEndpointTransport()
    fake.states = ["ERROR"]
    service = _create(fake)
    record = service.get()
    assert record["status"]["state"] == "ERROR"
    assert service.status == "ERROR"
    assert service.record is record
    assert service.url == "https://ep.example.nebius.cloud"
    assert fake.deletes == []


def test_url_ignores_non_https_entries() -> None:
    fake = FakeEndpointTransport()
    fake.urls = ["tls://ep.example:443"]
    service = _create(fake)
    service.get()
    assert service.url is None


def test_wait_polls_until_running_with_a_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    _install_clock(monkeypatch, clock)
    fake = FakeEndpointTransport()
    fake.states = ["PROVISIONING", "IMAGE_PULLING", "RUNNING", "RUNNING"]
    fake.urls = [None, None, None, "https://ep.example.nebius.cloud"]
    service = _create(fake)
    record = service.wait()
    assert record["status"]["state"] == "RUNNING"
    assert service.url == "https://ep.example.nebius.cloud"
    assert len(fake.gets) == 4
    assert fake.deletes == []
    assert clock.now == 15.0


@pytest.mark.parametrize("state", ["ERROR", "STOPPED"])
def test_wait_returns_stopped_states_as_data(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    _install_clock(monkeypatch, FakeClock())
    fake = FakeEndpointTransport()
    fake.states = ["STARTING", state]
    fake.urls = [None]
    service = _create(fake)
    record = service.wait()
    assert record["status"]["state"] == state
    assert service.status == state
    assert service.url is None


def test_wait_timeout_zero_raises_and_does_not_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    _install_clock(monkeypatch, clock)
    fake = FakeEndpointTransport()
    fake.states = ["RUNNING"]
    fake.urls = [None]
    service = _create(fake, auth="token", token=TOKEN)
    with pytest.raises(ns.WaitTimeout) as caught:
        service.wait(timeout=0)
    assert caught.value.run_id == "endpoint-1"
    assert "endpoints.get" in caught.value.recovery
    assert TOKEN not in str(caught.value)
    assert SECRET not in str(caught.value)
    assert fake.gets == ["endpoint-1"]
    assert fake.deletes == []
    assert clock.now == 0.0


def test_wait_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    _install_clock(monkeypatch, clock)
    fake = FakeEndpointTransport()
    fake.states = ["PROVISIONING"]
    service = _create(fake)
    with pytest.raises(ns.WaitTimeout):
        service.wait(timeout=12)
    assert clock.now == 12.0
    with pytest.raises(ns.ValidationError):
        service.wait(timeout=-1)


def test_delete_requests_deletion_once() -> None:
    fake = FakeEndpointTransport()
    service = _create(fake)
    payload = service.delete()
    assert payload == {"id": "endpoint-1", "resource_id": "endpoint-1"}
    assert fake.deletes == ["endpoint-1"]
    assert not hasattr(service, "cancel")


def test_stop_requests_a_stop_once_and_does_not_delete() -> None:
    fake = FakeEndpointTransport()
    service = _create(fake)
    payload = service.stop()
    assert payload == {"id": "endpoint-1", "resource_id": "endpoint-1"}
    assert fake.stops == ["endpoint-1"]
    assert fake.deletes == []


def test_endpoints_get_reloads_without_creating() -> None:
    fake = FakeEndpointTransport()
    service = _create(fake, auth="token")
    assert service.token is not None
    loaded = ns.endpoints.get(service.id, transport=fake)
    assert isinstance(loaded, ns.Service)
    assert loaded.id == service.id
    assert loaded.name == "saved-endpoint"
    assert loaded.status == "RUNNING"
    assert loaded.record is not None
    assert loaded.url == "https://ep.example.nebius.cloud"
    assert loaded.token is None
    assert len(fake.bodies) == 1


def test_display_prints_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    fake = FakeEndpointTransport()
    service = _create(fake)
    service.get(quiet=True)
    service.display()
    text = stream.getvalue()
    assert "endpoint-1" in text
    assert "https://ep.example.nebius.cloud" in text


def test_progress_names_the_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch, FakeClock())
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    fake = FakeEndpointTransport()
    fake.states = ["STARTING", "RUNNING"]
    service = _create(fake, name="qwen-tiny")
    service.wait()
    service.stop()
    service.delete()
    text = stream.getvalue()
    assert "Submitted endpoint endpoint-1 qwen-tiny" in text
    assert "endpoint-1 RUNNING" in text
    assert "https://ep.example.nebius.cloud" in text
    assert "Stop accepted for endpoint-1" in text
    assert "Delete accepted for endpoint-1" in text
    assert "jobs.get" not in text


def test_error_state_progress_says_it_still_bills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    fake = FakeEndpointTransport()
    fake.states = ["ERROR"]
    _create(fake).wait(timeout=0)
    text = stream.getvalue()
    assert "ERROR" in text
    assert "still bills until delete" in text


# Async twins


def test_sync_calls_inside_a_loop_name_the_twin() -> None:
    fake = FakeEndpointTransport()
    service = _create(fake)

    async def inside() -> None:
        with pytest.raises(ns.ValidationError, match="create_async"):
            _endpoint().create(dry_run=True, region="eu-north1", project="p")
        with pytest.raises(ns.ValidationError, match="wait_async"):
            service.wait()
        with pytest.raises(ns.ValidationError, match="stop_async"):
            service.stop()
        with pytest.raises(ns.ValidationError, match="delete_async"):
            service.delete()
        with pytest.raises(
            ns.ValidationError, match="endpoints.get_async"
        ):
            ns.endpoints.get(service.id, transport=fake)

    asyncio.run(inside())
    assert fake.stops == []
    assert fake.deletes == []


def test_awaited_twins(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_clock(monkeypatch, FakeClock())
    fake = FakeEndpointTransport()
    endpoint = _endpoint()

    async def flow() -> None:
        service = await endpoint.create_async(
            region="eu-north1",
            project="p",
            transport=fake,
        )
        assert isinstance(service, ns.Service)
        record = await service.wait_async()
        assert record["status"]["state"] == "RUNNING"
        assert (await service.get_async())["status"]["state"] == "RUNNING"
        loaded = await ns.endpoints.get_async(service.id, transport=fake)
        assert loaded.id == service.id
        await service.stop_async()
        await service.delete_async()

    asyncio.run(flow())
    assert fake.stops == ["endpoint-1"]
    assert fake.deletes == ["endpoint-1"]


# YAML


ENDPOINT_YAML = """\
schema_version: 1
kind: endpoint
image: vllm/vllm-openai:v0.19.1
command:
  - bash
  - -c
  - >-
    python3 -m vllm.entrypoints.openai.api_server
    --model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000
gpu: l40s
port: 8000
auth: token
disk: 500Gi
pricing: spot
name: qwen-tiny
"""


def test_from_yaml_matches_the_python_recipe(tmp_path: Path) -> None:
    path = tmp_path / "endpoint.yaml"
    path.write_text(ENDPOINT_YAML)
    endpoint = ns.Endpoint.from_yaml(path)
    assert endpoint.command == tuple(SERVE)
    assert endpoint.port == 8000
    assert endpoint.gpu == "l40s"
    assert endpoint.auth == "token"
    assert endpoint.pricing == "spot"
    assert endpoint.disk == "500Gi"
    assert endpoint.name == "qwen-tiny"


def test_from_yaml_without_command(tmp_path: Path) -> None:
    path = tmp_path / "endpoint.yaml"
    path.write_text(
        "schema_version: 1\nkind: endpoint\nimage: nginx:1.27\nport: 80\n"
    )
    endpoint = ns.Endpoint.from_yaml(path)
    assert endpoint.command is None
    assert endpoint.gpu is None


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("schema_version: 1\nkind: job\nimage: a:b\nport: 80\n", "Job"),
        (
            "schema_version: 1\nkind: endpoint\nimage: a:b\nport: 80\n"
            "timeout: 1h\n",
            "Unknown YAML keys: timeout",
        ),
        ("schema_version: 2\nkind: endpoint\nimage: a:b\nport: 80\n",
         "schema_version"),
        ("schema_version: 1\nkind: endpoint\nimage: a:b\nport: '80'\n",
         "port"),
        ("schema_version: 1\nkind: endpoint\nimage: a:b\nport: 80\n"
         "gpu: cpu\n", "Omit gpu"),
        ("schema_version: 1\nkind: endpoint\nimage: &i a:b\nport: 80\n",
         "anchors"),
    ],
)
def test_from_yaml_rejections(
    tmp_path: Path,
    text: str,
    match: str,
) -> None:
    path = tmp_path / "endpoint.yaml"
    path.write_text(text)
    with pytest.raises(ns.ValidationError, match=match):
        ns.Endpoint.from_yaml(path)


def test_job_loader_points_endpoint_documents_elsewhere(
    tmp_path: Path,
) -> None:
    path = tmp_path / "endpoint.yaml"
    path.write_text(ENDPOINT_YAML)
    with pytest.raises(ns.ValidationError, match="Endpoint.from_yaml"):
        ns.Job.from_yaml(path)


# SDK adapter


class _Operation:
    def __init__(self, resource_id: str) -> None:
        self.resource_id = resource_id


class _Pending:
    def __init__(self, value: object) -> None:
        self._value = value

    def wait(self) -> object:
        return self._value


class _FakeEndpointClient:
    def __init__(self) -> None:
        self.created: list[object] = []
        self.stopped: list[str] = []
        self.deleted: list[str] = []

    def create(self, request: object) -> _Pending:
        self.created.append(request)
        return _Pending(_Operation("ep-sdk-1"))

    def get(self, request: object) -> _Pending:
        import nebius.api.nebius.ai.v1 as v1
        from nebius.api.nebius.common.v1 import ResourceMetadata

        return _Pending(
            v1.Endpoint(
                metadata=ResourceMetadata(id=request.id, name="qwen"),
                spec=v1.EndpointSpec(image=VLLM, auth_token=TOKEN),
                status=v1.EndpointStatus(
                    state=v1.EndpointStatus.State.RUNNING,
                    public_endpoints=["https://ep-sdk-1.nebius.cloud"],
                ),
            )
        )

    def stop(self, request: object) -> _Pending:
        self.stopped.append(request.id)
        return _Pending(_Operation(request.id))

    def delete(self, request: object) -> _Pending:
        self.deleted.append(request.id)
        return _Pending(_Operation(request.id))


def test_sdk_endpoint_transport_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nebius_serverless.sdk_transport import SdkEndpointTransport

    client = _FakeEndpointClient()
    monkeypatch.setattr(SdkEndpointTransport, "_client", lambda self: client)
    _install_clock(monkeypatch, FakeClock())
    transport = SdkEndpointTransport(profile=None, project="project-1")
    service = _create(transport, auth="token", token=TOKEN)  # type: ignore[arg-type]
    assert service.id == "ep-sdk-1"
    assert client.created[0].spec.auth_token == TOKEN
    record = service.wait()
    assert record["status"]["state"] == "RUNNING"
    assert "auth_token" not in record["spec"]
    assert TOKEN not in repr(record)
    assert service.url == "https://ep-sdk-1.nebius.cloud"
    assert service.stop() == {"id": "ep-sdk-1", "resource_id": "ep-sdk-1"}
    assert client.stopped == ["ep-sdk-1"]
    assert service.delete() == {
        "id": "ep-sdk-1",
        "resource_id": "ep-sdk-1",
    }
    assert client.deleted == ["ep-sdk-1"]


def test_quickstart_matches_the_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    root = Path(__file__).resolve().parents[1] / "examples"
    spec = importlib.util.spec_from_file_location(
        "endpoint_quickstart", root / "endpoint_quickstart.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    built: list[ns.Endpoint] = []
    original = ns.Endpoint.create

    def capture(self: ns.Endpoint, *args: object, **kwargs: object) -> None:
        built.append(self)
        raise SystemExit("stop before create")

    monkeypatch.setattr(ns.Endpoint, "create", capture)
    with pytest.raises(SystemExit):
        module.main()
    from_file = ns.Endpoint.from_yaml(
        root / "from_yaml" / "vllm_endpoint.yaml"
    )

    def described(endpoint: ns.Endpoint) -> object:
        return original(
            endpoint, dry_run=True, region="eu-north1", project="p"
        )

    assert described(built[0]) == described(from_file)
