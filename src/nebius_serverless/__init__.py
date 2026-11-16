"""Client for Nebius Serverless AI Jobs and Endpoints.

Constructing a ``Job`` or an ``Endpoint`` does not import the Nebius
SDK. The SDK is imported on the call that resolves a profile or
contacts Nebius.
"""

from .endpoint import Endpoint
from .errors import (
    AccessError,
    CapacityError,
    Error,
    KeyConflict,
    SubmissionUncertain,
    TransportError,
    UnsupportedFeature,
    ValidationError,
    WaitTimeout,
)
from .job import Job
from .log import configure_logging, diagnostics
from .run import Run, jobs
from .service import Service, endpoints

__version__ = "0.0.1"

__all__ = [
    "AccessError",
    "CapacityError",
    "Endpoint",
    "Error",
    "Job",
    "KeyConflict",
    "Run",
    "Service",
    "SubmissionUncertain",
    "TransportError",
    "UnsupportedFeature",
    "ValidationError",
    "WaitTimeout",
    "configure_logging",
    "diagnostics",
    "endpoints",
    "jobs",
]
