import os
import re
from pathlib import Path

import pytest

from app.lambda_handler import load_parameters, parameter_map
from shared import ingest
from shared.pipeline import PipelineConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeSSM:

    def __init__(self, values: dict[str, str]):
        self.values = values
        self.requested: list[str] = []

    def get_parameters(self, Names, WithDecryption):

        assert WithDecryption is True
        self.requested = Names

        return {
            "Parameters": [{"Name": n, "Value": self.values[n]} for n in Names if n in self.values],
            "InvalidParameters": [n for n in Names if n not in self.values],
        }


def pins(path: Path) -> dict[str, str]:

    found = {}

    for line in path.read_text().splitlines():
        match = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]+\])?==(\S+)$", line.strip())

        if match:
            found[match.group(1).lower()] = match.group(3)

    return found


def test_parameter_map_reads_env_and_parameter_pairs():

    spec = "QDRANT_URL=/scribe-lab/qdrant_url, QDRANT_API_KEY=/scribe-lab/qdrant_api_key,"

    assert parameter_map(spec) == {
        "QDRANT_URL": "/scribe-lab/qdrant_url",
        "QDRANT_API_KEY": "/scribe-lab/qdrant_api_key",
    }


def test_load_parameters_sets_missing_env_vars_only(monkeypatch):

    monkeypatch.setattr(os, "environ", {"QDRANT_API_KEY": "ya-definida"})
    ssm = FakeSSM({"/scribe-lab/qdrant_url": "https://q.example", "/scribe-lab/qdrant_api_key": "secreta"})

    load_parameters("QDRANT_URL=/scribe-lab/qdrant_url,QDRANT_API_KEY=/scribe-lab/qdrant_api_key", ssm)

    assert os.environ["QDRANT_URL"] == "https://q.example"
    assert os.environ["QDRANT_API_KEY"] == "ya-definida"
    assert ssm.requested == ["/scribe-lab/qdrant_api_key", "/scribe-lab/qdrant_url"]


def test_load_parameters_fails_on_a_missing_parameter():

    with pytest.raises(RuntimeError, match="/scribe-lab/no_existe"):
        load_parameters("X=/scribe-lab/no_existe", FakeSSM({}))


def test_empty_spec_does_not_call_ssm():

    load_parameters("", client=None)


def test_lambda_requirements_use_the_same_pins_as_the_app():

    app_pins = pins(REPO_ROOT / "app" / "requirements.txt")
    lambda_pins = pins(REPO_ROOT / "app" / "requirements-lambda.txt")
    shared = app_pins.keys() & lambda_pins.keys()

    assert {k: lambda_pins[k] for k in shared} == {k: app_pins[k] for k in shared}
    assert not {"torch", "transformers", "uvicorn"} & lambda_pins.keys()


def test_cloud_pipeline_retrieves_like_the_local_one_without_llm():

    local = PipelineConfig.from_yaml(REPO_ROOT / "app" / "config" / "pipeline.yaml")
    cloud = PipelineConfig.from_yaml(REPO_ROOT / "app" / "config" / "pipeline-cloud.yaml")
    retrieval = ("collection", "chunk_size", "chunk_overlap", "method", "reranker", "pool_size", "top_k")

    assert [getattr(cloud, k) for k in retrieval] == [getattr(local, k) for k in retrieval]
    assert (cloud.llm.provider, cloud.memory.enabled, cloud.grounding.enabled) == ("none", False, False)


def test_qdrant_url_selects_the_cloud_client(monkeypatch):

    monkeypatch.setattr(ingest.settings, "qdrant_url", "https://cluster.example:6333")
    monkeypatch.setattr(ingest.settings, "qdrant_api_key", "clave")

    client = ingest.build_qdrant_client()

    assert client._client.rest_uri.startswith("https://cluster.example:6333")
