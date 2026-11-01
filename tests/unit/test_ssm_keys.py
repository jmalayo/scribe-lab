import json
import stat

import boto3
import pytest
from botocore.stub import ANY, Stubber

from app import deps, manage_keys, ssm_keys
from app.auth import KeyStore

PARAMETER = "/scribe-lab/api_keys"


@pytest.fixture
def ssm():

    client = boto3.client("ssm", region_name="us-east-2", aws_access_key_id="test", aws_secret_access_key="test")

    with Stubber(client) as stubber:
        yield client, stubber


def stored(entries):

    return {"Parameter": {"Name": PARAMETER, "Type": "SecureString", "Value": json.dumps(entries)}}


def test_missing_parameter_means_no_keys(ssm):

    client, stubber = ssm
    stubber.add_client_error("get_parameter", service_error_code="ParameterNotFound")

    assert ssm_keys.load_active_keys(PARAMETER, client) == []


def test_created_key_is_found_and_revoked_key_is_not(ssm):

    client, stubber = ssm
    key, entries = ssm_keys.new_key([], "demo-publica", 30)
    _, entries = ssm_keys.new_key(entries, "canario-nube", 10)
    _, revoked = ssm_keys.revoke_owner(entries, "canario-nube")

    stubber.add_response("get_parameter", stored(revoked), {"Name": PARAMETER, "WithDecryption": True})
    store = KeyStore(lambda: ssm_keys.load_active_keys(PARAMETER, client))

    found = store.find(key)

    assert (found.dueno, found.limite_por_minuto) == ("demo-publica", 30)
    assert [k.dueno for k in store.active_keys()] == ["demo-publica"]
    assert key not in json.dumps(revoked)


def test_entries_are_written_as_a_secure_string(ssm):

    client, stubber = ssm
    _, entries = ssm_keys.new_key([], "demo-publica", 30)
    stubber.add_response(
        "put_parameter",
        {"Version": 2},
        {"Name": PARAMETER, "Value": ANY, "Type": "SecureString", "Overwrite": True},
    )

    ssm_keys.write_entries(client, PARAMETER, entries)


def test_revoke_counts_only_active_keys_of_the_owner():

    _, entries = ssm_keys.new_key([], "demo-publica", 30)
    _, entries = ssm_keys.new_key(entries, "demo-publica", 30)
    _, entries = ssm_keys.new_key(entries, "canario-nube", 10)

    first, entries = ssm_keys.revoke_owner(entries, "demo-publica")
    second, entries = ssm_keys.revoke_owner(entries, "demo-publica")

    assert (first, second) == (2, 0)
    assert [r[1] for r in ssm_keys.active_rows(entries)] == ["canario-nube"]


def test_key_is_saved_to_env_without_duplicating_the_variable(tmp_path):

    env = tmp_path / ".env"
    env.write_text("OTRA=1\nCANARIO_NUBE_API_KEY=vieja\n")

    manage_keys.save_to_env(env, "CANARIO_NUBE_API_KEY", "sk_scribe_nueva")

    assert env.read_text() == "OTRA=1\nCANARIO_NUBE_API_KEY=sk_scribe_nueva\n"
    assert stat.S_IMODE(env.stat().st_mode) == 0o600


def test_ssm_store_does_not_need_a_database(monkeypatch):

    monkeypatch.setattr(deps.settings, "api_key_store", "ssm")
    monkeypatch.setattr(ssm_keys, "load_active_keys", lambda parameter: [("id", "x", "h", 1)] if parameter else [])

    store = deps.build_key_store(None)

    assert store.load() == [("id", "x", "h", 1)]
