import os


def parameter_map(spec: str) -> dict[str, str]:

    pairs = (item.split("=", 1) for item in spec.split(",") if item.strip())

    return {env.strip(): name.strip() for env, name in pairs}


def load_parameters(spec: str, client=None) -> None:

    names = parameter_map(spec)

    if not names:
        return

    if client is None:
        import boto3

        client = boto3.client("ssm")

    response = client.get_parameters(Names=sorted(set(names.values())), WithDecryption=True)

    if response.get("InvalidParameters"):
        raise RuntimeError(f"Parámetros de SSM inexistentes: {', '.join(response['InvalidParameters'])}")

    values = {p["Name"]: p["Value"] for p in response["Parameters"]}

    for env, name in names.items():
        os.environ.setdefault(env, values[name])


def build_handler():

    load_parameters(os.environ.get("SSM_PARAMETERS", ""))

    from mangum import Mangum

    from app.deps import startup
    from app.main import app
    from app.telemetry import flush

    startup(app)
    adapter = Mangum(app, lifespan="off")

    def handler(event, context):

        try:
            return adapter(event, context)

        finally:
            flush()

    return handler


handler = build_handler() if os.environ.get("AWS_LAMBDA_FUNCTION_NAME") else None
