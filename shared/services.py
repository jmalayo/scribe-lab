import subprocess

import requests


def service_info(url: str) -> dict:
    return requests.get(f"{url}/health", timeout=10).json()


def container_id(service: str) -> str:
    out = subprocess.run(["docker", "compose", "ps", "-q", service], capture_output=True, text=True, check=True).stdout
    return out.strip()
