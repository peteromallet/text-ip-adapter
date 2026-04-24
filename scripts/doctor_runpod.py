from __future__ import annotations

import json
import os
import socket
from pathlib import Path

from dotenv import load_dotenv


def _resolve(host: str) -> dict[str, str]:
    try:
        address = socket.getaddrinfo(host, 443)[0][4][0]
        return {"host": host, "status": "ok", "detail": address}
    except Exception as exc:
        return {"host": host, "status": "error", "detail": f"{type(exc).__name__}: {exc}"}


def _connect(host: str, port: int) -> dict[str, str]:
    sock = socket.socket()
    sock.settimeout(5)
    try:
        sock.connect((host, port))
        return {"target": f"{host}:{port}", "status": "ok", "detail": "connected"}
    except Exception as exc:
        return {"target": f"{host}:{port}", "status": "error", "detail": f"{type(exc).__name__}: {exc}"}
    finally:
        sock.close()


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    env_file = os.getenv("RUNPOD_LIFECYCLE_ENV") or str(repo_root.parent / "runpod-lifecycle" / ".env")
    load_dotenv(env_file)

    result = {
        "env_file": env_file,
        "has_runpod_api_key": bool(os.getenv("RUNPOD_API_KEY")),
        "storage_name": os.getenv("RUNPOD_STORAGE_NAME"),
        "proxy_env": {
            key: os.getenv(key)
            for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
            if os.getenv(key)
        },
        "dns": [
            _resolve("api.runpod.io"),
            _resolve("runpod.io"),
            _resolve("google.com"),
        ],
        "socket_checks": [
            _connect("1.1.1.1", 53),
            _connect("8.8.8.8", 53),
        ],
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
