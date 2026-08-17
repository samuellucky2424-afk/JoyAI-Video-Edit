import json
import os
import urllib.error
import urllib.request

import uvicorn
from fastapi import FastAPI, Response


app = FastAPI()


@app.get("/ping")
def ping():
    """RunPod-only readiness probe served on PORT_HEALTH.

    The public JoyAI server owns /health. A worker remains in RunPod's
    initializing state (204) until that endpoint confirms the runtime loaded.
    """
    model_port = os.getenv(
        "JOYOMNI_PORT",
        os.getenv("PORT", "8080"),
    )

    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{model_port}/health",
            timeout=2,
        ) as response:
            health = json.loads(response.read())

        if health.get("ok") is True and health.get("runtime_loaded") is True:
            return {
                "status": "healthy",
                "model": "ready",
            }

    except (
        urllib.error.URLError,
        TimeoutError,
        ValueError,
    ):
        pass

    # RunPod load-balancer health contract:
    # 200 = ready, 204 = still initializing, anything else = unhealthy.
    return Response(status_code=204)


if __name__ == "__main__":
    health_port = int(os.getenv("PORT_HEALTH", "8081"))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=health_port,
        log_level="info",
    )
