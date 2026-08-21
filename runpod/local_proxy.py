"""Local authenticated proxy for the JoyAI RunPod load-balancer endpoint.

Browsers cannot attach a secret RunPod bearer token to normal navigation or a
WebSocket constructor. This proxy keeps the token in the local process and
adds it to upstream HTTP and WebSocket requests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from aiohttp import (
    ClientError,
    ClientSession,
    ClientTimeout,
    TCPConnector,
    WSCloseCode,
    WSMsgType,
    web,
)


HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

RUNPOD_WORKER_HEADER = "X-Runpod-Worker-Id"
# RunPod's load balancer may not count an otherwise active WebSocket as worker
# activity.  Keep this below the platform's 5-second default idle timeout so a
# live stream cannot be scaled down underneath the browser.
RUNPOD_KEEPALIVE_SECONDS = 3.0


def proxy_exception_handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """Ignore the harmless Windows Proactor reset logged after a peer closes."""
    error = context.get("exception")
    message = context.get("message", "")
    if (
        os.name == "nt"
        and isinstance(error, ConnectionResetError)
        and getattr(error, "winerror", None) == 10054
        and "_call_connection_lost" in message
    ):
        return
    loop.default_exception_handler(context)


def filtered_headers(headers) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
        and name.lower() != "authorization"
    }


def upstream_url(request: web.Request) -> str:
    return f"{request.app['upstream']}{request.rel_url}"


def upstream_headers(
    app: web.Application,
    *,
    affinity: bool = True,
) -> dict[str, str]:
    """Authenticate requests and optionally prefer the selected worker."""
    headers = dict(app["auth_headers"])
    worker_id = app["proxy_state"].get("worker_id")
    if affinity and worker_id:
        # Soft affinity never waits behind a worker RunPod considers at
        # capacity. Max workers=1 remains the hard guarantee that a fallback
        # cannot create a second H200 during one-viewer testing.
        headers[RUNPOD_WORKER_HEADER] = worker_id
    return headers


def remember_worker(app: web.Application, headers) -> str | None:
    """Remember the worker chosen by RunPod for later HTTP and WS requests."""
    worker_id = str(headers.get(RUNPOD_WORKER_HEADER, "")).strip()
    if not worker_id:
        return None
    state = app["proxy_state"]
    if state.get("worker_id") != worker_id:
        state["worker_id"] = worker_id
        print(f"JoyAI pinned to RunPod worker {worker_id}.", flush=True)
    return worker_id


async def keep_worker_active(app: web.Application) -> None:
    """Reset RunPod's idle timer while the browser has a live WebSocket."""
    session: ClientSession = app["session"]
    failures = 0
    while True:
        await asyncio.sleep(RUNPOD_KEEPALIVE_SECONDS)
        try:
            async with session.get(
                f"{app['upstream']}/health",
                # RunPod can hold an affinity request while a long-lived WS is
                # occupying the selected worker. With max workers=1, normal
                # routing still reaches the only worker without that wait.
                headers=upstream_headers(app, affinity=False),
                timeout=ClientTimeout(total=10),
            ) as response:
                await response.read()
                remember_worker(app, response.headers)
                if response.status != 200:
                    raise ClientError(
                        f"RunPod keepalive returned HTTP {response.status}"
                    )
            failures = 0
        except asyncio.CancelledError:
            raise
        except (ClientError, asyncio.TimeoutError, OSError) as error:
            failures += 1
            if failures == 1 or failures % 5 == 0:
                print(
                    f"JoyAI RunPod keepalive failed ({failures}): {error}",
                    flush=True,
                )


async def proxy_websocket(request: web.Request) -> web.WebSocketResponse:
    downstream = web.WebSocketResponse(heartbeat=30)
    await downstream.prepare(request)

    session: ClientSession = request.app["session"]
    proxy_state = request.app["proxy_state"]
    upstream = None
    keepalive_task = None
    try:
        # Do not put strict/soft affinity on the WebSocket upgrade. RunPod can
        # hold an affinity upgrade outside the worker for several minutes even
        # when ordinary HTTP requests reach it immediately. Max workers=1 pins
        # normal routing to the only worker without blocking the handshake.
        upstream = await session.ws_connect(
            upstream_url(request),
            headers=upstream_headers(request.app, affinity=False),
            # The browser already sends an application-level JSON ping every
            # second. A protocol heartbeat adds a second liveness mechanism
            # that can falsely close with code 1006 when RunPod's load
            # balancer does not relay a Pong during graph/session rebuilds.
            heartbeat=None,
            max_msg_size=0,
        )
        response = getattr(upstream, "_response", None)
        if response is not None:
            remember_worker(request.app, response.headers)
        print("JoyAI WebSocket connected through RunPod.", flush=True)
        keepalive_task = asyncio.create_task(keep_worker_active(request.app))

        # The local proxy is intentionally single-viewer. Replacing a stale
        # browser socket explicitly tells the server to release its session
        # ticket, avoiding a long queue after a page refresh.
        async with proxy_state["websocket_lock"]:
            previous_upstream = proxy_state["active_upstream"]
            previous_downstream = proxy_state["active_downstream"]
            proxy_state["active_upstream"] = upstream
            proxy_state["active_downstream"] = downstream

        # Close the replaced pair outside the lock. The old handler also uses
        # this lock during cleanup, so awaiting its close while holding the
        # lock would deadlock a page refresh.
        if previous_upstream is not None and not previous_upstream.closed:
            try:
                await previous_upstream.send_json({"type": "stop"})
            except (ClientError, asyncio.TimeoutError, OSError):
                pass
            await previous_upstream.close(
                code=WSCloseCode.GOING_AWAY,
                message=b"replaced by a refreshed browser",
            )
        if previous_downstream is not None and not previous_downstream.closed:
            await previous_downstream.close(
                code=WSCloseCode.GOING_AWAY,
                message=b"replaced by a refreshed browser",
            )

        async def browser_to_runpod() -> None:
            async for message in downstream:
                if message.type == WSMsgType.TEXT:
                    await upstream.send_str(message.data)
                elif message.type == WSMsgType.BINARY:
                    await upstream.send_bytes(message.data)
                elif message.type == WSMsgType.ERROR:
                    break

        async def runpod_to_browser() -> None:
            async for message in upstream:
                if message.type == WSMsgType.TEXT:
                    await downstream.send_str(message.data)
                elif message.type == WSMsgType.BINARY:
                    await downstream.send_bytes(message.data)
                elif message.type == WSMsgType.ERROR:
                    break

        tasks = [
            asyncio.create_task(browser_to_runpod()),
            asyncio.create_task(runpod_to_browser()),
        ]
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*done, *pending, return_exceptions=True)

    except (ClientError, asyncio.TimeoutError, OSError) as error:
        if not downstream.closed:
            await downstream.send_json(
                {
                    "type": "proxy_error",
                    "error": f"RunPod connection failed: {error}",
                }
            )
    finally:
        if keepalive_task is not None:
            keepalive_task.cancel()
            await asyncio.gather(keepalive_task, return_exceptions=True)
        upstream_code = getattr(upstream, "close_code", None)
        downstream_code = getattr(downstream, "close_code", None)
        print(
            "JoyAI WebSocket closed "
            f"(browser={downstream_code}, RunPod={upstream_code}).",
            flush=True,
        )
        if upstream is not None and not upstream.closed:
            try:
                await upstream.send_json({"type": "stop"})
            except (ClientError, asyncio.TimeoutError, OSError):
                pass
            await upstream.close()
        async with proxy_state["websocket_lock"]:
            if proxy_state["active_upstream"] is upstream:
                proxy_state["active_upstream"] = None
            if proxy_state["active_downstream"] is downstream:
                proxy_state["active_downstream"] = None
        if not downstream.closed:
            await downstream.close()

    return downstream


async def proxy_http(request: web.Request) -> web.Response:
    session: ClientSession = request.app["session"]
    headers = filtered_headers(request.headers)
    # Windows PowerShell 5.1 cannot reliably decode Brotli responses. Ask the
    # upstream load balancer for an identity response so both PowerShell's
    # readiness probe and the browser receive bytes they can consume directly.
    headers["Accept-Encoding"] = "identity"
    headers.update(upstream_headers(request.app))

    try:
        async with session.request(
            request.method,
            upstream_url(request),
            headers=headers,
            data=await request.read(),
            allow_redirects=False,
        ) as response:
            body = await response.read()
            remember_worker(request.app, response.headers)
            return web.Response(
                body=body,
                status=response.status,
                reason=response.reason,
                headers=filtered_headers(response.headers),
            )
    except (ClientError, asyncio.TimeoutError, OSError) as error:
        return web.Response(
            status=503,
            content_type="application/json",
            text=json.dumps(
                {
                    "status": 503,
                    "title": "RunPod connection unavailable",
                    "detail": str(error),
                }
            ),
        )


async def route_request(request: web.Request):
    if request.path == "/ws":
        return await proxy_websocket(request)
    return await proxy_http(request)


async def create_session(app: web.Application) -> None:
    if os.name == "nt":
        asyncio.get_running_loop().set_exception_handler(proxy_exception_handler)
    app["session"] = ClientSession(
        timeout=ClientTimeout(total=None, connect=60, sock_connect=60),
        # Windows DNS briefly failed during a live test after the first
        # minute. Reuse the resolved RunPod address instead of asking the OS
        # resolver again for every short-lived keepalive connection.
        connector=TCPConnector(
            use_dns_cache=True,
            ttl_dns_cache=600,
            keepalive_timeout=30,
        ),
        # Relay compressed bytes and their Content-Encoding header unchanged.
        # The browser can decode Brotli itself; the local Python proxy should
        # not require the optional Brotli package just to forward RunPod HTML.
        auto_decompress=False,
    )
    # aiohttp warns when top-level application state is mutated after startup.
    # Keep live socket state inside one mutable object registered at startup.
    app["proxy_state"] = {
        "websocket_lock": asyncio.Lock(),
        "active_upstream": None,
        "active_downstream": None,
        "worker_id": None,
    }


async def close_session(app: web.Application) -> None:
    await app["session"].close()


def create_app(endpoint_id: str, api_key: str) -> web.Application:
    app = web.Application(client_max_size=1024**3)
    app["upstream"] = f"https://{endpoint_id}.api.runpod.ai"
    app["auth_headers"] = {"Authorization": f"Bearer {api_key}"}
    app.on_startup.append(create_session)
    app.on_cleanup.append(close_session)
    app.router.add_route("*", "/{tail:.*}", route_request)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=9000, type=int)
    args = parser.parse_args()

    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY is required")

    web.run_app(
        create_app(args.endpoint_id, api_key),
        host=args.host,
        port=args.port,
        print=lambda message: print(message, flush=True),
    )


if __name__ == "__main__":
    main()
