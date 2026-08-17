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

from aiohttp import ClientError, ClientSession, ClientTimeout, WSMsgType, web


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


async def proxy_websocket(request: web.Request) -> web.WebSocketResponse:
    downstream = web.WebSocketResponse(heartbeat=30)
    await downstream.prepare(request)

    session: ClientSession = request.app["session"]
    try:
        async with session.ws_connect(
            upstream_url(request),
            headers=request.app["auth_headers"],
            heartbeat=30,
            max_msg_size=0,
        ) as upstream:

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
        if not downstream.closed:
            await downstream.close()

    return downstream


async def proxy_http(request: web.Request) -> web.Response:
    session: ClientSession = request.app["session"]
    headers = filtered_headers(request.headers)
    headers.update(request.app["auth_headers"])

    try:
        async with session.request(
            request.method,
            upstream_url(request),
            headers=headers,
            data=await request.read(),
            allow_redirects=False,
        ) as response:
            body = await response.read()
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
        # Relay compressed bytes and their Content-Encoding header unchanged.
        # The browser can decode Brotli itself; the local Python proxy should
        # not require the optional Brotli package just to forward RunPod HTML.
        auto_decompress=False,
    )


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
