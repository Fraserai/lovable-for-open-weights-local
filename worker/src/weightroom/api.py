"""The web service: REST API, the remote MCP server at /mcp, the trained-model endpoint, accounts,
Stripe webhooks and the pages. Run it with:

    uvicorn weightroom.api:app --host 0.0.0.0 --port $PORT

Training happens in the separate worker process (python -m weightroom.worker), never in here.
"""

import contextlib
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from fastapi import Body, FastAPI, Header, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from mcp.server.transport_security import TransportSecuritySettings

from . import accounts, billing, config, data, db, guide, pages, security, service
from .errors import WRError
from .mcp_server import mcp
from .storage import store

COOKIE = "wr_session"


def _mcp_app():
    hosts = [urlparse(config.get().public_base_url).netloc, os.environ.get("RAILWAY_PUBLIC_DOMAIN", ""),
             *os.environ.get("EXTRA_HOSTS", "").split(",")]
    security_settings = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[h.strip() for h in hosts if h.strip()] + ["localhost:*", "127.0.0.1:*", "testserver"],
        allowed_origins=[config.get().public_base_url, "http://localhost:*", "http://127.0.0.1:*"])
    return mcp.streamable_http_app(streamable_http_path="/mcp", stateless_http=True, json_response=True,
                                   transport_security=security_settings)


def create_app() -> FastAPI:
    mcp_app = _mcp_app()

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        config.require_safe_production()
        db.init()
        async with mcp.session_manager.run():
            yield

    app = FastAPI(title="Weightroom", lifespan=lifespan, docs_url="/api-docs", redoc_url=None)

    @app.exception_handler(WRError)
    async def wr_error(_req: Request, e: WRError):
        return JSONResponse(e.to_dict(), status_code=e.status)

    # Pages ----------------------------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def landing():
        return pages.landing()

    @app.get("/signup", response_class=HTMLResponse, include_in_schema=False)
    def signup_page():
        return pages.signup()

    @app.get("/login", response_class=HTMLResponse, include_in_schema=False)
    def login_page():
        return pages.login()

    @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    def dashboard_page():
        return pages.dashboard()

    @app.get("/docs/agents", response_class=HTMLResponse, include_in_schema=False)
    def agent_guide():
        return pages.agent_guide()

    @app.get("/llms.txt", response_class=PlainTextResponse, include_in_schema=False)
    def llms_txt():
        return guide.text()

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        return {"ok": True}

    # The tools over REST (same JSON as MCP) -----------------------------------------------------------
    @app.post("/v1/uploads")
    def create_upload(body: dict = Body(...), authorization: str | None = Header(None)):
        return service.upload_data(authorization, str(body.get("filename") or ""))

    @app.put("/v1/uploads/{dataset_id}")
    async def receive_upload(dataset_id: str, token: str, request: Request):
        limit = config.get().max_upload_mb * 1024 * 1024
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "upload"
            size = 0
            with path.open("wb") as f:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > limit:
                        raise WRError("upload_too_large", f"The file is over {config.get().max_upload_mb} MB.",
                                      "Upload a smaller file (fewer rows), or ask about Enterprise for larger datasets.",
                                      status=413)
                    f.write(chunk)

            def save():
                with db.session() as s:
                    return data.save_upload(s, dataset_id, token, path, size)
            return await run_in_threadpool(save)

    @app.post("/v1/datasets/check")
    def check(body: dict = Body(...), authorization: str | None = Header(None)):
        allowed = {"dataset_id", "hf_dataset", "hf_config", "hf_split", "limit", "user_column", "assistant_column",
                   "instructions"}
        return service.check_data(authorization, **{k: v for k, v in body.items() if k in allowed})

    @app.post("/v1/quotes")
    def make_quote(body: dict = Body(...), authorization: str | None = Header(None)):
        return service.quote(authorization, str(body.get("dataset_id") or ""), body.get("base_model"), body.get("epochs"))

    @app.post("/v1/jobs")
    def start_job(body: dict = Body(...), authorization: str | None = Header(None)):
        return service.train(authorization, str(body.get("quote_id") or ""))

    @app.get("/v1/jobs")
    def jobs_list(limit: int = 20, authorization: str | None = Header(None)):
        return service.list_jobs(authorization, limit)

    @app.get("/v1/jobs/{job_id}")
    def job_status(job_id: str, authorization: str | None = Header(None)):
        return service.status(authorization, job_id)

    @app.post("/v1/jobs/{job_id}/deliver")
    def job_deliver(job_id: str, body: dict = Body(default={}), authorization: str | None = Header(None)):
        return service.deliver(authorization, job_id, bool(body.get("download", True)), body.get("hf_repo"),
                               bool(body.get("endpoint", True)))

    @app.get("/v1/models")
    def models(authorization: str | None = Header(None)):
        return service.list_models(authorization)

    @app.get("/v1/downloads/{token}", include_in_schema=False)
    def download(token: str):
        delivery_id = security.read_token("download", token)
        with db.session() as s:
            d = s.get(db.Delivery, delivery_id) if delivery_id else None
            if d is None or d.state != "ready" or d.kind != "download":
                raise WRError("download_link_invalid", "This download link is invalid or has expired.",
                              "Call deliver again for a fresh link.", status=404)
            key, job_id = d.result, d.job_id
        path = store().local_path(key)
        if path is None:
            return RedirectResponse(store().presigned_get(key, 3600, "adapter.tar.gz"))
        return FileResponse(path, filename=f"{job_id}-adapter.tar.gz", media_type="application/gzip")

    @app.post("/v1/chat/completions")
    def chat_completions(body: dict = Body(...), authorization: str | None = Header(None)):
        return service.chat_completion(authorization, body)

    # Accounts (the dashboard) -----------------------------------------------------------------------
    def _json_only(request: Request) -> None:
        # Login is a cookie; requiring a JSON body blocks cross-site form posts (no CORS preflight).
        if not request.headers.get("content-type", "").startswith("application/json"):
            raise WRError("bad_request", "Send JSON.", "", status=415)

    def _login_response(payload: dict, account: db.Account) -> JSONResponse:
        resp = JSONResponse(payload)
        resp.set_cookie(COOKIE, accounts.session_token(account), max_age=accounts.SESSION_TTL, httponly=True,
                        samesite="lax", secure=config.get().is_production)
        return resp

    @app.post("/account/signup", include_in_schema=False)
    def account_signup(request: Request, body: dict = Body(...)):
        _json_only(request)
        with db.session() as s:
            ip = request.client.host if request.client else None
            account, raw = accounts.signup(s, body.get("email"), body.get("password"), ip)
            return _login_response({"api_key": raw, "note": "Copy this key now; it won't be shown again."}, account)

    @app.post("/account/login", include_in_schema=False)
    def account_login(request: Request, body: dict = Body(...)):
        _json_only(request)
        with db.session() as s:
            account = accounts.login(s, body.get("email"), body.get("password"))
            return _login_response({"ok": True}, account)

    @app.post("/account/logout", include_in_schema=False)
    def account_logout():
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(COOKIE)
        return resp

    def _with_account(request: Request, fn):
        with db.session() as s:
            account = accounts.from_session(s, request.cookies.get(COOKIE))
            return fn(s, account)

    @app.get("/account/me", include_in_schema=False)
    def account_me(request: Request):
        return _with_account(request, accounts.overview)

    @app.post("/account/keys", include_in_schema=False)
    def account_new_key(request: Request, body: dict = Body(...)):
        _json_only(request)
        return _with_account(request, lambda s, a: accounts.new_key(s, a, body.get("name") or "key",
                                                                    body.get("monthly_cap"), body.get("per_job_cap")))

    @app.post("/account/keys/{key_id}/caps", include_in_schema=False)
    def account_caps(key_id: str, request: Request, body: dict = Body(...)):
        _json_only(request)
        _with_account(request, lambda s, a: accounts.set_caps(s, a, key_id, body.get("monthly_cap"),
                                                              body.get("per_job_cap")))
        return {"ok": True}

    @app.post("/account/keys/{key_id}/revoke", include_in_schema=False)
    def account_revoke(key_id: str, request: Request):
        _json_only(request)
        _with_account(request, lambda s, a: accounts.revoke(s, a, key_id))
        return {"ok": True}

    @app.post("/account/hf-token", include_in_schema=False)
    def account_hf(request: Request, body: dict = Body(...)):
        _json_only(request)
        _with_account(request, lambda s, a: accounts.save_hf_token(s, a, body.get("token")))
        return {"ok": True}

    @app.post("/account/checkout", include_in_schema=False)
    def account_checkout(request: Request, body: dict = Body(...)):
        _json_only(request)
        return {"url": _with_account(request, lambda s, a: billing.checkout(s, a, str(body.get("item"))))}

    @app.post("/account/portal", include_in_schema=False)
    def account_portal(request: Request):
        _json_only(request)
        return {"url": _with_account(request, billing.portal)}

    @app.post("/stripe/webhook", include_in_schema=False)
    async def stripe_webhook(request: Request, stripe_signature: str | None = Header(None)):
        payload = await request.body()

        def handle():
            with db.session() as s:
                return billing.handle_webhook(s, payload, stripe_signature)
        return {"result": await run_in_threadpool(handle)}

    # The MCP server takes everything else under /mcp
    app.mount("/", mcp_app)
    return app


app = create_app()
