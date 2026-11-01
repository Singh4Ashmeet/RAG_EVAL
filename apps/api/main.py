"""Same-origin RAG_EVAL hosted API and static frontend server."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import dotenv
import httpx
from cryptography.fernet import Fernet, InvalidToken
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import DateTime, ForeignKey, String, Text, create_engine, delete, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, synonym

from core.generate.pipeline import NO_CONTEXT_ANSWER
from eval.datasets.schema import EvaluationCase, EvaluationDataset

dotenv.load_dotenv()


class Base(DeclarativeBase):
    pass


class Account(Base):
    __tablename__ = "accounts"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # Keep the legacy column name so existing hosted SQLite/DB deployments can
    # adopt the new identity providers without losing existing accounts.
    auth_subject: Mapped[str] = mapped_column("github_id", String(32), unique=True)
    github_id = synonym("auth_subject")
    login: Mapped[str] = mapped_column(String(128))
    email: Mapped[str] = mapped_column(String(320), unique=True)


class Invitation(Base):
    __tablename__ = "invitations"
    email: Mapped[str] = mapped_column(String(320), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class SessionRecord(Base):
    __tablename__ = "sessions"
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id", ondelete="CASCADE"))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class EmailLoginToken(Base):
    __tablename__ = "email_login_tokens"
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    storage_key: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(24), default="queued")
    sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class CorpusElement(Base):
    __tablename__ = "corpus_elements"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int]
    text: Mapped[str] = mapped_column(Text)
    embedding_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class DatasetRevision(Base):
    __tablename__ = "dataset_revisions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), index=True
    )
    revision: Mapped[int]
    sha256: Mapped[str] = mapped_column(String(64))
    cases_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(24), default="queued")
    detail: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), index=True
    )
    dataset_id: Mapped[str] = mapped_column(String(36), index=True)
    revision: Mapped[int]
    status: Mapped[str] = mapped_column(String(24), default="queued")
    report_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), index=True
    )
    dataset_id: Mapped[str] = mapped_column(String(36), index=True)
    revision: Mapped[int]
    status: Mapped[str] = mapped_column(String(24), default="queued")
    report_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class ProviderCredential(Base):
    __tablename__ = "provider_credentials"
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), primary_key=True
    )
    provider: Mapped[str] = mapped_column(String(16), primary_key=True)
    encrypted_key: Mapped[str] = mapped_column(Text)


class AccountResponse(BaseModel):
    id: str
    login: str


class DocumentResponse(BaseModel):
    id: str
    name: str
    status: str
    sha256: str
    created_at: datetime


class UploadResponse(DocumentResponse):
    element_count: int
    job_id: str


class ElementResponse(BaseModel):
    id: str
    ordinal: int
    text: str


class DatasetSummary(BaseModel):
    id: str
    name: str
    revision: int
    sha256: str | None
    case_count: int


class AskSource(BaseModel):
    element_id: str
    document: str
    text: str
    cosine_similarity: float


class AskResponse(BaseModel):
    answer: str
    sources: list[AskSource]
    citations: list[dict[str, str]]
    unresolved_citation_labels: list[str]
    provider: str
    model: str


DB_URL = os.getenv("DATABASE_URL", "sqlite:///./rag_eval_hosted.db")
if DB_URL.startswith("postgres://"):
    DB_URL = "postgresql+psycopg://" + DB_URL.removeprefix("postgres://")
elif DB_URL.startswith("postgresql://"):
    DB_URL = "postgresql+psycopg://" + DB_URL.removeprefix("postgresql://")
engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(engine, expire_on_commit=False)
if os.getenv("SKIP_SCHEMA_INIT") != "true":
    Base.metadata.create_all(engine)
if engine.dialect.name == "postgresql" and os.getenv("SKIP_RLS_SETUP") != "true":
    with engine.begin() as conn:
        for table in (
            "documents",
            "corpus_elements",
            "datasets",
            "dataset_revisions",
            "jobs",
            "evaluation_runs",
            "experiments",
            "provider_credentials",
        ):
            conn.execute(text(f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY'))
            conn.execute(text(f'ALTER TABLE "{table}" FORCE ROW LEVEL SECURITY'))
            conn.execute(text(f'DROP POLICY IF EXISTS workspace_isolation ON "{table}"'))
            conn.execute(
                text(f'''CREATE POLICY workspace_isolation ON "{table}"
                USING (workspace_id = current_setting('app.workspace_id', true))
                WITH CHECK (workspace_id = current_setting('app.workspace_id', true))''')
            )
app = FastAPI(title="RAG_EVAL API", version="1.0.0")
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
PRIVATE_DATA = Path(os.getenv("PRIVATE_DATA_DIR", "./private_data")).resolve()
PRIVATE_DATA.mkdir(parents=True, exist_ok=True)


@app.middleware("http")
async def same_origin_mutations(request: Request, call_next):
    if (
        request.method in {"POST", "PUT", "PATCH", "DELETE"}
        and request.url.path != "/api/v1/auth/callback"
    ):
        origin = request.headers.get("origin")
        if origin:
            from urllib.parse import urlsplit

            if urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "Cross-origin mutation rejected."}, status_code=403)
        elif request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "Cross-origin mutation rejected."}, status_code=403)
    return await call_next(request)


def db_session():
    with SessionLocal() as db:
        yield db


def current_account(request: Request, db=Depends(db_session)) -> Account:
    token = request.cookies.get("rag_eval_session")
    if not token:
        raise HTTPException(401, "Sign in with an invited GitHub account.")
    record = db.get(SessionRecord, hashlib.sha256(token.encode()).hexdigest())
    expiry = record.expires_at if record else None
    if expiry is not None and expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=UTC)
    if not record or expiry < datetime.now(UTC):
        raise HTTPException(401, "Session expired. Sign in again.")
    account = db.get(Account, record.account_id)
    if account is None:
        raise HTTPException(401, "Account is unavailable.")
    if db.bind and db.bind.dialect.name == "postgresql":
        db.execute(
            text("SELECT set_config('app.workspace_id', :workspace_id, true)"),
            {"workspace_id": account.id},
        )
    return account


def owned(db, model, account_id: str, item_id: str):
    item = db.scalar(select(model).where(model.id == item_id, model.workspace_id == account_id))
    if item is None:
        raise HTTPException(404, "Resource not found.")
    return item


def validate_dataset_cases(
    db, account_id: str, cases: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    try:
        parsed = [EvaluationCase.model_validate(case) for case in cases]
        EvaluationDataset(dataset_id="validation", sha256="0" * 64, cases=parsed)
    except ValidationError as exc:
        raise HTTPException(422, detail=exc.errors(include_url=False)) from exc
    element_ids = {element_id for case in parsed for element_id in case.relevant_element_ids}
    if element_ids:
        existing = set(
            db.scalars(
                select(CorpusElement.id).where(
                    CorpusElement.workspace_id == account_id,
                    CorpusElement.id.in_(element_ids),
                )
            )
        )
        if existing != element_ids:
            raise HTTPException(422, "A case references a foreign or missing source element.")
    return [case.model_dump(mode="json") for case in parsed]


def store_original(key: str, raw: bytes, content_type: str) -> None:
    bucket = os.getenv("S3_BUCKET")
    if bucket:
        import boto3
        from botocore.config import Config

        endpoint = os.getenv("S3_ENDPOINT_URL")
        options = {"endpoint_url": endpoint, "region_name": os.getenv("AWS_REGION", "us-east-1")}
        if endpoint:
            options["config"] = Config(s3={"addressing_style": "path"})
        client = boto3.client("s3", **options)
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=raw,
            ContentType=content_type,
            ServerSideEncryption="AES256",
        )
        return
    path = PRIVATE_DATA / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def delete_original(key: str) -> None:
    bucket = os.getenv("S3_BUCKET")
    if bucket:
        import boto3
        from botocore.config import Config

        endpoint = os.getenv("S3_ENDPOINT_URL")
        options = {"endpoint_url": endpoint, "region_name": os.getenv("AWS_REGION", "us-east-1")}
        if endpoint:
            options["config"] = Config(s3={"addressing_style": "path"})
        boto3.client("s3", **options).delete_object(Bucket=bucket, Key=key)
        return
    path = (PRIVATE_DATA / key).resolve()
    if PRIVATE_DATA in path.parents and path.is_file():
        path.unlink()


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "rag_eval"}


@app.get("/terms.html", include_in_schema=False)
def hosted_terms():
    return FileResponse(
        Path(__file__).resolve().parents[2] / "docs" / "hosted-terms.md", media_type="text/markdown"
    )


@app.get("/privacy.html", include_in_schema=False)
def hosted_privacy():
    return FileResponse(
        Path(__file__).resolve().parents[2] / "docs" / "hosted-privacy.md",
        media_type="text/markdown",
    )


def resolve_auth_account(db, subject: str, email: str, login: str) -> Account:
    account = db.scalar(select(Account).where(Account.auth_subject == subject))
    if account is None:
        account = db.scalar(select(Account).where(Account.email == email))
    if account is None:
        account = Account(
            id=str(uuid.uuid4()), auth_subject=subject, login=login[:128], email=email
        )
        db.add(account)
        db.flush()
    else:
        account.login = login[:128]
        account.email = email
        # Promote passwordless-email identities to a stable provider subject
        # when the user later chooses Google or GitHub.
        if account.auth_subject.startswith("e:"):
            account.auth_subject = subject
    return account


def create_session_response(db, account: Account, response: RedirectResponse | None = None):
    session_token = secrets.token_urlsafe(48)
    db.add(
        SessionRecord(
            token_hash=hashlib.sha256(session_token.encode()).hexdigest(),
            account_id=account.id,
            expires_at=datetime.now(UTC) + timedelta(days=14),
        )
    )
    db.commit()
    response = response or RedirectResponse("/")
    response.set_cookie(
        "rag_eval_session",
        session_token,
        httponly=True,
        secure=os.getenv("COOKIE_SECURE", "true").lower() == "true",
        samesite="lax",
        max_age=14 * 86400,
    )
    return response


def public_base_url(request: Request | None = None) -> str:
    configured = os.getenv("PUBLIC_BASE_URL") or os.getenv("RENDER_EXTERNAL_URL")
    if configured:
        return configured.rstrip("/")
    return str(request.base_url).rstrip("/") if request else ""


def oauth_redirect(provider: str, authorize_url: str, client_id: str, scope: str, request: Request):
    state = secrets.token_urlsafe(32)
    callback_path = (
        "/api/v1/auth/callback" if provider == "github" else f"/api/v1/auth/{provider}/callback"
    )
    query = httpx.QueryParams(
        {
            "client_id": client_id,
            "redirect_uri": f"{public_base_url(request)}{callback_path}",
            "response_type": "code",
            "scope": scope,
            "state": state,
        }
    )
    response = RedirectResponse(f"{authorize_url}?{query}")
    response.set_cookie(
        f"oauth_state_{provider}",
        state,
        httponly=True,
        secure=os.getenv("COOKIE_SECURE", "true").lower() == "true",
        samesite="lax",
        max_age=600,
    )
    return response


@app.get("/api/v1/auth/github")
def github_login(request: Request):
    client_id = os.getenv("GITHUB_CLIENT_ID")
    if not client_id or not os.getenv("GITHUB_CLIENT_SECRET"):
        raise HTTPException(503, "GitHub sign-in is not configured.")
    return oauth_redirect(
        "github",
        "https://github.com/login/oauth/authorize",
        client_id,
        "read:user user:email",
        request,
    )


@app.get("/api/v1/auth/callback")
async def github_callback(code: str, state: str, request: Request, db=Depends(db_session)):
    if not secrets.compare_digest(state, request.cookies.get("oauth_state_github", "")):
        raise HTTPException(400, "Invalid sign-in state.")
    async with httpx.AsyncClient(timeout=15) as client:
        token_response = await client.post(
            "https://github.com/login/oauth/access_token",
            json={
                "client_id": os.getenv("GITHUB_CLIENT_ID"),
                "client_secret": os.getenv("GITHUB_CLIENT_SECRET"),
                "code": code,
            },
            headers={"Accept": "application/json"},
        )
        token_response.raise_for_status()
        access = token_response.json().get("access_token")
        if not access:
            raise HTTPException(401, "GitHub sign-in failed.")
        headers = {"Authorization": f"Bearer {access}", "Accept": "application/vnd.github+json"}
        profile = (await client.get("https://api.github.com/user", headers=headers)).json()
        emails = (await client.get("https://api.github.com/user/emails", headers=headers)).json()
    email = next(
        (item["email"].lower() for item in emails if item.get("primary") and item.get("verified")),
        None,
    )
    if not email:
        raise HTTPException(403, "Add a verified primary email to your GitHub account to continue.")
    account = resolve_auth_account(db, str(profile["id"]), email, profile["login"])
    response = create_session_response(db, account)
    response.delete_cookie("oauth_state_github")
    return response


@app.get("/api/v1/auth/google")
def google_login(request: Request):
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    if not client_id:
        raise HTTPException(503, "Google sign-in is not configured yet.")
    return oauth_redirect(
        "google",
        "https://accounts.google.com/o/oauth2/v2/auth",
        client_id,
        "openid email profile",
        request,
    )


@app.get("/api/v1/auth/google/callback")
async def google_callback(code: str, state: str, request: Request, db=Depends(db_session)):
    if not secrets.compare_digest(state, request.cookies.get("oauth_state_google", "")):
        raise HTTPException(400, "Invalid sign-in state.")
    client_id, client_secret = os.getenv("GOOGLE_CLIENT_ID"), os.getenv("GOOGLE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise HTTPException(503, "Google sign-in is not configured yet.")
    redirect_uri = f"{public_base_url(request)}/api/v1/auth/google/callback"
    async with httpx.AsyncClient(timeout=15) as client:
        token_response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
            },
        )
        token_response.raise_for_status()
        access = token_response.json().get("access_token")
        if not access:
            raise HTTPException(401, "Google sign-in failed.")
        profile_response = await client.get(
            "https://openidconnect.googleapis.com/v1/userinfo",
            headers={"Authorization": f"Bearer {access}"},
        )
        profile_response.raise_for_status()
        profile = profile_response.json()
    if not profile.get("email_verified") or not profile.get("email") or not profile.get("sub"):
        raise HTTPException(403, "Google did not return a verified email for this account.")
    email = profile["email"].lower()
    account = resolve_auth_account(
        db,
        f"g:{profile['sub']}",
        email,
        profile.get("name") or email.split("@", 1)[0],
    )
    response = create_session_response(db, account)
    response.delete_cookie("oauth_state_google")
    return response


class EmailLinkRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        email = value.strip().lower()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
            raise ValueError("Enter a valid email address.")
        return email


@app.post("/api/v1/auth/email/request")
async def request_email_login(payload: EmailLinkRequest, request: Request, db=Depends(db_session)):
    api_key, sender = os.getenv("RESEND_API_KEY"), os.getenv("AUTH_EMAIL_FROM")
    dev_mode = os.getenv("AUTH_DEV_MODE", "false").lower() == "true" or api_key in (
        "console",
        "mock",
        "dev",
        "local",
    )
    if not dev_mode and (not api_key or not sender):
        raise HTTPException(503, "Email sign-in is not configured yet.")
    now = datetime.now(UTC)
    recent = db.scalar(
        select(EmailLoginToken).where(
            EmailLoginToken.email == payload.email,
            EmailLoginToken.created_at >= now - timedelta(seconds=60),
        )
    )
    if recent:
        return {"detail": "If that address can receive sign-in links, one will arrive shortly."}
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    db.add(
        EmailLoginToken(
            token_hash=token_hash,
            email=payload.email,
            created_at=now,
            expires_at=now + timedelta(minutes=15),
        )
    )
    db.commit()
    link = f"{public_base_url(request)}/api/v1/auth/email/consume?token={raw_token}"
    if dev_mode:
        print("\n==================================================", flush=True)
        print(f"[RAG_EVAL AUTH] Local sign-in link for {payload.email}:", flush=True)
        print(f"{link}", flush=True)
        print("==================================================\n", flush=True)
        return {
            "detail": "If that address can receive sign-in links, one will arrive shortly.",
            "dev_link": link,
        }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            sent = await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "from": sender,
                    "to": [payload.email],
                    "subject": "Your RAG_EVAL sign-in link",
                    "text": (
                        "Use this one-time link to sign in to RAG_EVAL. "
                        "It expires in 15 minutes.\n\n"
                        f"{link}\n\n"
                        "If you did not request this link, you can ignore this message."
                    ),
                },
            )
            sent.raise_for_status()
    except httpx.HTTPError as exc:
        token_record = db.get(EmailLoginToken, token_hash)
        if token_record:
            db.delete(token_record)
        db.commit()
        raise HTTPException(502, "We could not send the sign-in email. Please try again.") from exc
    return {"detail": "If that address can receive sign-in links, one will arrive shortly."}


@app.get("/api/v1/auth/email/consume")
def consume_email_login(token: str, db=Depends(db_session)):
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    record = db.get(EmailLoginToken, token_hash)
    expiry = record.expires_at if record else None
    if expiry is not None and expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=UTC)
    if not record or expiry < datetime.now(UTC):
        raise HTTPException(400, "This sign-in link is invalid or has expired. Request a new one.")
    email = record.email
    db.delete(record)
    email_subject = "e:" + hashlib.sha256(email.encode()).hexdigest()[:24]
    account = resolve_auth_account(db, email_subject, email, email.split("@", 1)[0])
    response = create_session_response(db, account)
    response.delete_cookie("oauth_state_google")
    return response


@app.post("/api/v1/auth/logout")
def logout(request: Request, db=Depends(db_session)):
    token = request.cookies.get("rag_eval_session")
    if token:
        record = db.get(SessionRecord, hashlib.sha256(token.encode()).hexdigest())
        if record:
            db.delete(record)
            db.commit()
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie("rag_eval_session")
    return response


@app.get("/api/v1/auth/me", response_model=AccountResponse)
def me(account=Depends(current_account)):
    return {"id": account.id, "login": account.login}


@app.get("/api/v1/auth/methods")
def auth_methods():
    dev_mode = os.getenv("AUTH_DEV_MODE", "false").lower() == "true" or os.getenv(
        "RESEND_API_KEY"
    ) in ("console", "mock", "dev", "local")
    has_resend = bool(os.getenv("RESEND_API_KEY") and os.getenv("AUTH_EMAIL_FROM"))
    return {
        "email": dev_mode or has_resend,
        "google": bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET")),
        "github": bool(os.getenv("GITHUB_CLIENT_ID") and os.getenv("GITHUB_CLIENT_SECRET")),
    }


@app.get("/api/v1/documents", response_model=list[DocumentResponse])
def documents(account=Depends(current_account), db=Depends(db_session)):
    return [
        {
            "id": d.id,
            "name": d.name,
            "status": d.status,
            "sha256": d.sha256,
            "created_at": d.created_at,
        }
        for d in db.scalars(
            select(Document)
            .where(Document.workspace_id == account.id)
            .order_by(Document.created_at.desc())
        )
    ]


@app.post("/api/v1/documents", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...), account=Depends(current_account), db=Depends(db_session)
):
    name = Path((file.filename or "").replace("\\", "/")).name
    if Path(name).suffix.lower() not in {".txt", ".md"}:
        raise HTTPException(415, "Upload a UTF-8 .txt or .md file.")
    raw = await file.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, f"File exceeds the {MAX_UPLOAD // (1024 * 1024)} MiB limit.")
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(422, "File must contain valid UTF-8 text.") from exc
    doc_id = str(uuid.uuid4())
    digest = hashlib.sha256(raw).hexdigest()
    key = f"{account.id}/documents/{doc_id}/{name}"
    store_original(key, raw, file.content_type or "text/plain; charset=utf-8")
    doc = Document(
        id=doc_id,
        workspace_id=account.id,
        name=name,
        storage_key=key,
        status="queued",
        sha256=digest,
    )
    db.add(doc)
    elements = [p.strip() for p in raw.decode("utf-8").split("\n\n") if p.strip()]
    rows = [
        CorpusElement(
            id=str(uuid.uuid4()), workspace_id=account.id, document_id=doc_id, ordinal=i, text=text
        )
        for i, text in enumerate(elements)
    ]
    db.add_all(rows)
    # The API records source elements immediately; the worker performs embeddings asynchronously.
    job = Job(
        id=str(uuid.uuid4()), workspace_id=account.id, kind="index", status="queued", detail=doc_id
    )
    db.add(job)
    db.commit()
    try:
        from redis import Redis
        from rq import Queue

        Queue(
            "rag_eval",
            connection=Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0")),
        ).enqueue("apps.worker.tasks.index_document", account.id, doc_id, job.id)
    except Exception:
        # Preserve an explicit failure state if the queue service is unreachable.
        if db.bind and db.bind.dialect.name == "postgresql":
            db.execute(
                text("SELECT set_config('app.workspace_id', :wid, true)"), {"wid": account.id}
            )
        doc.status = "failed"
        job.status = "failed"
        job.detail = "Queue unavailable"
        db.commit()
    return {
        "id": doc.id,
        "name": doc.name,
        "status": doc.status,
        "sha256": digest,
        "created_at": doc.created_at,
        "element_count": len(elements),
        "job_id": job.id,
    }


@app.get("/api/v1/documents/{doc_id}/elements", response_model=list[ElementResponse])
def document_elements(doc_id: str, account=Depends(current_account), db=Depends(db_session)):
    doc = owned(db, Document, account.id, doc_id)
    return [
        {"id": e.id, "ordinal": e.ordinal, "text": e.text}
        for e in db.scalars(
            select(CorpusElement)
            .where(CorpusElement.workspace_id == account.id, CorpusElement.document_id == doc.id)
            .order_by(CorpusElement.ordinal)
        )
    ]


@app.delete("/api/v1/documents/{doc_id}")
def delete_document(doc_id: str, account=Depends(current_account), db=Depends(db_session)):
    doc = owned(db, Document, account.id, doc_id)
    delete_original(doc.storage_key)
    db.execute(
        delete(CorpusElement).where(
            CorpusElement.workspace_id == account.id,
            CorpusElement.document_id == doc.id,
        )
    )
    db.delete(doc)
    db.commit()
    return {"deleted": True, "historical_reports_retain_excerpts": True}


@app.post("/api/v1/documents/{doc_id}/reindex")
def reindex_document(doc_id: str, account=Depends(current_account), db=Depends(db_session)):
    doc = owned(db, Document, account.id, doc_id)
    job = Job(
        id=str(uuid.uuid4()),
        workspace_id=account.id,
        kind="index",
        status="queued",
        detail=doc.id,
    )
    doc.status = "queued"
    db.add(job)
    db.commit()
    try:
        from redis import Redis
        from rq import Queue

        Queue(
            "rag_eval",
            connection=Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0")),
        ).enqueue("apps.worker.tasks.index_document", account.id, doc.id, job.id)
    except Exception as exc:
        if db.bind and db.bind.dialect.name == "postgresql":
            db.execute(
                text("SELECT set_config('app.workspace_id', :wid, true)"), {"wid": account.id}
            )
        doc.status = "failed"
        job.status = "failed"
        job.detail = "Queue unavailable"
        db.commit()
        raise HTTPException(503, "Indexing queue is unavailable.") from exc
    return {"job_id": job.id, "status": "queued"}


@app.delete("/api/v1/workspace")
def delete_workspace(account=Depends(current_account), db=Depends(db_session)):
    originals = list(db.scalars(select(Document).where(Document.workspace_id == account.id)))
    for document in originals:
        try:
            delete_original(document.storage_key)
        except Exception as exc:
            raise HTTPException(
                502, "Workspace deletion could not remove a stored source file."
            ) from exc
    for model in (
        CorpusElement,
        EvaluationRun,
        Experiment,
        Job,
        DatasetRevision,
        Dataset,
        ProviderCredential,
    ):
        db.execute(delete(model).where(model.workspace_id == account.id))
    db.execute(delete(Document).where(Document.workspace_id == account.id))
    db.execute(delete(SessionRecord).where(SessionRecord.account_id == account.id))
    db.delete(account)
    db.commit()
    response = JSONResponse({"deleted": True, "backup_expiry_days": 30})
    response.delete_cookie("rag_eval_session")
    return response


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    cases: list[dict[str, Any]] = Field(min_length=1)


class RunPayload(BaseModel):
    dataset_id: str
    revision: int = Field(ge=1)
    cutoffs: list[int] = Field(default_factory=lambda: [1, 3, 5], min_length=1, max_length=10)

    @field_validator("cutoffs")
    @classmethod
    def validate_cutoffs(cls, values: list[int]) -> list[int]:
        if any(value < 1 for value in values) or len(values) != len(set(values)):
            raise ValueError("Cutoffs must be positive unique integers.")
        return values


class ExperimentVariantPayload(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    retrieval_depth: int = Field(ge=1, le=20)


class ExperimentPayload(BaseModel):
    dataset_id: str
    revision: int = Field(ge=1)
    baseline_variant: str = Field(min_length=1)
    variants: list[ExperimentVariantPayload] = Field(min_length=2, max_length=6)
    cutoffs: list[int] = Field(default_factory=lambda: [1, 3, 5], min_length=1, max_length=10)
    bootstrap_resamples: int = Field(default=1000, ge=100, le=10000)
    random_seed: int = 0

    @field_validator("cutoffs")
    @classmethod
    def validate_experiment_cutoffs(cls, values: list[int]) -> list[int]:
        if any(value < 1 for value in values) or len(values) != len(set(values)):
            raise ValueError("Cutoffs must be positive unique integers.")
        return values

    @field_validator("variants")
    @classmethod
    def validate_variant_names(cls, variants: list[ExperimentVariantPayload]):
        if len({variant.name for variant in variants}) != len(variants):
            raise ValueError("Experiment variant names must be unique.")
        return variants


class AskPayload(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=5, ge=1, le=20)
    provider: Literal["ollama", "gemini", "groq"] = "ollama"

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Query cannot be whitespace-only.")
        return cleaned


@app.post("/api/v1/ask", response_model=AskResponse)
async def ask(payload: AskPayload, account=Depends(current_account), db=Depends(db_session)):
    elements = list(
        db.scalars(
            select(CorpusElement).where(
                CorpusElement.workspace_id == account.id, CorpusElement.embedding_json.is_not(None)
            )
        )
    )
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    chat_model = {
        "ollama": os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b"),
        "gemini": os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash"),
        "groq": os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b"),
    }[payload.provider]
    sources: list[dict[str, Any]] = []
    answer = NO_CONTEXT_ANSWER
    if elements:
        try:
            async with httpx.AsyncClient(timeout=90) as client:
                qresp = await client.post(
                    f"{base}/api/embed", json={"model": embed_model, "input": payload.query}
                )
                qresp.raise_for_status()
                query_vector = qresp.json()["embeddings"][0]
                ranked = []
                for element in elements:
                    vector = json.loads(element.embedding_json or "[]")
                    if len(vector) != len(query_vector) or not vector:
                        continue
                    denom = math.sqrt(sum(v * v for v in vector) * sum(v * v for v in query_vector))
                    score = (
                        sum(a * b for a, b in zip(vector, query_vector, strict=True)) / denom
                        if denom
                        else 0.0
                    )
                    ranked.append((score, element))
                ranked.sort(key=lambda pair: pair[0], reverse=True)
                sources = [
                    {
                        "element_id": element.id,
                        "document": db.get(Document, element.document_id).name,
                        "text": element.text,
                        "cosine_similarity": score,
                    }
                    for score, element in ranked[: payload.top_k]
                ]
                if sources:
                    prompt = (
                        "Answer only from the supplied source excerpts. If unsupported, say so. "
                        "Cite excerpts as [S1], [S2], etc.\n\n"
                        + "\n\n".join(
                            f"[S{index + 1}] {source['text']}"
                            for index, source in enumerate(sources)
                        )
                        + f"\n\nQuestion: {payload.query}"
                    )
                    if payload.provider == "ollama":
                        response = await client.post(
                            f"{base}/api/chat",
                            json={
                                "model": chat_model,
                                "stream": False,
                                "messages": [{"role": "user", "content": prompt}],
                            },
                        )
                        response.raise_for_status()
                        answer = response.json().get("message", {}).get("content", "")
                    else:
                        credential = db.get(ProviderCredential, (account.id, payload.provider))
                        encryption_key = os.getenv("CREDENTIAL_ENCRYPTION_KEY")
                        if credential is None or not encryption_key:
                            raise HTTPException(
                                409,
                                f"Save a {payload.provider.title()} chat key in Settings first.",
                            )
                        api_key = (
                            Fernet(encryption_key.encode())
                            .decrypt(credential.encrypted_key.encode())
                            .decode()
                        )
                        if payload.provider == "gemini":
                            response = await client.post(
                                f"https://generativelanguage.googleapis.com/v1beta/models/{chat_model}:generateContent",
                                headers={"x-goog-api-key": api_key},
                                json={"contents": [{"parts": [{"text": prompt}]}]},
                            )
                            response.raise_for_status()
                            answer = "".join(
                                part.get("text", "")
                                for part in response.json()
                                .get("candidates", [{}])[0]
                                .get("content", {})
                                .get("parts", [])
                            )
                        else:
                            response = await client.post(
                                "https://api.groq.com/openai/v1/chat/completions",
                                headers={"Authorization": f"Bearer {api_key}"},
                                json={
                                    "model": chat_model,
                                    "messages": [{"role": "user", "content": prompt}],
                                },
                            )
                            response.raise_for_status()
                            answer = (
                                response.json()
                                .get("choices", [{}])[0]
                                .get("message", {})
                                .get("content", "")
                            )
        except (httpx.HTTPError, InvalidToken, ValueError) as exc:
            raise HTTPException(
                503, "The selected model provider is unavailable or rejected the request."
            ) from exc
    cited_indexes = {int(index) for index in re.findall(r"\[S(\d+)\]", answer)}
    citations = [
        {"label": f"S{index}", "element_id": sources[index - 1]["element_id"]}
        for index in sorted(cited_indexes)
        if 1 <= index <= len(sources)
    ]
    unresolved = [
        f"S{index}" for index in sorted(cited_indexes) if index < 1 or index > len(sources)
    ]
    return {
        "answer": answer,
        "sources": sources,
        "citations": citations,
        "unresolved_citation_labels": unresolved,
        "provider": payload.provider.title(),
        "model": chat_model,
        "citation_labels": [f"S{i + 1}" for i in range(len(sources))],
    }


@app.get("/api/v1/datasets", response_model=list[DatasetSummary])
def list_datasets(account=Depends(current_account), db=Depends(db_session)):
    out = []
    for dataset in db.scalars(select(Dataset).where(Dataset.workspace_id == account.id)):
        rev = db.scalar(
            select(DatasetRevision)
            .where(
                DatasetRevision.workspace_id == account.id, DatasetRevision.dataset_id == dataset.id
            )
            .order_by(DatasetRevision.revision.desc())
        )
        out.append(
            {
                "id": dataset.id,
                "name": dataset.name,
                "revision": rev.revision if rev else 0,
                "sha256": rev.sha256 if rev else None,
                "case_count": len(json.loads(rev.cases_json)) if rev else 0,
            }
        )
    return out


@app.post("/api/v1/datasets", response_model=DatasetSummary)
def create_dataset(
    payload: DatasetCreate, account=Depends(current_account), db=Depends(db_session)
):
    cases = validate_dataset_cases(db, account.id, payload.cases)
    dataset = Dataset(id=str(uuid.uuid4()), workspace_id=account.id, name=payload.name)
    db.add(dataset)
    db.flush()
    encoded = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    db.add(
        DatasetRevision(
            id=str(uuid.uuid4()),
            dataset_id=dataset.id,
            workspace_id=account.id,
            revision=1,
            sha256=digest,
            cases_json=encoded,
        )
    )
    db.commit()
    return {
        "id": dataset.id,
        "name": dataset.name,
        "revision": 1,
        "sha256": digest,
        "case_count": len(cases),
    }


@app.post("/api/v1/datasets/{dataset_id}/revisions")
def revise_dataset(
    dataset_id: str,
    payload: DatasetCreate,
    account=Depends(current_account),
    db=Depends(db_session),
):
    dataset = owned(db, Dataset, account.id, dataset_id)
    current = db.scalar(
        select(DatasetRevision)
        .where(DatasetRevision.dataset_id == dataset.id, DatasetRevision.workspace_id == account.id)
        .order_by(DatasetRevision.revision.desc())
    )
    cases = validate_dataset_cases(db, account.id, payload.cases)
    encoded = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    rev = DatasetRevision(
        id=str(uuid.uuid4()),
        dataset_id=dataset.id,
        workspace_id=account.id,
        revision=(current.revision + 1 if current else 1),
        sha256=digest,
        cases_json=encoded,
    )
    db.add(rev)
    db.commit()
    return {
        "id": dataset.id,
        "revision": rev.revision,
        "sha256": digest,
        "case_count": len(cases),
    }


@app.get("/api/v1/runs")
def list_runs(account=Depends(current_account), db=Depends(db_session)):
    rows = db.scalars(
        select(EvaluationRun)
        .where(EvaluationRun.workspace_id == account.id)
        .order_by(EvaluationRun.created_at.desc())
    )
    return [
        {
            "id": row.id,
            "dataset_id": row.dataset_id,
            "revision": row.revision,
            "status": row.status,
            "created_at": row.created_at,
            "report": json.loads(row.report_json) if row.report_json else None,
        }
        for row in rows
    ]


@app.post("/api/v1/runs")
def create_run(payload: RunPayload, account=Depends(current_account), db=Depends(db_session)):
    owned(db, Dataset, account.id, payload.dataset_id)
    revision = db.scalar(
        select(DatasetRevision).where(
            DatasetRevision.workspace_id == account.id,
            DatasetRevision.dataset_id == payload.dataset_id,
            DatasetRevision.revision == payload.revision,
        )
    )
    if revision is None:
        raise HTTPException(404, "Dataset revision not found.")
    cases = json.loads(revision.cases_json)
    if not all(case.get("relevant_element_ids") for case in cases):
        raise HTTPException(
            422,
            "Hosted retrieval runs require source-element labels on every query.",
        )
    run_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    db.add(
        EvaluationRun(
            id=run_id,
            workspace_id=account.id,
            dataset_id=payload.dataset_id,
            revision=payload.revision,
            status="queued",
        )
    )
    db.add(
        Job(
            id=job_id,
            workspace_id=account.id,
            kind="evaluation",
            status="queued",
            detail=run_id,
        )
    )
    db.commit()
    try:
        from redis import Redis
        from rq import Queue

        Queue(
            "rag_eval",
            connection=Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0")),
        ).enqueue(
            "apps.worker.tasks.evaluate_dataset",
            account.id,
            run_id,
            job_id,
            revision.id,
            payload.cutoffs,
        )
    except Exception as exc:
        db.execute(
            text("SELECT set_config('app.workspace_id', :wid, true)"), {"wid": account.id}
        ) if db.bind and db.bind.dialect.name == "postgresql" else None
        row = db.get(EvaluationRun, run_id)
        if row:
            row.status = "failed"
        job = db.get(Job, job_id)
        if job:
            job.status = "failed"
            job.detail = "Queue unavailable"
        db.commit()
        raise HTTPException(503, "Evaluation queue is unavailable.") from exc
    return {"id": run_id, "job_id": job_id, "status": "queued"}


@app.get("/api/v1/experiments")
def list_experiments(account=Depends(current_account), db=Depends(db_session)):
    rows = db.scalars(
        select(Experiment)
        .where(Experiment.workspace_id == account.id)
        .order_by(Experiment.created_at.desc())
    )
    return [
        {
            "id": row.id,
            "dataset_id": row.dataset_id,
            "revision": row.revision,
            "status": row.status,
            "created_at": row.created_at,
            "report": json.loads(row.report_json) if row.report_json else None,
        }
        for row in rows
    ]


@app.post("/api/v1/experiments")
def create_experiment(
    payload: ExperimentPayload, account=Depends(current_account), db=Depends(db_session)
):
    if payload.baseline_variant not in {variant.name for variant in payload.variants}:
        raise HTTPException(422, "Baseline must match one of the named variants.")
    owned(db, Dataset, account.id, payload.dataset_id)
    revision = db.scalar(
        select(DatasetRevision).where(
            DatasetRevision.workspace_id == account.id,
            DatasetRevision.dataset_id == payload.dataset_id,
            DatasetRevision.revision == payload.revision,
        )
    )
    if revision is None:
        raise HTTPException(404, "Dataset revision not found.")
    cases = json.loads(revision.cases_json)
    if not all(case.get("relevant_element_ids") for case in cases):
        raise HTTPException(
            422, "Chunking or retrieval experiments require source-element labels on every query."
        )
    experiment_id, job_id = str(uuid.uuid4()), str(uuid.uuid4())
    db.add(
        Experiment(
            id=experiment_id,
            workspace_id=account.id,
            dataset_id=payload.dataset_id,
            revision=payload.revision,
            status="queued",
        )
    )
    db.add(
        Job(
            id=job_id,
            workspace_id=account.id,
            kind="experiment",
            status="queued",
            detail=experiment_id,
        )
    )
    db.commit()
    try:
        from redis import Redis
        from rq import Queue

        Queue(
            "rag_eval",
            connection=Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0")),
        ).enqueue(
            "apps.worker.tasks.run_experiment",
            account.id,
            experiment_id,
            job_id,
            revision.id,
            payload.model_dump(mode="json"),
        )
    except Exception as exc:
        if db.bind and db.bind.dialect.name == "postgresql":
            db.execute(
                text("SELECT set_config('app.workspace_id', :wid, true)"), {"wid": account.id}
            )
        experiment = db.get(Experiment, experiment_id)
        job = db.get(Job, job_id)
        if experiment:
            experiment.status = "failed"
        if job:
            job.status = "failed"
            job.detail = "Queue unavailable"
        db.commit()
        raise HTTPException(503, "Experiment queue is unavailable.") from exc
    return {"id": experiment_id, "job_id": job_id, "status": "queued"}


@app.get("/api/v1/datasets/{dataset_id}/revisions/{revision}")
def get_dataset_revision(
    dataset_id: str, revision: int, account=Depends(current_account), db=Depends(db_session)
):
    owned(db, Dataset, account.id, dataset_id)
    rev = db.scalar(
        select(DatasetRevision).where(
            DatasetRevision.workspace_id == account.id,
            DatasetRevision.dataset_id == dataset_id,
            DatasetRevision.revision == revision,
        )
    )
    if rev is None:
        raise HTTPException(404, "Dataset revision not found.")
    return {"revision": rev.revision, "sha256": rev.sha256, "cases": json.loads(rev.cases_json)}


@app.get("/api/v1/jobs")
def list_jobs(account=Depends(current_account), db=Depends(db_session)):
    return [
        {
            "id": j.id,
            "kind": j.kind,
            "status": j.status,
            "detail": j.detail,
            "created_at": j.created_at,
        }
        for j in db.scalars(
            select(Job).where(Job.workspace_id == account.id).order_by(Job.created_at.desc())
        )
    ]


class CredentialPayload(BaseModel):
    provider: str
    api_key: str = Field(min_length=1, max_length=4096)


@app.put("/api/v1/settings/credentials")
def save_credential(
    payload: CredentialPayload, account=Depends(current_account), db=Depends(db_session)
):
    if payload.provider not in {"gemini", "groq"}:
        raise HTTPException(422, "Chat credentials support Gemini or Groq.")
    key = os.getenv("CREDENTIAL_ENCRYPTION_KEY")
    if not key:
        raise HTTPException(503, "Credential encryption is not configured.")
    encrypted = Fernet(key.encode()).encrypt(payload.api_key.encode()).decode()
    record = db.get(ProviderCredential, (account.id, payload.provider))
    if record:
        record.encrypted_key = encrypted
    else:
        db.add(
            ProviderCredential(
                workspace_id=account.id, provider=payload.provider, encrypted_key=encrypted
            )
        )
    db.commit()
    return {"provider": payload.provider, "saved": True, "masked": True}


@app.get("/api/v1/settings/credentials")
def credentials(account=Depends(current_account), db=Depends(db_session)):
    return [
        {"provider": c.provider, "configured": True, "masked": True}
        for c in db.scalars(
            select(ProviderCredential).where(ProviderCredential.workspace_id == account.id)
        )
    ]


@app.post("/api/v1/settings/credentials/{provider}/check")
async def check_credential(provider: str, account=Depends(current_account), db=Depends(db_session)):
    if provider not in {"gemini", "groq"}:
        raise HTTPException(422, "Only Gemini and Groq tenant credentials can be checked.")
    credential = db.get(ProviderCredential, (account.id, provider))
    encryption_key = os.getenv("CREDENTIAL_ENCRYPTION_KEY")
    if credential is None or not encryption_key:
        raise HTTPException(409, "No saved key is available for this provider.")
    try:
        api_key = (
            Fernet(encryption_key.encode()).decrypt(credential.encrypted_key.encode()).decode()
        )
        async with httpx.AsyncClient(timeout=15) as client:
            if provider == "gemini":
                model = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
                response = await client.get(
                    "https://generativelanguage.googleapis.com/v1beta/models",
                    headers={"x-goog-api-key": api_key},
                )
            else:
                model = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
                response = await client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            response.raise_for_status()
            models = response.json().get("models", response.json().get("data", []))
            model_ids = {
                item.get("name", "").removeprefix("models/") or item.get("id", "")
                for item in models
            }
            if model not in model_ids:
                raise HTTPException(409, "The configured chat model is not available for this key.")
        return {"provider": provider, "available": True, "checked": True, "model": model}
    except (httpx.HTTPError, InvalidToken, ValueError) as exc:
        raise HTTPException(502, "Provider connection check failed.") from exc


@app.delete("/api/v1/settings/credentials/{provider}")
def delete_credential(provider: str, account=Depends(current_account), db=Depends(db_session)):
    record = db.get(ProviderCredential, (account.id, provider))
    if record:
        db.delete(record)
        db.commit()
    return {"deleted": True}


@app.get("/api/v1/capabilities")
async def capabilities(account=Depends(current_account)):
    url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(f"{url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
        return {
            "ollama": {"available": True, "models": [m.get("name") for m in models]},
            "chat_providers": ["ollama", "gemini", "groq"],
        }
    except (httpx.HTTPError, ValueError):
        return {
            "ollama": {"available": False, "models": []},
            "chat_providers": ["ollama", "gemini", "groq"],
        }


@app.get("/api/v1/invitations")
def invitations(x_admin_token: str | None = Header(default=None), db=Depends(db_session)):
    admin_token = os.getenv("INVITATION_ADMIN_TOKEN")
    if not admin_token or not secrets.compare_digest(x_admin_token or "", admin_token):
        raise HTTPException(403, "Invitation administration is restricted.")
    return [{"email": i.email, "created_at": i.created_at} for i in db.scalars(select(Invitation))]


class InvitationPayload(BaseModel):
    email: str


@app.post("/api/v1/invitations")
def invite(
    payload: InvitationPayload,
    x_admin_token: str | None = Header(default=None),
    db=Depends(db_session),
):
    admin_token = os.getenv("INVITATION_ADMIN_TOKEN")
    if not admin_token or not secrets.compare_digest(x_admin_token or "", admin_token):
        raise HTTPException(403, "Invitation administration is restricted.")
    email = payload.email.strip().lower()
    if "@" not in email:
        raise HTTPException(422, "Enter a valid email address.")
    db.merge(Invitation(email=email))
    db.commit()
    return {"email": email, "invited": True}


@app.delete("/api/v1/invitations/{email}")
def revoke_invitation(
    email: str,
    x_admin_token: str | None = Header(default=None),
    db=Depends(db_session),
):
    admin_token = os.getenv("INVITATION_ADMIN_TOKEN")
    if not admin_token or not secrets.compare_digest(x_admin_token or "", admin_token):
        raise HTTPException(403, "Invitation administration is restricted.")
    normalized_email = email.strip().lower()
    account = db.scalar(select(Account).where(Account.email == normalized_email))
    if account:
        db.execute(delete(SessionRecord).where(SessionRecord.account_id == account.id))
    invitation = db.get(Invitation, normalized_email)
    if invitation:
        db.delete(invitation)
        db.commit()
    return {"revoked": True, "active_sessions_ended": account is not None}


@app.get("/api/v1/openapi.json", include_in_schema=False)
def openapi_versioned():
    return app.openapi()


frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        requested = frontend_dist / path
        if path and requested.is_file() and frontend_dist in requested.resolve().parents:
            return FileResponse(requested)
        return FileResponse(frontend_dist / "index.html")
