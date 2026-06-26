# Outcomestar Companion API - standalone FastAPI service.
# Endpoints:
#   POST /v1/auth/extension-pair             redeem pairing code -> api_token
#   GET  /v1/extension/enrollments           list student_external_identifiers for token's tenant
#   POST /v1/extension/external-data-import  receive race payload, insert events
#   POST /v1/extract/paste                   parse pasted race text into RaceInput[]  (NEW v0.5.0)
# v0.5.0 (2026-06-27): paste-text extraction added; extension architecture deprecated.

import os
import re
import json
import secrets
import hashlib
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone, timedelta
from typing import Optional, Annotated

import asyncpg
from fastapi import FastAPI, APIRouter, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

API_VERSION = "0.5.0"
STEPHEN_USER_ID = "019ed384-56d8-77fb-bfe6-00b1d064da18"

COURSE_MAP = {"L": "LCM", "S": "SCM", "Y": "SCY"}
STROKE_MAP = {
    "free": "FR", "freestyle": "FR",
    "back": "BK", "backstroke": "BK",
    "breast": "BR", "breaststroke": "BR",
    "fly": "FL", "butterfly": "FL",
    "im": "IM", "medley": "IM",
}
MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

app = FastAPI(title="Outcomestar Companion API", version=API_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
router = APIRouter()

_pool: Optional[asyncpg.Pool] = None

async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            os.environ["DATABASE_URL"],
            min_size=1, max_size=5, command_timeout=30,
        )
    return _pool

@asynccontextmanager
async def db_conn(tenant_id: Optional[str] = None):
    pool = await get_pool()
    async with pool.acquire() as conn:
        if tenant_id:
            await conn.execute(f"SET app.current_tenant_id = '{tenant_id}'")
        yield conn

class PairRequest(BaseModel):
    code: str

class PairResponse(BaseModel):
    api_token: str
    tenant_id: str
    student_ids: list[str]
    expires_at: Optional[str]
    name: Optional[str] = None

class EnrollmentItem(BaseModel):
    student_id: str
    system_name: str
    external_id: str
    external_url: Optional[str]
    is_primary: bool
    last_synced_at: Optional[str]
    last_sync_status: Optional[str]

class RaceInput(BaseModel):
    raw_event: str
    raw_round: Optional[str] = None
    raw_time: str
    raw_place: Optional[str] = None
    is_personal_best: bool = False
    improvement: Optional[str] = None
    raw_row_text: Optional[str] = None
    meet_name: Optional[str] = None
    meet_dates: Optional[str] = None

class HighlightInput(BaseModel):
    year: Optional[int] = None
    meet: Optional[str] = None
    placement: Optional[int] = None
    raw_event: str
    raw_time: str
    raw_match: Optional[str] = None
    raw_row_text: Optional[str] = None

class TeamInput(BaseModel):
    name: str
    location: Optional[str] = None
    href: Optional[str] = None

class ProfileInput(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    primary_team: Optional[str] = None

class ImportRequest(BaseModel):
    system_name: str
    external_id: str
    page_url: str
    scraped_at: str
    extractor_version: Optional[str] = None
    framework_detected: Optional[str] = None
    profile: Optional[ProfileInput] = None
    races: list[RaceInput] = Field(default_factory=list)
    highlights: list[HighlightInput] = Field(default_factory=list)
    teams: list[TeamInput] = Field(default_factory=list)
    rankings: list[dict] = Field(default_factory=list)
    raw_html_archive: Optional[str] = None

class ImportResponse(BaseModel):
    inserted: int
    skipped: int
    total: int
    student_id: str
    errors: list[str]

class ExtractPasteRequest(BaseModel):
    raw_text: str
    meet_name: Optional[str] = None
    meet_dates: Optional[str] = None

class ExtractedSkip(BaseModel):
    line: str
    reason: str

class ExtractPasteResponse(BaseModel):
    extracted: list[RaceInput]
    skipped: list[ExtractedSkip]
    total_lines: int
    extracted_count: int

def parse_event(raw_event):
    if not raw_event:
        return None
    s = raw_event.strip()
    dist_match = re.match(r"^(\d+)", s)
    if not dist_match:
        return None
    distance = int(dist_match.group(1))
    rest = s[dist_match.end():].strip()
    course_code = None
    course_match = re.match(r"^([LSY])\s+", rest)
    if course_match:
        course_code = course_match.group(1)
        rest = rest[course_match.end():].strip()
    is_relay = "relay" in rest.lower()
    relay_stroke = None
    if is_relay:
        paren_match = re.search(r"\(([A-Za-z]+)\)", rest)
        if paren_match:
            relay_stroke = STROKE_MAP.get(paren_match.group(1).lower())
        stroke_short = "MED"
    else:
        stroke_word = rest.split()[0] if rest else None
        stroke_short = STROKE_MAP.get(stroke_word.lower()) if stroke_word else None
    if not stroke_short:
        return None
    return {
        "distance_m": distance,
        "stroke_short": stroke_short,
        "stroke_long": rest,
        "course_code": course_code,
        "course_long": COURSE_MAP.get(course_code),
        "is_relay": is_relay,
        "relay_stroke": relay_stroke,
    }

def parse_time(raw_time):
    if not raw_time:
        return None
    parts = raw_time.strip().split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except (ValueError, TypeError):
        return None
    return None

def parse_improvement(s):
    if not s:
        return None
    sign = -1 if s.startswith("-") else 1
    parsed = parse_time(s.lstrip("+-").strip())
    return parsed * sign if parsed is not None else None

def parse_meet_dates(s):
    if not s:
        return (None, None)
    s = s.strip()
    year_match = re.search(r"(\d{4})$", s)
    if not year_match:
        return (None, None)
    year = int(year_match.group(1))
    range_match = re.match(r"^([A-Za-z]+)\s+(\d{1,2})\s*[\-\u2013\u2014]\s*(\d{1,2}),?\s*\d{4}$", s)
    if range_match:
        m = MONTH_MAP.get(range_match.group(1).lower()[:3])
        if m:
            try:
                return (date(year, m, int(range_match.group(2))), date(year, m, int(range_match.group(3))))
            except ValueError:
                pass
    single_match = re.match(r"^([A-Za-z]+)\s+(\d{1,2}),?\s*\d{4}$", s)
    if single_match:
        m = MONTH_MAP.get(single_match.group(1).lower()[:3])
        if m:
            try:
                d = date(year, m, int(single_match.group(2)))
                return (d, d)
            except ValueError:
                pass
    return (None, None)

def normalize_round(raw_round):
    if not raw_round:
        return None
    r = raw_round.lower().strip()
    if "timed" in r:
        return "timed_finals"
    if "final" in r:
        return "finals"
    if "prelim" in r or "heat" in r:
        return "prelims"
    if "swim-off" in r or "swim off" in r:
        return "swim_off"
    return r.replace(" ", "_")

def normalize_place(raw_place):
    if not raw_place:
        return None
    m = re.match(r"(\d+)", raw_place.strip())
    return int(m.group(1)) if m else None

def build_source_id(meet_start, distance, stroke_short, course_long, swim_time, round_normalized, is_relay):
    date_part = meet_start.strftime("%Y%m%d") if meet_start else "00000000"
    course = course_long or "XXX"
    s = f"{date_part}_{distance}{stroke_short}_{course}_{swim_time}"
    if is_relay:
        s += "r"
    elif round_normalized == "prelims":
        s += "p"
    return s

# Regex patterns for paste extraction.
# Tolerates tabs, spaces, en/em dashes, mixed column orders.
EVENT_PATTERN = re.compile(
    r"(?<!\d)(\d{2,4})\s*([LSY])?\s*(Free(?:style)?|Back(?:stroke)?|Breast(?:stroke)?|Fly|Butterfly|IM|Medley(?:\s*Relay)?(?:\s*\([A-Za-z]+\))?)",
    re.IGNORECASE,
)
TIME_PATTERN = re.compile(r"\b(\d{1,2}:\d{2}:\d{2}\.\d{2}|\d{1,2}:\d{2}\.\d{2}|\d{2,3}\.\d{2})\b")
PLACE_PATTERN = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
ROUND_PATTERN = re.compile(r"\b(Prelims|Finals|Timed Finals|Swim-off|Swim Off|Heats?)\b", re.IGNORECASE)
IMPROVEMENT_PATTERN = re.compile(r"(?<![\d.])([+\-])(\d{1,2}:\d{2}\.\d{2}|\d{1,3}\.\d{2})\b")
PB_PATTERN = re.compile(r"\b(PB|Personal Best|Lifetime Best|LB|Best)\b", re.IGNORECASE)
HEADER_PATTERN = re.compile(r"^\s*(event|time|place|date|meet|round|improvement|best)\s*$", re.IGNORECASE)

def extract_races_from_text(raw_text: str, meet_name: Optional[str], meet_dates: Optional[str]) -> tuple[list[RaceInput], list[ExtractedSkip]]:
    extracted: list[RaceInput] = []
    skipped: list[ExtractedSkip] = []
    if not raw_text or not raw_text.strip():
        return extracted, skipped
    # Normalize whitespace within lines but preserve newlines
    lines = [ln.strip() for ln in raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [ln for ln in lines if ln]
    for ln in lines:
        # Skip obvious header lines (single column header)
        if HEADER_PATTERN.match(ln):
            skipped.append(ExtractedSkip(line=ln, reason="header"))
            continue
        event_match = EVENT_PATTERN.search(ln)
        time_match = TIME_PATTERN.search(ln)
        if not (event_match and time_match):
            # Not a race row — could be meet header, blank, or text
            if event_match or time_match:
                skipped.append(ExtractedSkip(line=ln, reason="incomplete: needs both event and time"))
            else:
                skipped.append(ExtractedSkip(line=ln, reason="no event/time pattern"))
            continue
        raw_event = event_match.group(0).strip()
        raw_time = time_match.group(1)
        # Place
        place_match = PLACE_PATTERN.search(ln)
        raw_place = f"{place_match.group(1)}{place_match.group(2).lower()}" if place_match else None
        # Round
        round_match = ROUND_PATTERN.search(ln)
        raw_round = round_match.group(1) if round_match else None
        # Improvement
        imp_match = IMPROVEMENT_PATTERN.search(ln)
        improvement = f"{imp_match.group(1)}{imp_match.group(2)}" if imp_match else None
        # PB flag
        is_pb = bool(PB_PATTERN.search(ln))
        extracted.append(RaceInput(
            raw_event=raw_event,
            raw_round=raw_round,
            raw_time=raw_time,
            raw_place=raw_place,
            is_personal_best=is_pb,
            improvement=improvement,
            raw_row_text=ln[:250],
            meet_name=meet_name,
            meet_dates=meet_dates,
        ))
    return extracted, skipped

async def authenticate(authorization: Annotated[Optional[str], Header()] = None):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    raw_token = authorization[7:].strip()
    if not raw_token:
        raise HTTPException(status_code=401, detail="Empty token")
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    async with db_conn() as conn:
        row = await conn.fetchrow(
            "SELECT tenant_id, student_ids, scope, revoked_at, expires_at FROM api_tokens WHERE token_hash = $1",
            token_hash,
        )
        if not row:
            raise HTTPException(status_code=401, detail="Invalid token")
        if row["revoked_at"]:
            raise HTTPException(status_code=401, detail="Token revoked")
        if row["expires_at"] and row["expires_at"] < datetime.now(timezone.utc):
            raise HTTPException(status_code=401, detail="Token expired")
        await conn.execute(
            "UPDATE api_tokens SET last_used_at = now() WHERE token_hash = $1",
            token_hash,
        )
    return {
        "tenant_id": str(row["tenant_id"]),
        "student_ids": [str(s) for s in row["student_ids"]],
        "scope": row["scope"],
    }

@app.get("/")
async def health():
    return {"status": "ok", "service": "outcomestar-companion-api", "version": API_VERSION}

@router.post("/v1/extract/paste", response_model=ExtractPasteResponse, tags=["extract"])
async def extract_paste(req: ExtractPasteRequest):
    # Parser-only endpoint. No auth, no DB writes. Pure text-to-RaceInput[] transform.
    # Frontend posts pasted text, gets back structured rows for review, then user
    # confirms via the existing /v1/extension/external-data-import endpoint.
    extracted, skipped = extract_races_from_text(req.raw_text, req.meet_name, req.meet_dates)
    return ExtractPasteResponse(
        extracted=extracted,
        skipped=skipped,
        total_lines=len([ln for ln in req.raw_text.replace("\r\n", "\n").split("\n") if ln.strip()]),
        extracted_count=len(extracted),
    )

@router.post("/v1/auth/extension-pair", response_model=PairResponse, tags=["extension"])
async def extension_pair(req: PairRequest):
    code = req.code.strip().upper()
    if not code or not (4 <= len(code) <= 32):
        raise HTTPException(status_code=400, detail="Invalid code format")
    async with db_conn() as conn:
        row = await conn.fetchrow(
            "SELECT tenant_id, student_ids, name, expires_at, redeemed_at, created_by FROM pairing_codes WHERE code = $1",
            code,
        )
        if not row:
            raise HTTPException(status_code=404, detail="Pairing code not found")
        if row["redeemed_at"]:
            raise HTTPException(status_code=410, detail="Pairing code already redeemed")
        if row["expires_at"] < datetime.now(timezone.utc):
            raise HTTPException(status_code=410, detail="Pairing code expired")
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        token_expires_at = datetime.now(timezone.utc) + timedelta(days=180)
        token_id = await conn.fetchval(
            "INSERT INTO api_tokens (tenant_id, token_hash, student_ids, name, scope, created_by, expires_at) VALUES ($1, $2, $3, $4, 'extension', $5, $6) RETURNING id",
            row["tenant_id"], token_hash, row["student_ids"],
            row["name"], row["created_by"], token_expires_at,
        )
        await conn.execute(
            "UPDATE pairing_codes SET redeemed_at = now(), redeemed_token_id = $1 WHERE code = $2",
            token_id, code,
        )
    return PairResponse(
        api_token=raw_token,
        tenant_id=str(row["tenant_id"]),
        student_ids=[str(s) for s in row["student_ids"]],
        expires_at=token_expires_at.isoformat(),
        name=row["name"],
    )

@router.get("/v1/extension/enrollments", response_model=list[EnrollmentItem], tags=["extension"])
async def get_enrollments(auth: dict = Depends(authenticate)):
    async with db_conn(tenant_id=auth["tenant_id"]) as conn:
        rows = await conn.fetch(
            "SELECT student_id, system_name, external_id, external_url, is_primary, last_synced_at, last_sync_status FROM student_external_identifiers WHERE student_id = ANY($1::uuid[]) AND deleted_at IS NULL ORDER BY student_id, system_name",
            [auth["student_ids"]],
        )
    return [
        EnrollmentItem(
            student_id=str(r["student_id"]),
            system_name=r["system_name"],
            external_id=r["external_id"],
            external_url=r["external_url"],
            is_primary=r["is_primary"],
            last_synced_at=r["last_synced_at"].isoformat() if r["last_synced_at"] else None,
            last_sync_status=r["last_sync_status"],
        )
        for r in rows
    ]

@router.post("/v1/extension/external-data-import", response_model=ImportResponse, tags=["extension"])
async def import_external_data(req: ImportRequest, auth: dict = Depends(authenticate)):
    errors = []
    inserted = 0
    skipped = 0
    async with db_conn(tenant_id=auth["tenant_id"]) as conn:
        student_row = await conn.fetchrow(
            "SELECT student_id FROM student_external_identifiers WHERE system_name = $1 AND external_id = $2 AND deleted_at IS NULL AND student_id = ANY($3::uuid[]) LIMIT 1",
            req.system_name, req.external_id, auth["student_ids"],
        )
        if not student_row:
            raise HTTPException(status_code=404, detail=f"No enrollment for system={req.system_name} external_id={req.external_id}")
        student_id = str(student_row["student_id"])
        default_start, default_end = parse_meet_dates(req.races[0].meet_dates if req.races else None)
        existing_rows = await conn.fetch(
            "SELECT source_id FROM events WHERE student_id = $1::uuid AND source_system = $2 AND deleted_at IS NULL AND source_id IS NOT NULL",
            student_id, req.system_name,
        )
        existing_ids = {r["source_id"] for r in existing_rows}
        to_insert = []
        for r in req.races:
            try:
                ev = parse_event(r.raw_event)
                if not ev:
                    errors.append(f"unparseable event: {r.raw_event!r}")
                    continue
                time_seconds = parse_time(r.raw_time)
                round_norm = normalize_round(r.raw_round)
                place_int = normalize_place(r.raw_place)
                imp_seconds = parse_improvement(r.improvement)
                start_d, end_d = (default_start, default_end)
                if r.meet_dates:
                    s2, e2 = parse_meet_dates(r.meet_dates)
                    if s2:
                        start_d, end_d = s2, e2
                source_id = build_source_id(start_d, ev["distance_m"], ev["stroke_short"], ev["course_long"], r.raw_time, round_norm, ev["is_relay"])
                if source_id in existing_ids:
                    skipped += 1
                    continue
                existing_ids.add(source_id)
                title_date = f" ({start_d.isoformat()})" if start_d else ""
                title = f"{ev['distance_m']} {ev['stroke_long']} {ev['course_long'] or ''} {r.raw_time}{title_date}".strip()
                details = {
                    "distance_m": ev["distance_m"],
                    "stroke": ev["stroke_short"],
                    "stroke_long": ev["stroke_long"],
                    "course": ev["course_long"],
                    "course_code": ev["course_code"],
                    "is_relay": ev["is_relay"],
                    "relay_stroke": ev["relay_stroke"],
                    "round": round_norm,
                    "raw_round": r.raw_round,
                    "swim_time": r.raw_time,
                    "time_seconds": time_seconds,
                    "place": place_int,
                    "is_personal_best": r.is_personal_best,
                    "improvement_seconds": imp_seconds,
                    "meet": r.meet_name,
                    "meet_dates_raw": r.meet_dates,
                    "raw_row_text": r.raw_row_text,
                    "extracted_via": "outcomestar_companion",
                    "extractor_version": req.extractor_version,
                }
                details = {k: v for k, v in details.items() if v is not None}
                to_insert.append((
                    auth["tenant_id"], student_id, "swim_race", title,
                    start_d, end_d, r.meet_name,
                    json.dumps(details), "private", req.system_name, source_id,
                    STEPHEN_USER_ID,
                ))
            except Exception as e:
                errors.append(f"race {r.raw_event!r}: {type(e).__name__}: {e}")
        if to_insert:
            async with conn.transaction():
                await conn.executemany(
                    "INSERT INTO events (tenant_id, student_id, event_type, title, event_date, event_end_date, location_name, details, visibility, source_system, source_id, created_by) VALUES ($1::uuid, $2::uuid, $3::event_type_enum, $4, $5, $6, $7, $8::jsonb, $9::visibility_enum, $10, $11, $12::uuid)",
                    to_insert,
                )
                inserted = len(to_insert)
        status_value = "success" if not errors else ("partial" if inserted > 0 else "failure")
        await conn.execute(
            "UPDATE student_external_identifiers SET last_synced_at = now(), last_sync_status = $1, last_sync_summary = $2, updated_at = now() WHERE student_id = $3::uuid AND system_name = $4 AND external_id = $5",
            status_value,
            f"Extension v{req.extractor_version or '?'}: {inserted} inserted, {skipped} skipped, {len(errors)} errors",
            student_id, req.system_name, req.external_id,
        )
        if req.raw_html_archive:
            await conn.execute(
                "INSERT INTO archive_entries (tenant_id, archive_type, archive_date, version, title, summary, detail, source, source_id, visibility, created_by) VALUES ($1::uuid, 'client_scrape_html', CURRENT_DATE, $2, $3, $4, $5, 'outcomestar_companion', $6, 'private', $7::uuid)",
                auth["tenant_id"], req.extractor_version or "unknown",
                f"Client scrape: {req.system_name} {req.external_id}",
                f"Captured {inserted} new races, {skipped} duplicates skipped",
                req.raw_html_archive,
                f"client_scrape_{req.system_name}_{req.external_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                STEPHEN_USER_ID,
            )
    return ImportResponse(
        inserted=inserted, skipped=skipped, total=len(req.races),
        student_id=student_id, errors=errors[:20],
    )

app.include_router(router)
