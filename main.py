# Outcomestar Companion API - standalone FastAPI service.
# v0.6.0 (2026-06-27): generic extractor registry replaces swim-only paste.
# Endpoints:
#   GET  /v1/extract/types                   list registered extractors + context schemas
#   POST /v1/extract/paste                   parse pasted text -> ExtractedEvent[] for ANY event_type
#   POST /v1/import/events                   insert ExtractedEvent[] into events table (Bearer auth)
#   POST /v1/auth/extension-pair             redeem pairing code -> api_token
#   GET  /v1/extension/enrollments           list student_external_identifiers for token's tenant

import os
import re
import json
import secrets
import hashlib
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone, timedelta
from typing import Optional, Annotated, ClassVar

import asyncpg
from fastapi import FastAPI, APIRouter, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

API_VERSION = "0.6.0"
STEPHEN_USER_ID = "019ed384-56d8-77fb-bfe6-00b1d064da18"

# ===== Shared parsing constants =====
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

# ===== FastAPI app =====
app = FastAPI(title="Outcomestar Companion API", version=API_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
router = APIRouter()

# ===== Connection pool =====
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

# ===== Pydantic models =====

# Context fields the frontend should collect per extractor
class ContextField(BaseModel):
    name: str
    label: str
    type: str  # "text" | "date" | "date_range" | "select"
    required: bool = False
    placeholder: Optional[str] = None
    options: Optional[list[str]] = None

class ExtractorInfo(BaseModel):
    event_type: str
    label: str
    description: str
    implemented: bool
    context_schema: list[ContextField]
    example_input: Optional[str] = None

class ExtractedEvent(BaseModel):
    event_type: str
    title: str
    event_date: Optional[str] = None
    event_end_date: Optional[str] = None
    location_name: Optional[str] = None
    details: dict = Field(default_factory=dict)
    source_id: Optional[str] = None
    raw_row_text: Optional[str] = None
    confidence: Optional[float] = None

class ExtractedSkip(BaseModel):
    line: str
    reason: str

class ExtractPasteRequest(BaseModel):
    event_type: str
    raw_text: str
    context: dict = Field(default_factory=dict)

class ExtractPasteResponse(BaseModel):
    event_type: str
    extracted: list[ExtractedEvent]
    skipped: list[ExtractedSkip]
    total_lines: int
    extracted_count: int

class ImportEventsRequest(BaseModel):
    student_id: str
    source_system: str
    external_id: Optional[str] = None
    events: list[ExtractedEvent]

class ImportEventsResponse(BaseModel):
    inserted: int
    skipped_duplicates: int
    total: int
    errors: list[str]

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

# ===== Shared parsing helpers (used by multiple extractors) =====

def parse_time_to_seconds(raw_time: str) -> Optional[float]:
    """'43.43' -> 43.43, '1:25.43' -> 85.43, '1:01:25.43' -> 3685.43"""
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

def parse_signed_time(s: Optional[str]) -> Optional[float]:
    """'-2.73' -> -2.73, '+0.50' -> 0.50"""
    if not s:
        return None
    sign = -1 if s.startswith("-") else 1
    parsed = parse_time_to_seconds(s.lstrip("+-").strip())
    return parsed * sign if parsed is not None else None

def parse_meet_dates(s: Optional[str]) -> tuple[Optional[date], Optional[date]]:
    """'Jun 5-8, 2026' -> (2026-06-05, 2026-06-08)"""
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

# ===== Extractor registry =====

class ExtractorBase(ABC):
    event_type: ClassVar[str]
    label: ClassVar[str]
    description: ClassVar[str]
    implemented: ClassVar[bool] = True
    context_schema: ClassVar[list[ContextField]] = []
    example_input: ClassVar[Optional[str]] = None

    @abstractmethod
    def extract(self, raw_text: str, context: dict) -> tuple[list[ExtractedEvent], list[ExtractedSkip]]:
        raise NotImplementedError

EXTRACTORS: dict[str, ExtractorBase] = {}

def register_extractor(cls):
    EXTRACTORS[cls.event_type] = cls()
    return cls

# ===== Swim race extractor (fully implemented) =====

SWIM_EVENT_PATTERN = re.compile(
    r"(?<!\d)(\d{2,4})\s*([LSY])?\s*(Free(?:style)?|Back(?:stroke)?|Breast(?:stroke)?|Fly|Butterfly|IM|Medley(?:\s*Relay)?(?:\s*\([A-Za-z]+\))?)",
    re.IGNORECASE,
)
SWIM_TIME_PATTERN = re.compile(r"\b(\d{1,2}:\d{2}:\d{2}\.\d{2}|\d{1,2}:\d{2}\.\d{2}|\d{2,3}\.\d{2})\b")
SWIM_PLACE_PATTERN = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
SWIM_ROUND_PATTERN = re.compile(r"\b(Prelims|Finals|Timed Finals|Swim-off|Swim Off|Heats?)\b", re.IGNORECASE)
SWIM_IMPROVEMENT_PATTERN = re.compile(r"(?<![\d.])([+\-])(\d{1,2}:\d{2}\.\d{2}|\d{1,3}\.\d{2})\b")
SWIM_PB_PATTERN = re.compile(r"\b(PB|Personal Best|Lifetime Best|LB)\b", re.IGNORECASE)
HEADER_PATTERN = re.compile(r"^\s*(event|time|place|date|meet|round|improvement|best)\s*$", re.IGNORECASE)

@register_extractor
class SwimRaceExtractor(ExtractorBase):
    event_type = "swim_race"
    label = "Swim Races"
    description = "Paste race results copied from SwimCloud, USA Swimming, or any tabular results source."
    implemented = True
    context_schema = [
        ContextField(name="meet_name", label="Meet Name", type="text",
                     placeholder="e.g., AR NWAA Memorial Classic"),
        ContextField(name="meet_dates", label="Meet Dates", type="text",
                     placeholder="e.g., Jun 5-8, 2026"),
    ]
    example_input = "50 L Breast\t39.58\t1st\t-2.73\n100 L Free\t1:09.42\t5th\n200 Medley Relay (Breast)\t43.43\t1st"

    def _parse_event_string(self, raw_event: str) -> Optional[dict]:
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
            "distance_m": distance, "stroke_short": stroke_short, "stroke_long": rest,
            "course_code": course_code, "course_long": COURSE_MAP.get(course_code),
            "is_relay": is_relay, "relay_stroke": relay_stroke,
        }

    def _normalize_round(self, raw_round: Optional[str]) -> Optional[str]:
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

    def _normalize_place(self, raw_place: Optional[str]) -> Optional[int]:
        if not raw_place:
            return None
        m = re.match(r"(\d+)", raw_place.strip())
        return int(m.group(1)) if m else None

    def _build_source_id(self, meet_start, distance, stroke_short, course_long, swim_time, round_normalized, is_relay):
        date_part = meet_start.strftime("%Y%m%d") if meet_start else "00000000"
        course = course_long or "XXX"
        s = f"{date_part}_{distance}{stroke_short}_{course}_{swim_time}"
        if is_relay:
            s += "r"
        elif round_normalized == "prelims":
            s += "p"
        return s

    def extract(self, raw_text: str, context: dict) -> tuple[list[ExtractedEvent], list[ExtractedSkip]]:
        extracted: list[ExtractedEvent] = []
        skipped: list[ExtractedSkip] = []
        if not raw_text or not raw_text.strip():
            return extracted, skipped
        meet_name = context.get("meet_name")
        meet_dates_raw = context.get("meet_dates")
        meet_start, meet_end = parse_meet_dates(meet_dates_raw)

        lines = [ln.strip() for ln in raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
        lines = [ln for ln in lines if ln]
        for ln in lines:
            if HEADER_PATTERN.match(ln):
                skipped.append(ExtractedSkip(line=ln, reason="header"))
                continue
            event_match = SWIM_EVENT_PATTERN.search(ln)
            time_match = SWIM_TIME_PATTERN.search(ln)
            if not (event_match and time_match):
                reason = "no event/time pattern"
                if event_match or time_match:
                    reason = "incomplete: needs both event and time"
                skipped.append(ExtractedSkip(line=ln, reason=reason))
                continue
            raw_event = event_match.group(0).strip()
            raw_time = time_match.group(1)
            ev = self._parse_event_string(raw_event)
            if not ev:
                skipped.append(ExtractedSkip(line=ln, reason=f"could not parse event: {raw_event}"))
                continue
            place_match = SWIM_PLACE_PATTERN.search(ln)
            raw_place = f"{place_match.group(1)}{place_match.group(2).lower()}" if place_match else None
            round_match = SWIM_ROUND_PATTERN.search(ln)
            raw_round = round_match.group(1) if round_match else None
            imp_match = SWIM_IMPROVEMENT_PATTERN.search(ln)
            improvement = f"{imp_match.group(1)}{imp_match.group(2)}" if imp_match else None
            is_pb = bool(SWIM_PB_PATTERN.search(ln))
            round_norm = self._normalize_round(raw_round)
            place_int = self._normalize_place(raw_place)
            time_sec = parse_time_to_seconds(raw_time)
            imp_sec = parse_signed_time(improvement)
            source_id = self._build_source_id(meet_start, ev["distance_m"], ev["stroke_short"],
                                               ev["course_long"], raw_time, round_norm, ev["is_relay"])
            title_date = f" ({meet_start.isoformat()})" if meet_start else ""
            title = f"{ev['distance_m']} {ev['stroke_long']} {ev['course_long'] or ''} {raw_time}{title_date}".strip()
            details = {
                "distance_m": ev["distance_m"],
                "stroke": ev["stroke_short"],
                "stroke_long": ev["stroke_long"],
                "course": ev["course_long"],
                "course_code": ev["course_code"],
                "is_relay": ev["is_relay"],
                "relay_stroke": ev["relay_stroke"],
                "round": round_norm,
                "raw_round": raw_round,
                "swim_time": raw_time,
                "time_seconds": time_sec,
                "place": place_int,
                "is_personal_best": is_pb,
                "improvement_seconds": imp_sec,
                "meet": meet_name,
                "meet_dates_raw": meet_dates_raw,
                "extracted_via": "outcomestar_paste_extractor",
                "extractor_version": "1.0",
            }
            details = {k: v for k, v in details.items() if v is not None}
            extracted.append(ExtractedEvent(
                event_type="swim_race",
                title=title,
                event_date=meet_start.isoformat() if meet_start else None,
                event_end_date=meet_end.isoformat() if meet_end else None,
                location_name=meet_name,
                details=details,
                source_id=source_id,
                raw_row_text=ln[:250],
                confidence=0.9,
            ))
        return extracted, skipped

# ===== Stub extractors (architecture supports them; implementation pending) =====

@register_extractor
class CompetitionExtractor(ExtractorBase):
    event_type = "competition"
    label = "Competitions"
    description = "Academic, STEM, or robotics competition results (AMC, MATHCOUNTS, FLL, debate, etc.). Implementation pending — paste samples in a future session and we will build."
    implemented = False
    context_schema = [
        ContextField(name="competition_name", label="Competition", type="text",
                     placeholder="e.g., AMC 8, MATHCOUNTS Chapter"),
        ContextField(name="event_date", label="Event Date", type="date"),
    ]
    def extract(self, raw_text, context):
        return ([], [ExtractedSkip(line="(entire input)", reason="competition extractor not yet implemented")])

@register_extractor
class MusicPerformanceExtractor(ExtractorBase):
    event_type = "music_performance"
    label = "Music Performances"
    description = "Recitals, ensemble performances, or solo programs. Implementation pending."
    implemented = False
    context_schema = [
        ContextField(name="venue", label="Venue", type="text"),
        ContextField(name="event_date", label="Performance Date", type="date"),
        ContextField(name="role", label="Role", type="select", options=["soloist", "ensemble", "section_lead", "accompanist"]),
    ]
    def extract(self, raw_text, context):
        return ([], [ExtractedSkip(line="(entire input)", reason="music performance extractor not yet implemented")])

# ===== Auth =====
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

# ===== Endpoints =====

@app.get("/")
async def health():
    return {"status": "ok", "service": "outcomestar-companion-api", "version": API_VERSION}

@router.get("/v1/extract/types", response_model=list[ExtractorInfo], tags=["extract"])
async def list_extractor_types():
    return [
        ExtractorInfo(
            event_type=e.event_type,
            label=e.label,
            description=e.description,
            implemented=e.implemented,
            context_schema=list(e.context_schema),
            example_input=e.example_input,
        )
        for e in EXTRACTORS.values()
    ]

@router.post("/v1/extract/paste", response_model=ExtractPasteResponse, tags=["extract"])
async def extract_paste(req: ExtractPasteRequest):
    extractor = EXTRACTORS.get(req.event_type)
    if not extractor:
        raise HTTPException(
            status_code=400,
            detail=f"No extractor for event_type={req.event_type}. Available: {sorted(EXTRACTORS.keys())}",
        )
    if not extractor.implemented:
        return ExtractPasteResponse(
            event_type=req.event_type, extracted=[],
            skipped=[ExtractedSkip(line="(entire input)", reason=f"{extractor.label} extractor not yet implemented")],
            total_lines=0, extracted_count=0,
        )
    extracted, skipped = extractor.extract(req.raw_text, req.context or {})
    total_lines = len([ln for ln in req.raw_text.replace("\r\n", "\n").split("\n") if ln.strip()])
    return ExtractPasteResponse(
        event_type=req.event_type, extracted=extracted, skipped=skipped,
        total_lines=total_lines, extracted_count=len(extracted),
    )

@router.post("/v1/import/events", response_model=ImportEventsResponse, tags=["import"])
async def import_events(req: ImportEventsRequest, auth: dict = Depends(authenticate)):
    if req.student_id not in auth["student_ids"]:
        raise HTTPException(status_code=403, detail="Token not authorized for this student")
    errors: list[str] = []
    inserted = 0
    skipped_duplicates = 0
    async with db_conn(tenant_id=auth["tenant_id"]) as conn:
        existing_rows = await conn.fetch(
            "SELECT source_id FROM events WHERE student_id = $1::uuid AND source_system = $2 AND deleted_at IS NULL AND source_id IS NOT NULL",
            req.student_id, req.source_system,
        )
        existing_ids = {r["source_id"] for r in existing_rows}
        to_insert = []
        for ev in req.events:
            try:
                if ev.source_id and ev.source_id in existing_ids:
                    skipped_duplicates += 1
                    continue
                if ev.source_id:
                    existing_ids.add(ev.source_id)
                start_d = date.fromisoformat(ev.event_date) if ev.event_date else None
                end_d = date.fromisoformat(ev.event_end_date) if ev.event_end_date else None
                to_insert.append((
                    auth["tenant_id"], req.student_id, ev.event_type, ev.title,
                    start_d, end_d, ev.location_name,
                    json.dumps(ev.details or {}), "private", req.source_system, ev.source_id,
                    STEPHEN_USER_ID,
                ))
            except Exception as e:
                errors.append(f"event {ev.title!r}: {type(e).__name__}: {e}")
        if to_insert:
            async with conn.transaction():
                await conn.executemany(
                    "INSERT INTO events (tenant_id, student_id, event_type, title, event_date, event_end_date, location_name, details, visibility, source_system, source_id, created_by) VALUES ($1::uuid, $2::uuid, $3::event_type_enum, $4, $5, $6, $7, $8::jsonb, $9::visibility_enum, $10, $11, $12::uuid)",
                    to_insert,
                )
                inserted = len(to_insert)
        if req.external_id:
            await conn.execute(
                "UPDATE student_external_identifiers SET last_synced_at = now(), last_sync_status = $1, last_sync_summary = $2, updated_at = now() WHERE student_id = $3::uuid AND system_name = $4 AND external_id = $5",
                "success" if not errors else ("partial" if inserted > 0 else "failure"),
                f"Paste import: {inserted} inserted, {skipped_duplicates} duplicates, {len(errors)} errors",
                req.student_id, req.source_system, req.external_id,
            )
    return ImportEventsResponse(
        inserted=inserted, skipped_duplicates=skipped_duplicates,
        total=len(req.events), errors=errors[:20],
    )

@router.post("/v1/auth/extension-pair", response_model=PairResponse, tags=["auth"])
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
        api_token=raw_token, tenant_id=str(row["tenant_id"]),
        student_ids=[str(s) for s in row["student_ids"]],
        expires_at=token_expires_at.isoformat(), name=row["name"],
    )

@router.get("/v1/extension/enrollments", response_model=list[EnrollmentItem], tags=["auth"])
async def get_enrollments(auth: dict = Depends(authenticate)):
    async with db_conn(tenant_id=auth["tenant_id"]) as conn:
        rows = await conn.fetch(
            "SELECT student_id, system_name, external_id, external_url, is_primary, last_synced_at, last_sync_status FROM student_external_identifiers WHERE student_id = ANY($1::uuid[]) AND deleted_at IS NULL ORDER BY student_id, system_name",
            [auth["student_ids"]],
        )
    return [
        EnrollmentItem(
            student_id=str(r["student_id"]), system_name=r["system_name"],
            external_id=r["external_id"], external_url=r["external_url"],
            is_primary=r["is_primary"],
            last_synced_at=r["last_synced_at"].isoformat() if r["last_synced_at"] else None,
            last_sync_status=r["last_sync_status"],
        )
        for r in rows
    ]

app.include_router(router)
