from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import structlog

from src.agent.config import Config

logger = structlog.get_logger(__name__)


def _safe_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


@dataclass
class Consents:
    address_collection: bool = False
    contact_collection: bool = False
    follow_up: bool = False


@dataclass
class SpecialistMeeting:
    requested: bool = False
    date: Optional[str] = None
    time: Optional[str] = None
    timezone: Optional[str] = None
    channel: Optional[str] = None
    notes: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class RenewablesLeadMemory:
    user_name: Optional[str] = None
    user_is_minor: Optional[bool] = None
    user_address: Optional[str] = None
    user_location_general: Optional[str] = None
    home_type: Optional[str] = None
    ownership: Optional[str] = None
    primary_goal: Optional[str] = None
    monthly_bill_range: Optional[str] = None
    major_loads: list[str] = field(default_factory=list)
    interest_area: list[str] = field(default_factory=list)
    objections: list[str] = field(default_factory=list)
    preferred_contact_method: Optional[str] = None
    contact_value: Optional[str] = None
    timezone: str = "Africa/Tunis"
    specialist_meeting: SpecialistMeeting = field(default_factory=SpecialistMeeting)
    consents: Consents = field(default_factory=Consents)

    def update_from(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return

        for key in (
            "user_name",
            "user_address",
            "user_location_general",
            "home_type",
            "ownership",
            "primary_goal",
            "monthly_bill_range",
            "preferred_contact_method",
            "contact_value",
            "timezone",
        ):
            if key in data and data[key] is not None:
                setattr(self, key, str(data[key]).strip() or None)

        if "user_is_minor" in data and data["user_is_minor"] is not None:
            self.user_is_minor = bool(data["user_is_minor"])

        if "major_loads" in data:
            self.major_loads = _safe_list(data.get("major_loads"))

        if "interest_area" in data:
            self.interest_area = _safe_list(data.get("interest_area"))

        if "objections" in data:
            self.objections = _safe_list(data.get("objections"))

        consents = data.get("consents")
        if isinstance(consents, dict):
            if "address_collection" in consents and consents["address_collection"] is not None:
                self.consents.address_collection = bool(consents["address_collection"])
            if "contact_collection" in consents and consents["contact_collection"] is not None:
                self.consents.contact_collection = bool(consents["contact_collection"])
            if "follow_up" in consents and consents["follow_up"] is not None:
                self.consents.follow_up = bool(consents["follow_up"])

        meeting = data.get("specialist_meeting")
        if isinstance(meeting, dict):
            if "requested" in meeting and meeting["requested"] is not None:
                self.specialist_meeting.requested = bool(meeting["requested"])
            for key in ("date", "time", "timezone", "channel", "notes", "request_id"):
                if key in meeting and meeting[key] is not None:
                    setattr(self.specialist_meeting, key, str(meeting[key]).strip() or None)

    def to_public_dict(self) -> dict[str, Any]:
        return asdict(self)


class RenewablesToolExecutor:
    """
    Minimal tool layer for OpenAI Realtime function calling.

    This does not integrate a real calendar/CRM yet; it captures intent + data and
    returns structured results the model can speak about honestly.
    """

    def __init__(self, *, config: Config, call_sid: str):
        self.config = config
        self.call_sid = call_sid
        self.lead_id = f"lead_{uuid.uuid4().hex[:10]}"
        self.memory = RenewablesLeadMemory()

    def tool_definitions(self) -> list[dict[str, Any]]:
        common_note = (
            "Return JSON. If something can't be completed, return ok=false and a short reason."
        )
        return [
            {
                "type": "function",
                "name": "create_lead",
                "description": (
                    "Create/update the lead record for this caller (name, goals, address/contact only with consent). "
                    + common_note
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_name": {"type": "string"},
                        "user_is_minor": {"type": "boolean"},
                        "user_address": {"type": "string"},
                        "user_location_general": {"type": "string"},
                        "home_type": {"type": "string"},
                        "ownership": {"type": "string"},
                        "primary_goal": {"type": "string"},
                        "monthly_bill_range": {"type": "string"},
                        "major_loads": {"type": "array", "items": {"type": "string"}},
                        "interest_area": {"type": "array", "items": {"type": "string"}},
                        "objections": {"type": "array", "items": {"type": "string"}},
                        "preferred_contact_method": {"type": "string"},
                        "contact_value": {"type": "string"},
                        "timezone": {"type": "string"},
                        "specialist_meeting": {
                            "type": "object",
                            "properties": {
                                "requested": {"type": "boolean"},
                                "date": {"type": "string"},
                                "time": {"type": "string"},
                                "timezone": {"type": "string"},
                                "channel": {"type": "string"},
                                "notes": {"type": "string"},
                                "request_id": {"type": "string"},
                            },
                            "additionalProperties": False,
                        },
                        "consents": {
                            "type": "object",
                            "properties": {
                                "address_collection": {"type": "boolean"},
                                "contact_collection": {"type": "boolean"},
                                "follow_up": {"type": "boolean"},
                            },
                            "additionalProperties": False,
                        },
                        "notes": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "lookup_service_area",
                "description": (
                    "Check whether the company serves a given address/city/region. "
                    "If unknown, return supported=null and a next_step suggestion. "
                    + common_note
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_estimate",
                "description": (
                    "Return a rough, non-binding estimate category and assumptions (no guarantees). "
                    + common_note
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "primary_goal": {"type": "string"},
                        "home_type": {"type": "string"},
                        "ownership": {"type": "string"},
                        "monthly_bill_range": {"type": "string"},
                        "major_loads": {"type": "array", "items": {"type": "string"}},
                        "location": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "schedule_specialist",
                "description": (
                    "Create a specialist call request (may not instantly lock the slot). "
                    "Return status=requested and a request_id. " + common_note
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "time": {"type": "string"},
                        "timezone": {"type": "string"},
                        "contact_method": {"type": "string"},
                        "contact_value": {"type": "string"},
                        "notes": {"type": "string"},
                    },
                    "required": ["date", "time", "timezone"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "send_confirmation",
                "description": (
                    "Send an SMS/email confirmation of the recap. If not configured, return ok=false. "
                    + common_note
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "contact_method": {"type": "string"},
                        "contact_value": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                    "required": ["contact_method", "contact_value", "summary"],
                    "additionalProperties": False,
                },
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        try:
            if tool_name == "create_lead":
                return self._create_lead(args, started=started)
            if tool_name == "lookup_service_area":
                return self._lookup_service_area(args, started=started)
            if tool_name == "get_estimate":
                return self._get_estimate(args, started=started)
            if tool_name == "schedule_specialist":
                return self._schedule_specialist(args, started=started)
            if tool_name == "send_confirmation":
                return self._send_confirmation(args, started=started)
            return {"ok": False, "error": f"unknown_tool:{tool_name}", "lead_id": self.lead_id}
        except Exception as e:
            logger.exception("Realtime tool execution failed", tool=tool_name)
            return {"ok": False, "error": str(e), "lead_id": self.lead_id}

    def _create_lead(self, args: dict[str, Any], *, started: float) -> dict[str, Any]:
        self.memory.update_from(args)
        notes = (args.get("notes") or "").strip()
        logger.info(
            "Lead updated",
            lead_id=self.lead_id,
            call_sid=self.call_sid,
            fields=list(args.keys()),
            notes_len=len(notes),
            ms=int((time.time() - started) * 1000),
        )
        return {"ok": True, "lead_id": self.lead_id, "memory": self.memory.to_public_dict()}

    def _lookup_service_area(self, args: dict[str, Any], *, started: float) -> dict[str, Any]:
        location = str(args.get("location") or "").strip()
        if not location:
            return {"ok": False, "lead_id": self.lead_id, "error": "missing_location"}
        # Placeholder: assume supported unless the caller explicitly says otherwise.
        supported: Optional[bool] = True
        if "outside" in location.lower() or "out of area" in location.lower():
            supported = None
        return {
            "ok": True,
            "lead_id": self.lead_id,
            "location": location,
            "supported": supported,
            "ms": int((time.time() - started) * 1000),
        }

    def _get_estimate(self, args: dict[str, Any], *, started: float) -> dict[str, Any]:
        # Intentionally conservative: no numeric promises here.
        return {
            "ok": True,
            "lead_id": self.lead_id,
            "estimate_type": "non_binding",
            "notes": "Provide a high-level fit and suggest the next step (eligibility check or specialist).",
            "inputs": {k: args.get(k) for k in ("primary_goal", "home_type", "ownership", "monthly_bill_range", "location")},
            "ms": int((time.time() - started) * 1000),
        }

    def _schedule_specialist(self, args: dict[str, Any], *, started: float) -> dict[str, Any]:
        date = str(args.get("date") or "").strip()
        time_str = str(args.get("time") or "").strip()
        tz = str(args.get("timezone") or "").strip() or self.memory.timezone
        if not date or not time_str:
            return {"ok": False, "lead_id": self.lead_id, "error": "missing_date_or_time"}

        request_id = f"req_{uuid.uuid4().hex[:10]}"
        self.memory.specialist_meeting.requested = True
        self.memory.specialist_meeting.date = date
        self.memory.specialist_meeting.time = time_str
        self.memory.specialist_meeting.timezone = tz
        self.memory.specialist_meeting.channel = (args.get("contact_method") or None) and str(
            args.get("contact_method")
        ).strip()
        self.memory.specialist_meeting.request_id = request_id
        self.memory.specialist_meeting.notes = (args.get("notes") or None) and str(args.get("notes")).strip()

        logger.info(
            "Specialist scheduled (request)",
            lead_id=self.lead_id,
            call_sid=self.call_sid,
            request_id=request_id,
            date=date,
            time=time_str,
            timezone=tz,
            ms=int((time.time() - started) * 1000),
        )
        return {
            "ok": True,
            "lead_id": self.lead_id,
            "status": "requested",
            "request_id": request_id,
            "date": date,
            "time": time_str,
            "timezone": tz,
        }

    def _send_confirmation(self, args: dict[str, Any], *, started: float) -> dict[str, Any]:
        # Placeholder: we don't have SMS/email configured in this repo.
        contact_method = str(args.get("contact_method") or "").strip().lower()
        contact_value = str(args.get("contact_value") or "").strip()
        if not contact_method or not contact_value:
            return {"ok": False, "lead_id": self.lead_id, "error": "missing_contact"}

        return {
            "ok": False,
            "lead_id": self.lead_id,
            "error": "not_configured",
            "notes": "No outbound confirmation channel configured; capture consent and have a human follow up.",
            "ms": int((time.time() - started) * 1000),
        }


def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return json.dumps({"ok": False, "error": "json_encode_failed"})

