"""Fixed answers for application-access questions, decided in code from KB-011 and KB-008 (no model call).

The model used to mix the two documents: KB-008 says who has an app by default, KB-011 says which group gives it,
who may be a member and who approves. It sometimes ignored eligibility ("request FIN-Ledger" to an Engineer) or the
group and approver. Here both tables are copied word for word (tests/test_access_rules.py compares them with the KB
files, so a KB change cannot silently leave this stale) and the answer is built from them and the directory record.

  * Only a question that names ONE known app or group AND asks for access ("access", "add me", "join", "not
    assigned", NID-512 ...) is answered here; any other error code means troubleshooting and is left to the model.
  * Department and contractor status come from the directory record in the state (triage), never from the message.
    Without a directory record nothing is answered here (the normal "who are you" clarification applies).
  * Out of scope on purpose: VPN-Users (it is in the table, but general VPN questions are not intercepted), shared
    drives and mailboxes (no fixed group), and ticket requests (they never reach the answer node).
  * Contractors: KB-011 allows ENG-Code only for "Contractors sponsored by Engineering". The directory has no sponsor,
    so the condition is stated, never inferred from the contractor's manager.

The answer node calls access_answer() right after the security-policy refusal; its citations are issued by code and
checked by valid_citations() in the respond node.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

OUTCOME = "access_info"
RULE_SOURCE = "access_rule"
PORTAL = "https://access.nakhla.example"
CONTRACTORS = "Contractors"
TITLES = {"KB-008": "Single sign-on (NakhlaID) errors in business apps",
          "KB-011": "Access permissions and access requests"}


@dataclass(frozen=True)
class Group:
    """One row of KB-011's "Standard groups" table (kb_* fields word for word) and what code reads from it."""
    name: str
    kb_gives: str                     # "Gives access to"
    kb_members: str                   # "Who can be a member"
    kb_approvers: str                 # "Approvers"
    members: tuple[str, ...]          # departments that may be members
    read_only: tuple[str, ...] = ()   # departments that may have read-only membership
    second_approver: str | None = None  # who approves after the manager (KB-011: "manager, then the data owner")
    app: str | None = None            # the KB-008 app this group opens
    sponsored_contractors: bool = False  # "Contractors sponsored by Engineering"


GROUPS: tuple[Group, ...] = (
    Group("VPN-Users", "NakhlaConnect VPN", "all employees, never Contractors", "manager", members=()),
    Group("ENG-Code", "NakhlaCode repositories", "Engineering, Contractors sponsored by Engineering",
          "manager + repository owner", members=("Engineering",), second_approver="repository owner",
          app="NakhlaCode", sponsored_contractors=True),
    Group("SALES-CRM", "NakhlaCRM", "Sales (Support: read-only)", "Sales Director", members=("Sales",),
          read_only=("Support",), second_approver="Sales Director", app="NakhlaCRM"),
    Group("FIN-Ledger", "NakhlaLedger", "Finance", "Finance Manager", members=("Finance",),
          second_approver="Finance Manager", app="NakhlaLedger"),
    Group("HR-People", "NakhlaPeople (HR data)", "HR", "HR Director", members=("HR",), second_approver="HR Director",
          app="NakhlaPeople"),
    Group("EXEC-Board", "board documents", "Executive Office", "Chief Executive Officer",
          members=("Executive Office",), second_approver="Chief Executive Officer"),
)
BY_NAME = {group.name: group for group in GROUPS}

# KB-008 "Every Nakhla business app ..." table: app -> (what it is, who has it by default), word for word,
# and the departments that have it by default.
KB008_APPS: dict[str, tuple[str, str]] = {
    "NakhlaCRM": ("customer relationship management", "Sales (Support: read-only on request)"),
    "NakhlaPeople": ("HR system", "HR (everyone sees their own profile)"),
    "NakhlaLedger": ("finance and accounting", "Finance"),
    "NakhlaCode": ("source code hosting", "Engineering"),
    "Nakhla Wiki": ("company wiki", "everyone"),
}
DEFAULT_DEPARTMENTS = {"NakhlaCRM": ("Sales",), "NakhlaPeople": ("HR",), "NakhlaLedger": ("Finance",),
                       "NakhlaCode": ("Engineering",)}

# Names that point to one group. "NakhlaCode Studio" is a Software Center app (KB-009), not the repositories.
_ALIASES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"(?i)\bfin[- ]?ledger\b|\bnakhla\s*ledger\b"), "FIN-Ledger"),
    (re.compile(r"(?i)\beng[- ]?code\b|\bnakhla\s*code\b(?!\s*studio)"), "ENG-Code"),
    (re.compile(r"(?i)\bsales[- ]?crm\b|\bnakhla\s*crm\b"), "SALES-CRM"),
    (re.compile(r"(?i)\bhr[- ]?people\b|\bnakhla\s*people\b"), "HR-People"),
    (re.compile(r"(?i)\bexec[- ]?board\b|\bboard\s+(?:documents?|docs|papers|materials?)\b"), "EXEC-Board"),
)
_ACCESS = re.compile(r"(?i)\b(?:access|permissions?|add\s+me|added\s+to|join|membership|grant|not\s+assigned|"
                     r"assigned\s+to\s+me|NID-512)\b")
_ERROR_CODE = re.compile(r"\b[A-Z]{2,5}-\d{2,5}\b")


@dataclass(frozen=True)
class AccessAnswer:
    reply: str
    citations: list[dict]
    group: str


def requested_group(text: str) -> Group | None:
    """The one group an access question is about, or None (not about access, no known group, several groups,
    or another error code, which means troubleshooting)."""
    text = text or ""
    if not _ACCESS.search(text):
        return None
    if any(code != "NID-512" for code in _ERROR_CODE.findall(text)):
        return None
    names = {name for pattern, name in _ALIASES if pattern.search(text)}
    return BY_NAME[next(iter(names))] if len(names) == 1 else None


def _citation(doc_id: str) -> dict:
    return {"id": doc_id, "title": TITLES[doc_id], "source": RULE_SOURCE}


def _request_steps(group: Group) -> str:
    approvers = (f"Your manager approves first, then the {group.second_approver}" if group.second_approver
                 else "Your manager approves it")
    return (f"Request the {group.name} group at {PORTAL} > New request, and write why you need it and for how long. "
            f"{approvers}. Most requests finish within 2 business days; sign out and in again after approval. "
            "If you had this access before and lost it, say so in the request.")


def _only_for(group: Group) -> str:
    who = " and ".join(group.members)
    extra = f" ({', '.join(group.read_only)}: read-only)" if group.read_only else ""
    return f"{group.name} ({group.kb_gives}) is only for {who} staff{extra}"


def access_answer(text: str, employee: dict | None) -> AccessAnswer | None:
    """The fixed answer to an access question for this directory record, or None (left to the normal flow)."""
    group = requested_group(text)
    department = employee.get("department") if isinstance(employee, dict) else None
    if group is None or not department:
        return None
    contractor = bool(employee.get("contractor")) or department == CONTRACTORS
    cite = ["KB-011"]

    if contractor and group.sponsored_contractors:
        reply = (f"Contractors can join {group.name} ({group.kb_gives}) only when their contract is sponsored by "
                 "Engineering. If Engineering sponsors you, ask your sponsoring manager to support the request. "
                 f"{_request_steps(group)} Contractor access lasts at most 90 days at a time; your sponsoring "
                 "manager renews it.")
    elif contractor:
        reply = f"{_only_for(group)}, so it is not a group contractors can be added to."
    elif department in group.members:
        if group.app and department in DEFAULT_DEPARTMENTS.get(group.app, ()):
            reply = (f"{department} staff normally have {group.app} by default. If it is missing, for example "
                     f"NakhlaID shows NID-512 (\"This app is not assigned to you\"): {_request_steps(group)}")
            cite.insert(0, "KB-008")
        else:
            reply = f"{group.kb_gives[:1].upper()}{group.kb_gives[1:]} access comes from the {group.name} group. " \
                    f"{_request_steps(group)}"
    elif department in group.read_only:
        reply = (f"{department} staff can have {group.app or group.kb_gives} read-only, on request. "
                 f"{_request_steps(group)} Ask for read-only access in the request.")
        if group.app:
            cite.insert(0, "KB-008")
    else:
        reply = (f"{_only_for(group)}, so {department} staff can't be added to it, and a request would not be "
                 "approved.")
        if group.app == "NakhlaPeople":
            reply += " Everyone can see their own profile in NakhlaPeople."
            cite.insert(0, "KB-008")
    return AccessAnswer(reply, [_citation(doc_id) for doc_id in cite], group.name)


def valid_citations(raw) -> list[dict]:
    """Code-issued access citations from this turn, kept only if they name KB-008 or KB-011 with this source."""
    out = [_citation(c["id"]) for c in raw or []
           if isinstance(c, dict) and c.get("id") in TITLES and c.get("source") == RULE_SOURCE]
    return list({c["id"]: c for c in out}.values())
