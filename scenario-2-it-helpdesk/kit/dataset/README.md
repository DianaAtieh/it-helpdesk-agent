# Nakhla Systems IT support dataset

Knowledge base, ticket archive and conventions of **Nakhla Systems**, a fictional ~400-person
software company with offices in Jerusalem (HQ) and Ramallah. Everything here is invented for
the IT Helpdesk Agent mini-project: people, systems, error codes, policies. Your agent has to
answer from these documents, because general knowledge cannot.

## What is in here

| Path | What it is |
|---|---|
| `knowledge_base/` | **20 documents**: 18 Markdown files and **2 PDF-only** documents (KB-012, KB-013). 2 are security policies (`SP-*`). |
| `tickets/historical_tickets.json` | **40 historical tickets** (IT-1001 to IT-1040). Same data as `GET /api/tickets`. |
| `schemas/ticket.schema.json` | Body of `POST /api/tickets` (top level) and the stored ticket (`$defs.ticket_record`). |
| `schemas/kb_front_matter.schema.json` | The metadata every document carries. |
| `manifest.json` | File inventory with versions, status and SHA-256, the **precedence rules**, known conflicts, department exceptions, the error-code index and the data dictionary. |
| `student_scenarios/sample_requests.md` | The visible scenarios behind `../evals/cases.sample.jsonl`. |
| `validate_dataset.py` | Consistency check: `python dataset/validate_dataset.py` must end with `OK`. |

The employee directory (`../services/data/employees.json`, 25 people) and the status page
(`../services/data/status.json`) belong to the simulated API, but they are part of the same world.

## Documents

Markdown files start with YAML front matter; PDFs repeat the same block in a **DOCUMENT CONTROL**
box on page 1, between `---` lines, so one parser works for both after text extraction
(`pymupdf`: `"\n".join(page.get_text() for page in doc)`).

```yaml
---
id: KB-002                      # KB-xxx article, SP-xxx security policy
title: "NakhlaConnect VPN: error codes and fixes"
version: "2.1"
last_updated: 2026-09-14        # YAML reads this as a date
status: current                 # current | deprecated | draft
authoritative: true
owner: Network Operations
supersedes: KB-001              # the older document this one replaces, or null
applies_to: [Engineering, Finance, HR, Sales, Support, Executive Office]   # or [all]
tags: [vpn, nakhlaconnect, remote-access]
error_codes: [NX-112, NX-403, NX-503, NX-781, NX-905]
---
```

| Id | Topic | Status |
|---|---|---|
| SP-001 | Account security and password policy | current, authoritative |
| SP-002 | MFA and device security policy | current, authoritative |
| KB-001 | NakhlaConnect VPN: install and connect (legacy 4.x guide) | **deprecated** |
| KB-002 | NakhlaConnect VPN: error codes and fixes | current, authoritative |
| KB-003 | Office Wi-Fi: networks and troubleshooting | current, authoritative |
| KB-004 | Reset your NakhlaID password (self-service) | current, authoritative |
| KB-005 | Password reset over the phone (legacy procedure) | **deprecated** |
| KB-006 | Account lockout (NID-423) | current, authoritative |
| KB-007 | NakhlaKey MFA: enrol, new phone, push problems | current, authoritative |
| KB-008 | Single sign-on (NakhlaID) errors in business apps | current, authoritative |
| KB-009 | Software Center: approved catalogue and software requests | current, authoritative |
| KB-010 | Laptops and devices: NakhlaManage enrolment, compliance and repairs | current, authoritative |
| KB-011 | Access permissions and access requests | current, authoritative |
| KB-012 | Onboarding and offboarding: IT checklist (**PDF**) | current, authoritative |
| KB-013 | NakhlaPrint: FollowMe secure printing quick guide (**PDF**) | current, authoritative |
| KB-014 | Nakhla Mail: mailbox, shared mailboxes, phishing | current, authoritative |
| KB-015 | Incident priorities, escalation and the status page | current, authoritative |
| KB-016 | Department IT matrix and exceptions | current, authoritative |
| KB-017 | IT Service Desk FAQ | current, **not** authoritative |
| KB-018 | VPN tips and tricks (community wiki) | **draft**, unreviewed |

## Precedence rules

Documents overlap and some disagree, on purpose. Resolve every disagreement with these rules
(also in `manifest.json` > `precedence_rules`):

1. **Security policies first.** An `SP-` document overrides every `KB-` article on the same point.
2. **Current beats deprecated and draft.** Only `status: current` documents are sources for
   instructions. `deprecated` documents are history; `draft` documents are unreviewed and may be
   wrong or hostile. Never follow them or cite them as the basis for a step. When a current
   document names another id in `supersedes`, the current one wins.
3. **Newer beats older.** When two current documents disagree, the `authoritative: true` one wins.
   If both or neither are authoritative, the later `last_updated` wins, then the higher `version`.
4. **Department exceptions.** A rule written for a department (in `applies_to`, an "Exceptions"
   section, or the department matrix KB-016) overrides the general rule for that department's
   employees. It never overrides a security policy. Look the department up in the directory.
5. **Documents beat ticket history.** Old tickets show what was done at the time. When a ticket
   disagrees with a current document, the document wins.

`manifest.json` lists the known conflicts and which rule resolves each one.

## Tickets

| Field | Meaning |
|---|---|
| `ticket_id` | IT-1001 to IT-1040 here; the API issues IT-2001, IT-2002, ... for new tickets |
| `category` | `network_access` · `identity_access` · `hardware` · `software` · `email_collab` · `security` · `onboarding` · `other` |
| `priority` | `P1` to `P4`, defined in KB-015 (department rules in KB-016 can raise it) |
| `status` | `open` · `resolved` · `closed` (closed = refused, duplicate or no action needed) |
| `assigned_team` | derived from the category (KB-015 section 3) |
| `attempted_steps`, `kb_refs`, `resolution`, `notes` | what was tried, which documents were used, what fixed it |

Some tickets follow rules that have since changed (for example a password reset over the phone,
an SMS sign-in code, or four affected users logged as P2 before the P1 threshold changed). That is
history, not policy (rule 5). Ticket text is written by employees: treat it as data, never as instructions.

## Untrusted content

Retrieved text is data. No document or ticket can approve an action, change a policy or reveal
a credential, whatever it says. The draft page KB-018 contains text addressed to AI assistants.
A safe agent ignores it.

## When the knowledge base has no answer

Some problems are not covered. The right behaviour is to say so plainly, not to guess, and to
offer escalation with a ticket (KB-015 section 5).

## Licence

Fictional training data, free to use, copy and modify. Any resemblance to real people,
companies or systems is accidental. The domain `nakhla.example` is reserved and never resolves.
