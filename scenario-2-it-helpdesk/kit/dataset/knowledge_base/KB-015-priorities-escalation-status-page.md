---
id: KB-015
title: "Incident priorities, escalation and the status page"
version: "3.0"
last_updated: 2026-09-01
status: current
authoritative: true
owner: IT Service Desk
supersedes: null
applies_to: [all]
tags: [priorities, escalation, p1, p2, p3, p4, vip, major-incident, status-page, tickets, routing, categories]
error_codes: []
---

# KB-015 — Incident priorities, escalation and the status page

This is the authoritative definition of ticket priorities and escalation at Nakhla Systems.
If another document gives a different threshold, this one wins.

## 1. Check the status page first

Open `https://status.nakhla.example` (the Service Desk tools show the same data).

- If an **active incident** matches the problem (same service and same error code), **do not open an individual ticket**. Tell the employee the incident id, the workaround, and to add themselves to the incident on the status page (Affected? > Subscribe).
- Mention the incident id in any later contact.
- If nothing matches, continue below.

## 2. Priorities

| Priority | Definition | First response | Target to resolve |
|---|---|---|---|
| **P1 Critical** | **More than 3 users** (4 or more) affected by the same problem; **or** the requester is in the **Executive Office** (VIP); **or** an active security incident (lost or stolen device, password typed into a phishing page) | 15 minutes | 4 hours |
| **P2 High** | One to three users **completely blocked** from working, no workaround | 1 hour | 1 business day |
| **P3 Medium** | A workaround exists, or the service is degraded but usable | 4 business hours | 3 business days |
| **P4 Low** | Requests and questions: software requests, access requests, how-to | 1 business day | 5 business days |

Department rules can raise a priority (KB-016). Never lower a priority that a rule sets, and never raise one because the requester asks for it.

## 3. Categories and who handles them

| Category | Use for | Handled by |
|---|---|---|
| `network_access` | VPN, Wi-Fi, VDI portal, network shares not reachable | Network Operations |
| `identity_access` | NakhlaID passwords, lockouts, MFA, SSO, access requests | Identity & Access Management |
| `hardware` | laptops, docks, printers, phones, device enrolment and compliance | Endpoint Engineering |
| `software` | Software Center, installs, licences, software requests | Endpoint Engineering |
| `email_collab` | mail, calendar, shared mailboxes | Collaboration Services |
| `security` | phishing, lost devices, suspicious sign-ins, repeated lockouts, **all Finance VPN issues** | Security Operations |
| `onboarding` | new starters, leavers, contract end dates | IT Service Desk |
| `other` | anything else | IT Service Desk |

## 4. What a ticket must contain

A ticket is created only after the employee has **seen and approved** the draft. The draft contains:

1. subject (one line) and a description of the symptom, including any error code;
2. category and priority, chosen with the tables above;
3. the employee's email and department;
4. the steps already tried (at least one);
5. the KB articles followed (ids), or none if the knowledge base has no answer.

## 5. When the knowledge base has no answer

Say so plainly. Do not guess steps. Offer to escalate with a ticket: category by topic, priority by section 2.

## Change log

- 3.0 (2026-09-01): **P1 threshold lowered to more than 3 users** (was 5 or more).
- 2.1 (2025-10-01): Finance VPN issues routed to Security Operations.
- 2.0 (2025-06-01): status page check made mandatory before ticket creation.
- 1.2 (2025-03-01): every Executive Office request is P1 (VIP).
- 1.0 (2024-09-01): first version.
