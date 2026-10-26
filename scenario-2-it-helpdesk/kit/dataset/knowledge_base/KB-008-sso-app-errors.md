---
id: KB-008
title: "Single sign-on (NakhlaID) errors in business apps"
version: "1.1"
last_updated: 2026-01-15
status: current
authoritative: true
owner: Identity & Access Management
supersedes: null
applies_to: [all]
tags: [sso, single-sign-on, nakhlaid, apps, nakhlacrm, nakhlapeople, nakhlaledger, nakhlacode]
error_codes: [NID-512, NID-530]
---

# KB-008 — Single sign-on (NakhlaID) errors in business apps

Every Nakhla business app signs you in with NakhlaID (single sign-on):

| App | What it is | Who has it by default |
|---|---|---|
| NakhlaCRM | customer relationship management | Sales (Support: read-only on request) |
| NakhlaPeople | HR system | HR (everyone sees their own profile) |
| NakhlaLedger | finance and accounting | Finance |
| NakhlaCode | source code hosting | Engineering |
| Nakhla Wiki | company wiki | everyone |

## Error codes

| Code | Meaning | What to do |
|---|---|---|
| `NID-512` | "This app is not assigned to you" | You do not have access to this app yet. Request it at `https://access.nakhla.example` (KB-011). Clearing the browser will not help. |
| `NID-530` | "Sign-in time mismatch" | Your laptop clock is wrong by more than 5 minutes. Set date and time to automatic, then retry. |

## The app keeps sending me back to the sign-in page

1. Open the app in a private (incognito) window. If it works there, clear cookies for `id.nakhla.example` in your normal browser.
2. Make sure you sign in with your work email, not a personal account.
3. Password expired or wrong? See KB-004. Account locked? See KB-006.

## Escalation

If the app still fails after these steps and you have access to it, open a ticket: category `identity_access`, priority **P3** (P2 if you cannot work at all). Include the app name, the error code and a screenshot.

## Change log

- 1.1 (2026-01-15): `NID-530` added.
- 1.0 (2025-03-01): first version.
