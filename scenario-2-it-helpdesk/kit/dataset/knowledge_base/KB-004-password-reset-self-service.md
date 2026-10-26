---
id: KB-004
title: Reset your NakhlaID password (self-service)
version: "1.2"
last_updated: 2026-08-20
status: current
authoritative: true
owner: Identity & Access Management
supersedes: KB-005
applies_to: [all]
tags: [password, password-reset, nakhlaid, self-service, expired-password]
error_codes: [NID-401, NID-460]
---

# KB-004 — Reset your NakhlaID password (self-service)

NakhlaID is the single sign-on account for email, the VPN, Software Center and every business app.
Password resets are **self-service only** (SP-001). The Service Desk cannot reset your password over the phone, chat or email.

## Reset it yourself

1. Go to `https://id.nakhla.example/reset` (it works from any browser, including Nakhla-Guest Wi-Fi or your phone).
2. Enter your work email (`name.surname@nakhla.example`).
3. Approve the NakhlaKey prompt, or enter a 6-digit NakhlaKey code, or one of your backup codes.
4. Choose a new password: at least 14 characters, not containing "Nakhla", not one of your last 10.
5. Sign out and back in on your laptop. Update the password on your phone's mail app (KB-014) so it does not lock you out (KB-006).

## Error codes at sign-in

| Code | Meaning | What to do |
|---|---|---|
| `NID-401` | Wrong username or password | Check Caps Lock and the email address. After 5 failures the account locks (`NID-423`, KB-006). Forgot it? Reset it above. |
| `NID-460` | Password expired (passwords expire every 180 days) | Reset it at `https://id.nakhla.example/reset`. |

## When self-service is not possible

If you cannot approve the reset (no NakhlaKey device and no backup codes):

1. The Service Desk opens a ticket: category `identity_access`, priority **P2** if you cannot work.
2. Your **manager must approve** the reset in the ticket.
3. You verify your identity **in person** at an IT Service Desk counter or by **video call** with your badge visible.
4. Identity & Access Management sends a one-time reset link to your manager's email. Nobody reads a password to you.

## Change log

- 1.2 (2026-08-20): reset link now goes to the manager after an assisted reset.
- 1.0 (2026-02-01): first version; replaces the phone procedure KB-005.
