---
id: KB-006
title: "Account lockout (NID-423)"
version: "1.3"
last_updated: 2026-04-08
status: current
authoritative: true
owner: Identity & Access Management
supersedes: null
applies_to: [all]
tags: [lockout, locked-account, nakhlaid, password, security]
error_codes: [NID-423]
---

# KB-006 — Account lockout (NID-423)

## Why it happens

Your NakhlaID account locks after **5 failed sign-ins within 15 minutes**. Sign-in pages then show `NID-423` ("account locked").

## Unlock it

You have two options:

1. **Wait 30 minutes.** The lock is released automatically.
2. **Unlock now** at `https://id.nakhla.example/unlock`: enter your work email and approve the NakhlaKey prompt (or use a backup code). This does not change your password.

If you also forgot the password, reset it instead (KB-004). A reset unlocks the account too.

## It keeps locking

Repeated lockouts almost always come from a device that still tries an **old password**:

- the mail app on your phone (remove the account and add it again through the NakhlaManage work profile, KB-014);
- a mapped network drive or a saved credential on another laptop;
- an old NakhlaConnect profile that saved your password.

Update or remove the old password everywhere, then unlock.

## Escalation

- Locked three or more times in one day and you cannot find an old password anywhere: this may be someone guessing your password. Report it to Security Operations: ticket category `security`, priority **P2** (SP-001).
- Do not ask the Service Desk to unlock or reset you over the phone. They cannot (SP-001).

## Change log

- 1.3 (2026-04-08): repeated lockouts are reported as `security`, P2.
- 1.0 (2025-02-10): first version.
