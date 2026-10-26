---
id: KB-007
title: "NakhlaKey MFA: enrol, new phone, push problems"
version: "2.0"
last_updated: 2026-03-01
status: current
authoritative: true
owner: Identity & Access Management
supersedes: null
applies_to: [all]
tags: [mfa, nakhlakey, authenticator, new-phone, backup-codes, push]
error_codes: [MFA-201, MFA-310]
---

# KB-007 — NakhlaKey MFA: enrol, new phone, push problems

NakhlaKey is the Nakhla authenticator app. It approves sign-ins with a push prompt or a 6-digit code.
MFA is mandatory for everyone, Contractors included (SP-002).

## Enrol (first time)

1. Install NakhlaKey from your phone's app store (publisher: Nakhla Systems).
2. On a laptop, open `https://id.nakhla.example/mfa` and sign in.
3. Scan the QR code with NakhlaKey.
4. Save your **10 backup codes** offline. Each code works once.

## Push prompt not received (`MFA-201`)

1. Open NakhlaKey and pull down to refresh. The prompt expires after **60 seconds**.
2. Check the phone has internet and the date and time are set automatically.
3. Allow NakhlaKey notifications and exclude it from battery optimisation.
4. While you fix it, choose "Enter a code instead" and type the 6-digit code from the app.

## New phone (`MFA-310` "device not registered")

- **You still have the old phone:** in NakhlaKey on the old phone, open Settings > Transfer and scan the code shown on the new phone. Done.
- **The old phone is gone but you have backup codes:** sign in at `https://id.nakhla.example/mfa` with a backup code, remove the old device, and enrol the new phone.
- **No old phone and no backup codes:** the Service Desk opens a ticket (category `identity_access`). Your manager approves it and you verify your identity in person or by video call (SP-001). Then Identity & Access Management lets you enrol again.
- **Lost or stolen phone:** report it first as a security incident (SP-002): category `security`, priority **P1**.

## Forgot the phone at home

Use one of your backup codes. SMS codes do not exist any more (removed on 2026-03-01, SP-002), so do not ask for one.

## Change log

- 2.0 (2026-03-01): SMS fallback removed; transfer steps for a new phone added.
- 1.0 (2025-02-03): first version.
