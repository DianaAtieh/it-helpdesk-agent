---
id: KB-001
title: "NakhlaConnect VPN: install and connect (legacy 4.x guide)"
version: "1.4"
last_updated: 2025-04-10
status: deprecated
authoritative: false
owner: Network Operations
supersedes: null
applies_to: [all]
tags: [vpn, nakhlaconnect, install, legacy]
error_codes: [NX-403, NX-781]
---

# KB-001 — NakhlaConnect VPN: install and connect (legacy 4.x guide)

> **DEPRECATED.** This guide describes the NakhlaConnect 4.x era. It was replaced by
> KB-002 on 2026-06-01. Do not follow it. Kept for reference only.

## Supported versions

- Minimum supported client: **NakhlaConnect 4.8**.
- Recommended client: **4.9.1**.
- Older clients (4.7 and below) show `NX-781` when they connect.

## Install

1. Download the installer from the IT share `\\files.nakhla.example\IT\NakhlaConnect\4.9.1\`.
2. Run the installer and accept the defaults.
3. Open NakhlaConnect and enter the gateway address `vpn.nakhla.example`.
4. Sign in with your NakhlaID email and password and approve the NakhlaKey prompt.

## Common problems

| Error | What to do |
|---|---|
| `NX-781` client version not supported | Reinstall NakhlaConnect 4.8 or later from the IT share. Any 4.8+ version is fine. |
| `NX-403` access denied | Call the Service Desk on ext. 4357 and ask to be added to the VPN group. |
| Slow connection | Disconnect and reconnect. |

## Escalation

If the VPN still does not work, call the Service Desk on ext. 4357.

## Change log

- 1.4 (2025-04-10): recommended version 4.9.1.
- 1.0 (2024-06-02): first version.
