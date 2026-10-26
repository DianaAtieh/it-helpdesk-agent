---
id: KB-009
title: "Software Center: approved catalogue and software requests"
version: "3.0"
last_updated: 2026-07-01
status: current
authoritative: true
owner: Endpoint Engineering
supersedes: null
applies_to: [all]
tags: [software, software-center, install, catalogue, software-request, elevate, licences, admin-rights]
error_codes: [SWC-0404, SWC-1603, SWC-2001]
---

# KB-009 — Software Center: approved catalogue and software requests

Software Center (`https://software.nakhla.example`, also an app on every company laptop) is the only way to install software.
You do not need administrator rights for anything in the catalogue.

## Approved catalogue (extract)

| App | Version | Who |
|---|---|---|
| NakhlaConnect (VPN) | 5.4.2 | everyone except Contractors |
| NakhlaPrint driver | 3.1 | everyone |
| Nakhla Dock D3 firmware | 2.4 | everyone with a D3 dock |
| Visual Studio Code | latest | everyone |
| Python | 3.12 | Engineering, Support |
| Docker Desktop | latest | Engineering |
| NakhlaCode Studio (licensed IDE) | 2026.2 | Engineering (licence pool) |
| Postman | latest | Engineering, Support |
| PDF reader | latest | everyone |
| 7-Zip | latest | everyone |

## Install from the catalogue

1. Open Software Center, search for the app, click **Install**.
2. Keep the laptop on power and on Nakhla-Corp Wi-Fi or the VPN until it finishes.

| Error | Meaning | What to do |
|---|---|---|
| `SWC-1603` | Install failed | Restart the laptop (a pending update often blocks installs), make sure at least **10 GB** is free, then try again. |
| `SWC-2001` | Licence pool exhausted | Nothing to fix on your side. Open a ticket: category `software`, priority **P3**. Endpoint Engineering adds licences. |
| `SWC-0404` | App not available for your device or department | Ask your manager whether an approved alternative exists, or submit a software request (below). |

## Software that is not in the catalogue

1. Submit a **software request**: ticket category `software`, priority **P4**.
2. Include the app name, the version, the business reason and your manager's name.
3. Your manager approves, then Security Operations reviews it. Target: **5 business days**.
4. If it is approved, Endpoint Engineering publishes it in Software Center for you.

## Administrator rights

- Nobody has permanent local admin rights (SP-002).
- **Engineering only:** Software Center > **Elevate** gives local admin rights for **30 minutes**, for tools that need admin to install. Every session is logged.
- Contractors cannot install software. Their tools are provided in the VDI portal.

## Change log

- 3.0 (2026-07-01): `SWC-0404` added; catalogue extract updated.
- 2.1 (2026-06-01): NakhlaConnect 5.4.2 published (minimum client is now 5.4, KB-002).
- 2.0 (2025-11-20): Elevate for Engineering replaces admin-rights requests.
- 1.0 (2025-01-08): first version.
