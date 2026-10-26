---
id: KB-010
title: "Laptops and devices: NakhlaManage enrolment, compliance and repairs"
version: "2.2"
last_updated: 2026-06-15
status: current
authoritative: true
owner: Endpoint Engineering
supersedes: null
applies_to: [all]
tags: [devices, laptops, mdm, nakhlamanage, enrolment, compliance, certificates, hardware, docks, loaner]
error_codes: [MDM-12, MDM-31, MDM-44]
---

# KB-010 — Laptops and devices: NakhlaManage enrolment, compliance and repairs

NakhlaManage is the device management service. It keeps company laptops compliant and gives them the
**device certificate** that the Nakhla-Corp Wi-Fi (KB-003) and the NakhlaConnect VPN (KB-002) need.

## Enrolment

- Company laptops are enrolled by Endpoint Engineering before handover. Check: the NakhlaManage app shows "Compliant".
- Personal phones may enrol a **work profile** for mail and NakhlaKey only. Personal laptops are never enrolled (SP-002).
- Each person can enrol at most **3 devices**. A fourth shows `MDM-31`: remove an old device in NakhlaManage > My devices.
- Contractors' own devices are not enrolled. Contractors use the VDI portal.

## Compliance rules

| Rule | Error if broken | Fix |
|---|---|---|
| Disk encryption on (BitLocker or FileVault) | `MDM-12` | Turn encryption back on from NakhlaManage > Fix, keep the laptop on power until it finishes. |
| Operating system at or above the minimum: Windows 11 23H2, macOS 14, Ubuntu 22.04 LTS (Ubuntu: Engineering only) | `MDM-44` | Install the pending OS update; Ubuntu users upgrade to 22.04 LTS. |
| Screen lock after 10 minutes or less | shown as a warning | Set it in system settings. |

A laptop that stays non-compliant for 24 hours loses its device certificate. The VPN then shows `NX-112` and the Wi-Fi shows `WLAN-17`.
After you fix the problem, open NakhlaManage > **Sync** and restart.

A laptop that was offline for more than 30 days must sync once (on Nakhla-Guest or any home network) before the VPN accepts it again.

## Hardware problems

- **Nakhla Dock D3** (USB-C dock): if monitors or USB devices are not detected, update the dock firmware from Software Center ("Nakhla Dock D3 firmware 2.4"), reconnect the cable, restart.
- **Broken or damaged laptop:** a loaner laptop is available at the IT Service Desk counter (Jerusalem HQ floor 2, Ramallah office floor 1). Ticket category `hardware`, priority **P3**, or **P2** if you cannot work at all and cannot collect a loaner.
- **Swollen battery, smoke or burning smell:** stop using the device, unplug it, bring it to the Service Desk counter.
- **Lost or stolen laptop:** security incident, category `security`, priority **P1** (SP-002).

## Change log

- 2.2 (2026-06-15): offline-for-30-days sync rule added.
- 2.0 (2025-12-01): Ubuntu 22.04 LTS allowed for Engineering.
- 1.0 (2025-01-15): first version.
