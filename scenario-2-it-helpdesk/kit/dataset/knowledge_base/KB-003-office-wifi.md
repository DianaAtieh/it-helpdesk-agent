---
id: KB-003
title: "Office Wi-Fi: networks and troubleshooting"
version: "1.6"
last_updated: 2026-05-20
status: current
authoritative: true
owner: Network Operations
supersedes: null
applies_to: [all]
tags: [wifi, wireless, nakhla-corp, nakhla-guest, nakhla-iot, office, visitors]
error_codes: [WLAN-17]
---

# KB-003 — Office Wi-Fi: networks and troubleshooting

Covers Jerusalem HQ (floors 1-4) and the Ramallah office (floors 1-2).

## The three networks

| Network (SSID) | For | How you connect | Access |
|---|---|---|---|
| **Nakhla-Corp** | Company laptops enrolled in NakhlaManage | Automatic, with the device certificate. You never type a Wi-Fi password. | Internal systems and internet |
| **Nakhla-Guest** | Visitors, personal phones and personal laptops | Daily access code from reception, or the host creates one at `https://guest.nakhla.example`. A code is valid for **8 hours**. | Internet only |
| **Nakhla-IoT** | Printers and meeting-room devices | Hidden network, managed by Network Operations | Never connect laptops or phones |

There is no shared Wi-Fi password for staff. If someone gives you one, it is not a Nakhla network.

## Nakhla-Corp does not connect

1. Make sure Wi-Fi is on and you are in a Nakhla office.
2. Forget Nakhla-Corp and join it again.
3. Open NakhlaManage > Sync, wait 2 minutes, then restart the laptop. This renews the device certificate.
4. Error **`WLAN-17`** ("authentication rejected") means the network did not accept your device certificate. The laptop is usually not compliant or not fully enrolled. Fix the compliance problem shown in NakhlaManage (KB-010), sync, then retry.
5. As a temporary workaround you can use a wired dock port or Nakhla-Guest for internet-only work.

## Escalation

- One person cannot connect after these steps: ticket category `network_access`, priority **P3** if a workaround works (wired port), **P2** if you cannot work at all.
- Several people in the same office lose Wi-Fi at the same time: this can be a major incident. Use the P1 threshold in KB-015 and say which office and floor.

## Visitors

Visitors never get Nakhla-Corp. The host requests a Nakhla-Guest code at reception or on the guest portal. Codes cannot be extended beyond 8 hours; create a new one the next day.

## Change log

- 1.6 (2026-05-20): guest codes valid for 8 hours (was 12).
- 1.4 (2025-09-01): `WLAN-17` added.
- 1.0 (2025-01-20): first version.
