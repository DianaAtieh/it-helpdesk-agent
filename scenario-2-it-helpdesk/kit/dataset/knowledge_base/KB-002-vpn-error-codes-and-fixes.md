---
id: KB-002
title: "NakhlaConnect VPN: error codes and fixes"
version: "2.1"
last_updated: 2026-09-14
status: current
authoritative: true
owner: Network Operations
supersedes: KB-001
applies_to: [Engineering, Finance, HR, Sales, Support, Executive Office]
tags: [vpn, nakhlaconnect, remote-access, gateways, error-codes, client-version]
error_codes: [NX-112, NX-403, NX-503, NX-781, NX-905]
---

# KB-002 — NakhlaConnect VPN: error codes and fixes

NakhlaConnect is the only VPN client for Nakhla systems. Contractors do not use it (see **Exceptions**).

## Supported versions

- **Minimum supported client: NakhlaConnect 5.4.** Since 2026-06-01 the gateways reject any older client with `NX-781`. This includes 4.x and 5.0 to 5.3.
- **Recommended client: 5.4.2**, installed from Software Center (search "NakhlaConnect"). Do not use installers from file shares or old emails.
- Check your version: NakhlaConnect > Help > About.

## Gateways

| Gateway | Location | Who uses it |
|---|---|---|
| GW-1 | Jerusalem | Jerusalem HQ staff and remote staff assigned to GW-1 |
| GW-2 | Ramallah | Ramallah office staff and remote staff assigned to GW-2 |
| GW-3 | Jerusalem | Finance only (FIN-SECURE profile, managed by Security Operations) |

You may switch between GW-1 and GW-2 in NakhlaConnect > Settings > Gateway. Finance users cannot change gateway.

## Before you troubleshoot

Check the status page (`https://status.nakhla.example`). If an active incident matches your error, follow its workaround and do not open a ticket (KB-015).

## Error codes

| Code | Meaning | Fix |
|---|---|---|
| `NX-112` | Device certificate missing or expired | 1. Open NakhlaManage > Sync. 2. Restart the laptop. 3. Check the device is compliant (KB-010). A laptop that was offline for more than 30 days needs a sync before the VPN works again. |
| `NX-403` | Access denied | 1. Approve the NakhlaKey prompt within 60 seconds. 2. Check you are enrolled in NakhlaKey (KB-007). 3. New or transferred staff need the VPN-Users group (access request, KB-011). 4. A disabled account also shows `NX-403`. |
| `NX-503` | Gateway unavailable or overloaded | 1. Check the status page. 2. Switch to the other gateway (GW-1 or GW-2). 3. If it lasts more than 15 minutes and there is no incident, escalate. |
| `NX-781` | Client version below the minimum | Update to **5.4.2** from Software Center, then restart NakhlaConnect. If Software Center fails, see KB-009. Never edit the client configuration to skip the check (SP-002). |
| `NX-905` | Session expired after 12 hours | Normal. Reconnect and approve the NakhlaKey prompt. |

## Escalation

If the steps above do not fix it, the Service Desk opens a ticket:

- category `network_access`, priority **P2** when one user cannot connect;
- priority **P1** if more than 3 users are affected, or the requester is in the Executive Office (KB-015);
- include the error code, the client version, the gateway and the steps already tried.

## Exceptions

These department rules override the general steps above (KB-016):

- **Finance:** VPN and remote-access problems go **straight to Security Operations**. Finance laptops use the FIN-SECURE profile on GW-3, which only Security Operations may change. Do not walk Finance staff through reinstalling, updating or reconfiguring the client. Check the status page, then draft a ticket with category **`security`**, priority P2 (P1 if more than 3 users).
- **Contractors:** no VPN access. They use the Nakhla Workspace VDI portal (`https://vdi.nakhla.example`). VPN access requests for Contractors are refused; do not open a ticket for them.
- **Executive Office:** every ticket is **P1 (VIP)**.

## Change log

- 2.1 (2026-09-14): added `NX-905`; Finance exception clarified (category `security`).
- 2.0 (2026-06-01): minimum client raised from 4.8 to 5.4; replaces KB-001.
- 1.0 (2025-10-06): first version (`NX-112`, `NX-403`, `NX-503`).
