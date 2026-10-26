---
id: KB-016
title: Department IT matrix and exceptions
version: "2.3"
last_updated: 2026-09-10
status: current
authoritative: true
owner: IT Service Desk
supersedes: null
applies_to: [all]
tags: [departments, exceptions, engineering, finance, hr, sales, support, executive-office, contractors, vip, vdi, routing]
error_codes: []
---

# KB-016 — Department IT matrix and exceptions

Department rules **override the general rules** in other KB articles for employees of that department.
They never override a security policy (SP-001, SP-002).
Find an employee's department in the directory before applying a rule; do not guess it from the job title.

## Matrix

| Department | Remote access | Special rules | Priority rule |
|---|---|---|---|
| **Engineering** | NakhlaConnect VPN (GW-1 or GW-2) | May use Ubuntu 22.04 LTS laptops. Temporary local admin through Software Center > Elevate (30 minutes). Access to NakhlaCode via ENG-Code. | standard (KB-015) |
| **Finance** | NakhlaConnect VPN with the **FIN-SECURE** profile on GW-3 | **All VPN and remote-access problems go straight to Security Operations:** category `security`. Do not troubleshoot the client with Finance staff; check the status page, then draft the ticket. NakhlaLedger via FIN-Ledger. | standard; Finance VPN problems are P2 (P1 if more than 3 users) |
| **HR** | NakhlaConnect VPN | HR data (NakhlaPeople, HR-People group) is approved by the HR Director only. Never send HR data to external addresses. | standard |
| **Sales** | NakhlaConnect VPN | Often remote. NakhlaCRM via SALES-CRM. Phone mail only through the NakhlaManage work profile. | standard |
| **Support** (customer support for Nakhla products, not the IT Service Desk) | NakhlaConnect VPN | NakhlaCRM read-only on request. Postman and Python available in Software Center. | standard |
| **Executive Office** | NakhlaConnect VPN; FIDO2 hardware key available | **VIP: every ticket is P1**, whatever the issue. The Service Desk lead is informed of every Executive Office ticket. An executive assistant may report on behalf of an executive; the ticket is for the executive. | **always P1** |
| **Contractors** | **No VPN.** Only the Nakhla Workspace VDI portal `https://vdi.nakhla.example` | VPN access requests are refused by policy (SP-002): explain the VDI portal and **do not open a ticket** for VPN access. VDI problems are tickets with category `network_access`. No Software Center installs; tools are provided in VDI. Accounts expire automatically at the contract end date (KB-012). Access is approved by the sponsoring manager, 90 days at a time (KB-011). | standard |

## How to use the matrix

1. Look up the employee (directory: department, VIP flag, contractor flag).
2. Apply the general KB article for the problem.
3. Apply the department rule above where it differs. The department rule wins.
4. Apply the priority rules (KB-015). Executive Office is always P1.

## Change log

- 2.3 (2026-09-10): Contractors: no ticket for VPN access requests.
- 2.0 (2025-10-01): Finance VPN problems routed to Security Operations.
- 1.2 (2025-09-15): Contractors limited to the VDI portal (SP-002).
- 1.0 (2025-03-01): first version (Executive Office VIP rule).
