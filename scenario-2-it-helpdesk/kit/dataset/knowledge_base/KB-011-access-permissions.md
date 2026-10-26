---
id: KB-011
title: Access permissions and access requests
version: "1.5"
last_updated: 2026-02-20
status: current
authoritative: true
owner: Identity & Access Management
supersedes: null
applies_to: [all]
tags: [access, permissions, groups, shared-drives, shared-mailbox, approvals, access-request]
error_codes: []
---

# KB-011 — Access permissions and access requests

Access to apps, shared drives, shared mailboxes and VPN is given through **groups**. You ask for a group at
`https://access.nakhla.example`. Nobody at the Service Desk can add you to a group without the approvals below.

## Standard groups

| Group | Gives access to | Who can be a member | Approvers |
|---|---|---|---|
| VPN-Users | NakhlaConnect VPN | all employees, **never Contractors** | manager |
| ENG-Code | NakhlaCode repositories | Engineering, Contractors sponsored by Engineering | manager + repository owner |
| SALES-CRM | NakhlaCRM | Sales (Support: read-only) | Sales Director |
| FIN-Ledger | NakhlaLedger | Finance | Finance Manager |
| HR-People | NakhlaPeople (HR data) | HR | HR Director |
| EXEC-Board | board documents | Executive Office | Chief Executive Officer |

Shared drives are at `\\files.nakhla.example\<department>`. Each folder has an owner who approves access.
Shared mailboxes are groups too: request "Mailbox: name@nakhla.example".

## How to request access

1. Open `https://access.nakhla.example` > New request.
2. Choose the group, write why you need it and for how long.
3. Your manager approves, then the data owner approves. Most requests finish within **2 business days**.
4. Sign out and in again after approval (shared mailboxes can take 60 minutes to appear in Outlook).

If you used to have access and lost it, say so in the request.

## Contractors

- Contractor access lasts at most **90 days** at a time. The sponsoring manager renews it.
- Contractors can never be added to VPN-Users. Their access goes through the VDI portal (KB-016).

## Tickets

Access requests are tracked as category `identity_access`, priority **P4**. If you lost access you had yesterday and cannot work, use priority **P2**.

## Change log

- 1.5 (2026-02-20): shared mailboxes handled as groups.
- 1.0 (2025-01-20): first version.
