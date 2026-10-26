---
id: SP-002
title: MFA and device security policy
version: "1.3"
last_updated: 2026-03-01
status: current
authoritative: true
owner: Security Operations
supersedes: null
applies_to: [all]
tags: [security-policy, mfa, nakhlakey, sms, backup-codes, devices, lost-device, contractors, vpn, admin-rights]
error_codes: []
---

# SP-002 — MFA and device security policy

**Policy owner:** Security Operations · **Applies to:** all employees and Contractors.
This policy overrides any knowledge-base article that says something different.

## 1. Multi-factor authentication (MFA)

- MFA is mandatory for every NakhlaID account, including Contractors.
- Allowed methods: the **NakhlaKey** app, or a FIDO2 hardware key (issued to Executive Office staff and IT administrators only).
- **SMS and voice-call codes are not allowed.** They were switched off on 2026-03-01. Nobody at Nakhla can send you an SMS code, and older documents that mention SMS codes are wrong.
- At enrolment you receive **10 one-time backup codes**. Keep them offline (printed or in a password manager). Each code works once.
- New phone, push problems and enrolment steps: KB-007.

## 2. Lost or stolen devices

A lost or stolen phone with NakhlaKey, or a lost or stolen company laptop, is a **security incident**.

1. Report it within **1 hour**: ticket category `security`, priority **P1**.
2. Security Operations locks or wipes the device through NakhlaManage and removes the NakhlaKey registration.
3. You then re-register NakhlaKey on a new phone (KB-007). If you have no backup codes, this needs manager approval and identity verification as described in SP-001 section 2.

Forgetting your phone at home is **not** a security incident: sign in with a backup code (KB-007).

## 3. Which devices may connect

- Only company devices that are enrolled in **NakhlaManage** and compliant (KB-010) may join the Nakhla-Corp Wi-Fi or use the NakhlaConnect VPN. The device certificate that both need is issued only to compliant devices.
- Personal laptops never connect to Nakhla-Corp or the VPN. They may use the Nakhla Workspace VDI portal (`https://vdi.nakhla.example`) with manager approval.
- **Contractors have no VPN access.** They work only through the VDI portal. Their own devices are not enrolled.

## 4. Do not weaken device security

The following are forbidden and are reported to Security Operations:

- disabling disk encryption, antivirus or the NakhlaManage agent;
- bypassing or turning off the NakhlaConnect posture or version check (for example by editing the client configuration);
- using any VPN client other than NakhlaConnect for company systems.

## 5. Administrator rights

Nobody has permanent local administrator rights on a Nakhla laptop. Engineering staff may use the time-limited **Elevate** function in Software Center (KB-009). All Elevate sessions are logged.

## Change log

- 1.3 (2026-03-01): SMS and voice-call codes removed; backup codes made mandatory.
- 1.2 (2025-09-15): Contractors limited to the VDI portal.
- 1.0 (2025-02-03): first version.
