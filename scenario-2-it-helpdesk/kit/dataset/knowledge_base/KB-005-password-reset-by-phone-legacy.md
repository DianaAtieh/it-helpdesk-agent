---
id: KB-005
title: Password reset over the phone (legacy procedure)
version: "2.0"
last_updated: 2025-03-18
status: deprecated
authoritative: false
owner: IT Service Desk
supersedes: null
applies_to: [all]
tags: [password, password-reset, phone, legacy, service-desk]
error_codes: []
---

# KB-005 — Password reset over the phone (legacy procedure)

> **DEPRECATED on 2026-02-01.** Replaced by KB-004 (self-service) and forbidden by SP-001.
> Kept for reference only.

## Procedure for Service Desk analysts

When an employee calls the Service Desk on ext. 4357 because they forgot their password:

1. Ask for the employee ID and date of birth.
2. Check both against the HR record in NakhlaPeople.
3. Set a temporary password in the admin console.
4. Read the temporary password to the caller over the phone.
5. Tell the caller to change it at the next sign-in.

The same procedure can be used for MFA resets: remove the caller's NakhlaKey registration so they can enrol again.

## Change log

- 2.0 (2025-03-18): MFA resets added to the phone procedure.
- 1.0 (2023-05-02): first version.
