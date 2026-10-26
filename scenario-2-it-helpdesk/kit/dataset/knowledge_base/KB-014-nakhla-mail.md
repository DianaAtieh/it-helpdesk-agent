---
id: KB-014
title: "Nakhla Mail: mailbox, shared mailboxes, phishing"
version: "1.7"
last_updated: 2026-06-30
status: current
authoritative: true
owner: Collaboration Services
supersedes: null
applies_to: [all]
tags: [email, outlook, nakhla-mail, mailbox, archive, shared-mailbox, phishing, mobile-mail, calendar, delegates]
error_codes: [NXM-421, NXM-552]
---

# KB-014 — Nakhla Mail: mailbox, shared mailboxes, phishing

Nakhla Mail is the company email and calendar service. Use Outlook on your laptop, Outlook on the web, or the mail app inside the NakhlaManage work profile on your phone.

## Mailbox full (`NXM-552`)

- Every mailbox holds **50 GB**. When it is full, you cannot send mail and senders get `NXM-552`.
- Every mailbox also has an **Online Archive**. Right-click a folder > Archive, or set the archive policy to "move items older than 1 year".
- Empty Deleted Items and remove large attachments you no longer need.
- Still full after archiving? Ticket category `email_collab`, priority **P3**.

## Mail is late (`NXM-421`)

`NXM-421` means "temporarily delayed, will retry". The message is not lost. Check the status page for a mail incident. If a message is more than 4 hours late and there is no incident, open a ticket (category `email_collab`, P3).

## Outlook keeps asking for my password

1. Make sure your password has not expired (KB-004) and your account is not locked (KB-006).
2. Close all Office apps, sign out of Outlook (File > Office Account > Sign out), and sign in again with your work email.
3. On Windows, remove old "MicrosoftOffice" entries from Credential Manager, then restart.
4. On your phone, remove the mail account and add it again through the NakhlaManage work profile.

## Shared mailboxes and delegates

- Shared mailboxes are requested as a group at `https://access.nakhla.example` (KB-011). They appear in Outlook within 60 minutes.
- Calendar delegates (for example an executive assistant) are set in Outlook > File > Account settings > Delegate access. After a mailbox migration delegates may need to be added again by Collaboration Services.

## Rules for email

- **Automatic forwarding to external addresses is blocked** by Security Operations. Rules that forward outside `nakhla.example` silently fail.
- Scan-to-email from printers only sends to `@nakhla.example` addresses.

## Phishing

1. Do not click links or open attachments in a suspicious message.
2. Use the **Report Phish** button in Outlook. It sends the message to Security Operations and removes it from your inbox.
3. If you already typed your password into the page, follow SP-001 section 4: reset your password now and report it as category `security`, priority **P1**.

## Change log

- 1.7 (2026-06-30): Online Archive enabled for all mailboxes.
- 1.5 (2025-11-03): external auto-forwarding blocked.
- 1.0 (2025-01-10): first version.
