# Incident Response Runbook

## Roles

- **Incident Commander (IC):** owns the incident, makes decisions, runs the bridge call.
- **Communications Lead:** updates the public status page every 30 minutes during a Severity 1 incident.
- **Scribe:** keeps the timeline in the incident document.

## Process

1. Declare the incident in the #incidents channel with `/incident declare` and pick a severity.
2. The on-call engineer becomes IC until handing off explicitly.
3. Mitigate first, diagnose second: roll back the last deploy if it happened within 2 hours of the first alert.
4. Post the first status-page update within 15 minutes of declaring a Severity 1.
5. Close the incident only after error rates have been normal for 30 minutes.

## Postmortems

A blameless postmortem is required for every Severity 1 and Severity 2 incident, due within 5 business days. It must include a timeline, root cause analysis using the "5 whys", and action items with owners and due dates.
