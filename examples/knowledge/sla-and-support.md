# Service Level Agreement and Support Policy

## Uptime commitments

Monthly uptime is measured per region as the percentage of one-minute intervals in which the query API returned successful responses.

| Plan       | Uptime SLA | Service credit if missed |
|------------|------------|--------------------------|
| Starter    | none       | none                     |
| Growth     | 99.9%      | 10% of monthly fee       |
| Enterprise | 99.99%     | 25% of monthly fee       |

A 99.9% SLA allows roughly 43.8 minutes of downtime per month; 99.99% allows roughly 4.4 minutes.

Service credits must be requested within 30 days of the incident by opening a ticket with the subject "SLA credit request". Credits are applied to the next invoice and cannot be exchanged for cash.

## Severity levels

- **Severity 1 (Critical):** production is down or data integrity is at risk. Enterprise first response: 30 minutes, 24/7.
- **Severity 2 (High):** major degradation, workaround exists. Enterprise first response: 2 hours.
- **Severity 3 (Normal):** minor issue or question. First response: 1 business day.

## Exclusions

The SLA does not cover scheduled maintenance announced at least 72 hours in advance, outages caused by customer misconfiguration, or beta features (anything labelled "preview" in the console).
