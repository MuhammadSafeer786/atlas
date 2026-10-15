# Nimbus API Rate Limits

Rate limits are enforced per API key using a token-bucket algorithm.

## Limits by plan

- Starter: 10 requests per second, burst of 20.
- Growth: 200 requests per second, burst of 400.
- Enterprise: 2,000 requests per second by default; raised on request.

Upserts count as 5 requests each because they trigger re-indexing.

## Handling 429 responses

When a limit is exceeded the API returns HTTP 429 with error code `NIMBUS_RATE_LIMITED` and a `retry-after` header in seconds.

Recommended client behaviour:
1. Honour `retry-after` when present.
2. Otherwise retry with exponential backoff starting at 500 ms, doubling each attempt, with full jitter, capped at 30 seconds.
3. Give up after 6 attempts and surface the error.

## Error code reference

- `NIMBUS_RATE_LIMITED` (429): too many requests.
- `NIMBUS_INDEX_BUSY` (503): the index is rebuilding; retry after a few seconds.
- `NIMBUS_DIM_MISMATCH` (400): vector dimension does not match the index (e.g. sending 768-dim vectors to a 1024-dim index).
- `NIMBUS_QUOTA_EXCEEDED` (402): monthly allowance used up on Starter; upgrade or wait for the next cycle.
