# HelixDB — Data Management, Integrations & Support

## Data Export

### Export Formats
HelixDB supports exporting data in the following formats:
- **CSV** — available on all paid plans; includes headers by default
- **JSON** — newline-delimited JSON (NDJSON); available on all paid plans
- **Apache Parquet** — columnar format for analytics pipelines; Growth and Enterprise plans
- **Apache Arrow IPC** — for direct integration with pandas, Polars, and DuckDB; Growth and Enterprise plans

### Exporting via the Dashboard
1. Open the query editor and run your query
2. Click **Export** in the top-right of the results panel
3. Choose your format and date range
4. Click **Download** — files up to 100 MB are downloaded directly; larger files are staged and a download link is sent by email within 10 minutes

### Exporting via the REST API
```bash
curl -X POST https://api.helixdb.io/v1/export \
  -H "Authorization: Bearer hx_live_sk_xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "SELECT * FROM cpu.usage WHERE time > now() - 7d",
    "format": "parquet",
    "compression": "snappy"
  }'
```
The response includes a presigned S3 URL valid for 24 hours.

### Scheduled Exports (Growth and Enterprise)
Configure automated exports under **Settings → Scheduled Exports**. Exports can be delivered to:
- Amazon S3 (provide bucket name and IAM role ARN)
- Google Cloud Storage
- Azure Blob Storage
- SFTP server

---

## Data Retention

### Retention Policies
Each metric namespace can have an independent retention policy. After the retention period expires, data is permanently deleted and cannot be recovered.

Default retention by plan:
- Free: 7 days
- Starter: 90 days
- Growth: 365 days (1 year)
- Enterprise: Configurable up to 3,650 days (10 years)

### Setting a Custom Retention Policy
```bash
curl -X PUT https://api.helixdb.io/v1/namespaces/cpu.usage/retention \
  -H "Authorization: Bearer hx_live_sk_xxx" \
  -d '{"retention_days": 180}'
```

### Data Deletion Requests
To delete all data for a specific metric or time range:
```bash
curl -X DELETE https://api.helixdb.io/v1/data \
  -H "Authorization: Bearer hx_live_sk_xxx" \
  -d '{"metric": "cpu.usage", "start": "2024-01-01", "end": "2024-03-31"}'
```
Deletion is asynchronous. A deletion job ID is returned and can be polled at `/v1/jobs/{id}`.

---

## File Upload Limits

### Ingest Payload Limits
- Single HTTP write request: 5 MB maximum
- Single JSON object: 1 MB maximum
- Line protocol batch: 50,000 lines maximum per request

### Bulk Import via CSV
- Maximum CSV file size: 500 MB
- Maximum rows per import: 10 million
- Supported timestamp formats: Unix epoch (seconds or milliseconds), ISO 8601

### Dashboard Attachment Uploads
- Maximum file size for dashboard annotations: 25 MB
- Supported formats: PNG, JPG, SVG, PDF

---

## Third-Party Integrations

### Monitoring and Alerting
- **Grafana** — official HelixDB data source plugin available in the Grafana plugin catalogue
- **PagerDuty** — native alert routing; configure under **Settings → Notifications**
- **Opsgenie** — webhook-based integration; use the generic webhook URL with JSON payload
- **Slack** — OAuth-based app; alerts delivered to any Slack channel

### Data Pipeline Tools
- **Apache Kafka** — helix-kafka-connector (see ingestion docs)
- **Apache Spark** — `helix-spark-connector` for batch reads via DataFrame API
- **dbt** — HelixDB adapter available on PyPI as `dbt-helix`
- **Airbyte** — community connector in the Airbyte connector catalogue

### Observability Platforms
- **Datadog** — HelixDB integration available in the Datadog marketplace
- **New Relic** — OpenTelemetry OTLP endpoint compatible (see ingestion docs)
- **Prometheus** — remote write adapter for forwarding Prometheus metrics to HelixDB

### Webhook Configuration
Webhooks are available on Growth and Enterprise plans. A webhook fires on:
- Alert state change (triggered, resolved)
- Data import completion
- Scheduled export completion

Configure webhooks at **Settings → Webhooks**. Each webhook can target any HTTPS endpoint. Payloads are signed with HMAC-SHA256 using a secret you provide.

---

## Uptime and Service Level Agreements

### HelixDB Cloud SLA

| Plan | Monthly Uptime Guarantee | Credits |
|------|-------------------------|---------|
| Free | No SLA | — |
| Starter | 99.5% | 10% credit for each 0.1% below target |
| Growth | 99.9% | 10% credit for each 0.1% below target |
| Enterprise | 99.99% | 25% credit for each 0.01% below target |

Uptime is measured from the HelixDB status page (`https://status.helixdb.io`) using external monitors in 5 regions every 60 seconds.

### Scheduled Maintenance
Scheduled maintenance windows occur on the third Sunday of each month between 02:00–04:00 UTC. Maintenance windows are announced at least 72 hours in advance via the status page and email. Planned maintenance does not count against the uptime SLA.

### Incident Response
- **P1 (service down or data loss)** — response within 15 minutes (Enterprise), 1 hour (Growth)
- **P2 (degraded performance)** — response within 2 hours (Enterprise), 8 hours (Growth)
- **P3 (minor issues)** — response within 1 business day

---

## Cancellation and Refunds

### Cancellation Policy
You can cancel your HelixDB subscription at any time from **Settings → Billing → Cancel Plan**. Cancellation takes effect at the end of the current billing period. Your data remains accessible until the retention period expires.

### Refund Policy
HelixDB offers a 30-day money-back guarantee for first-time subscribers on Starter and Growth plans. To request a refund within the first 30 days, contact support at `billing@helixdb.io` with your account email and reason for cancellation. Refunds are processed within 5–7 business days to the original payment method.

After 30 days, HelixDB does not offer prorated refunds for unused subscription time. Enterprise contract terms may vary.

### Payment Failure
If a payment fails:
1. HelixDB retries the charge after 3 days, then again after 7 days
2. A notification email is sent to the account owner after each failed attempt
3. If the payment fails three times, the account is downgraded to the Free Tier
4. All data above Free Tier retention limits (7 days) is scheduled for deletion 30 days after downgrade
5. Reactivate by updating your payment method under **Settings → Billing**

---

## Customer Support

### Support Channels by Plan

| Plan | Channel | Response Time |
|------|---------|---------------|
| Free | Community forum (`community.helixdb.io`) | Best effort |
| Starter | Email (`support@helixdb.io`) | 48 hours |
| Growth | Priority email | 8 hours |
| Enterprise | Email, phone, dedicated Slack channel | 1 hour (24/7) |

### How to Contact Support
- **Email**: `support@helixdb.io`
- **Enterprise Slack**: Your dedicated channel is created during onboarding
- **Phone** (Enterprise only): +1-800-HELIXDB (US) or +44-20-HELIX-01 (UK)
- **Community forum**: `https://community.helixdb.io`

### Filing a Bug Report
Include the following in your bug report:
1. Your account ID (found under **Settings → Account**)
2. The exact query or API call that triggered the issue
3. The full error message and HTTP status code
4. The approximate time the issue occurred (UTC)
5. The SDK version if using a client library

### SDK and Programming Language Support
HelixDB provides officially maintained SDKs for:
- **Python** (`pip install helixdb`) — Python 3.9+
- **JavaScript/TypeScript** (`npm install @helixdb/client`) — Node.js 18+
- **Go** (`go get github.com/helixdb/helix-go`) — Go 1.21+
- **Java** (Maven/Gradle: `io.helixdb:helix-client:2.1.0`) — Java 11+
- **Rust** (`cargo add helix-client`) — Rust 1.70+

Community-maintained SDKs are available for Ruby, PHP, and .NET.
