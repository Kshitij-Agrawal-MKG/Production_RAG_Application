# HelixDB — Product Overview

## What is HelixDB?

HelixDB is a managed time-series database platform designed for engineering and data teams that need to store, query, and visualise high-frequency metrics, events, and sensor data at scale. HelixDB handles ingestion rates of up to 2 million data points per second per tenant and provides sub-10ms query latency at the 99th percentile for datasets up to 500 GB.

HelixDB is available as a fully managed cloud service (HelixDB Cloud) and as a self-hosted deployment using the HelixDB Community or Enterprise editions.

---

## Core Features

### High-Frequency Ingestion
HelixDB uses a log-structured merge-tree (LSM-tree) storage engine optimised for append-heavy write workloads. Incoming data is buffered in memory, compressed using the Gorilla delta-encoding algorithm, and flushed to persistent storage in immutable segments. This design achieves compression ratios of 8–12× on typical numeric time-series data.

### Flexible Query Language — HelixQL
HelixQL is a SQL-like query language with native time-series extensions. Key operators include:
- `DOWNSAMPLE(interval, aggregation)` — reduces resolution across a time window
- `FILL(strategy)` — fills gaps with forward-fill, backward-fill, linear interpolation, or a constant
- `ANOMALY(sensitivity)` — detects statistical outliers using Z-score or IQR methods
- `ALIGN BY` — aligns multiple series to a common timestamp grid

### Built-in Dashboards
HelixDB includes a web-based dashboard at `https://app.helixdb.io`. The dashboard supports:
- Real-time streaming charts (WebSocket-backed, 100ms refresh)
- Custom alerting rules with Slack, PagerDuty, and email notifications
- Shared workspaces for team collaboration
- Dark mode and responsive mobile layout

---

## Pricing

### Free Tier
- Up to 10,000 data points per day
- 7-day data retention
- 1 dashboard workspace
- Community support (forum only)
- No credit card required

### Starter Plan — $49 per month
- Up to 5 million data points per day
- 90-day data retention
- 5 dashboard workspaces
- CSV and JSON export
- Email support with 48-hour response SLA

### Growth Plan — $199 per month
- Up to 50 million data points per day
- 1-year data retention
- Unlimited dashboard workspaces
- All export formats (CSV, JSON, Parquet, Arrow)
- Priority email support with 8-hour response SLA
- REST API and Python/JavaScript SDKs included
- Webhook integrations

### Enterprise Plan — Custom pricing
- Unlimited data points
- Custom retention periods (up to 10 years)
- Dedicated infrastructure
- SSO/SAML authentication
- Audit logging and compliance reports (SOC 2 Type II, HIPAA)
- 24/7 phone and Slack support with 1-hour response SLA
- Custom SLAs available

### Free Trial
All paid plans include a 14-day free trial with full feature access. No credit card is required to start the trial. The trial automatically downgrades to the Free Tier at expiry unless a payment method is added.

---

## Supported Data Formats

HelixDB accepts data through the following ingestion methods:

### Line Protocol (default)
```
metric_name,tag1=value1,tag2=value2 field1=1.23,field2=4.56 1698765432000000000
```
Compatible with the InfluxDB line protocol specification. Libraries exist for Python, Go, Java, JavaScript, Ruby, and Rust.

### JSON over HTTP
```json
{
  "metric": "cpu.usage",
  "tags": {"host": "web-01", "region": "us-east-1"},
  "fields": {"value": 72.4},
  "timestamp": 1698765432
}
```
POST to `https://ingest.helixdb.io/v1/write`.

### CSV Bulk Import
Upload CSV files up to 500 MB via the dashboard or the `/v1/import` REST endpoint. The first row must be a header. Timestamp column must be named `timestamp` or `ts` and contain Unix epoch seconds or ISO 8601 strings.

### Apache Kafka Integration
HelixDB provides a Kafka connector compatible with Confluent Platform 6.0+ and Apache Kafka 2.8+. Configure with the `helix-kafka-connector` package. The connector supports exactly-once semantics when used with Kafka transactions.

### OpenTelemetry (OTLP)
HelixDB accepts metrics via the OpenTelemetry Protocol (OTLP) over gRPC and HTTP. This allows direct ingestion from any OpenTelemetry-instrumented application without a separate collector.

---

## System Requirements — Self-Hosted

### HelixDB Community Edition
- OS: Ubuntu 20.04+, Debian 11+, RHEL 8+, or any Linux with kernel 5.4+
- CPU: 4 cores minimum, 8 recommended
- RAM: 8 GB minimum, 16 GB recommended for production
- Disk: SSD required; NVMe recommended for write-heavy workloads
- Network: 1 Gbps minimum

### HelixDB Enterprise Edition
- All Community Edition requirements, plus:
- CPU: 16 cores minimum
- RAM: 64 GB minimum
- Disk: NVMe SSD with 10 GB/s sustained write throughput
- Network: 10 Gbps recommended

### Docker Deployment
```bash
docker pull helixdb/community:latest
docker run -d -p 8086:8086 -v /data/helix:/var/lib/helixdb helixdb/community:latest
```

### Kubernetes Helm Chart
```bash
helm repo add helixdb https://charts.helixdb.io
helm install helixdb helixdb/helixdb --namespace helix --create-namespace
```
