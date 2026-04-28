# HelixDB — Authentication & Security

## Authentication Methods

### API Key Authentication
Every HelixDB account is issued a primary API key on registration. API keys are passed in the `Authorization` header:

```
Authorization: Bearer hx_live_sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

API keys have three permission scopes:
- **Read-only** — can query data and read dashboard configurations
- **Write** — can ingest data and manage retention policies
- **Admin** — full access including user management and billing

To create additional API keys, go to **Settings → API Keys → Create New Key**. Each key can be assigned a name, an expiry date, and an IP allowlist.

### Password Authentication (Dashboard)
The HelixDB web dashboard uses email and password authentication with the following requirements:
- Minimum 12 characters
- At least one uppercase letter, one number, and one special character
- Passwords are hashed with bcrypt (cost factor 12) and never stored in plain text

### Password Reset
To reset your password:
1. Go to `https://app.helixdb.io/forgot-password`
2. Enter your registered email address
3. Check your email for a reset link (valid for 1 hour)
4. Click the link and enter your new password
5. All existing sessions are invalidated after a password reset

### Multi-Factor Authentication (MFA)
MFA is available for all paid plans. Supported methods:
- TOTP apps (Google Authenticator, Authy, 1Password)
- Hardware security keys (WebAuthn/FIDO2)
- Backup codes (10 single-use codes generated at MFA setup)

MFA is mandatory for Admin-scoped accounts on Enterprise plans.

### SSO / SAML 2.0 (Enterprise only)
Enterprise customers can configure SSO via SAML 2.0 with any compliant identity provider:
- Okta
- Microsoft Entra ID (Azure AD)
- Google Workspace
- OneLogin

SAML configuration is available under **Settings → Security → SSO Configuration**.

---

## Security Architecture

### Encryption in Transit
All traffic between clients and HelixDB Cloud is encrypted using TLS 1.3. TLS 1.0 and 1.1 are disabled. The dashboard enforces HTTPS and sends HTTP Strict Transport Security (HSTS) headers with a max-age of 1 year.

### Encryption at Rest
Data stored in HelixDB Cloud is encrypted at rest using AES-256-GCM. Encryption keys are managed by AWS KMS (for AWS-hosted tenants) or GCP Cloud KMS (for GCP-hosted tenants). Key rotation is performed automatically every 90 days.

### Network Isolation
Each HelixDB Cloud tenant is deployed in an isolated VPC. Cross-tenant network traffic is blocked at the infrastructure level. Enterprise customers can additionally configure VPC peering or AWS PrivateLink to avoid traffic traversing the public internet.

### Compliance Certifications
- SOC 2 Type II (annual audit, report available on request)
- ISO 27001:2022
- HIPAA Business Associate Agreement (BAA) available for Enterprise plans
- GDPR compliant — all EU data remains within the EU-WEST-1 region by default

### Penetration Testing
HelixDB undergoes external penetration testing twice per year. Customers on Enterprise plans may request the executive summary of the most recent pen test report under NDA.

---

## API Rate Limits

### Default Rate Limits by Plan

| Plan | Write requests/sec | Query requests/min | Bulk import requests/hour |
|------|-------------------|-------------------|--------------------------|
| Free | 10 | 30 | 0 (not available) |
| Starter | 100 | 300 | 10 |
| Growth | 1,000 | 3,000 | 100 |
| Enterprise | Custom (default 10,000) | Custom | Custom |

### Rate Limit Headers
Every API response includes rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1698765500
```

### Handling Rate Limit Errors
When a rate limit is exceeded, the API returns HTTP 429 with a `Retry-After` header indicating the number of seconds to wait. The recommended backoff strategy is exponential backoff with jitter, starting at 1 second.

### Burst Allowance
All plans include a burst allowance of 3× the base rate for up to 10 seconds per minute. After the burst allowance is exhausted, standard rate limits apply for the remainder of the minute.

---

## User Roles and Permissions

### Built-in Roles

**Owner**
- Created automatically for the account creator
- Full access to all features including billing and plan management
- Cannot be deleted; ownership can be transferred to another admin

**Admin**
- Can manage users, API keys, and security settings
- Cannot access billing details
- Can create and delete workspaces

**Editor**
- Can create and edit dashboards, queries, and alerts
- Can ingest data via the API
- Cannot manage users or security settings

**Viewer**
- Read-only access to dashboards and query results
- Cannot ingest data or modify any configuration
- Suitable for stakeholders and external collaborators

### Custom Roles (Enterprise only)
Enterprise customers can define custom roles with granular permission sets. Permissions are available at the workspace, metric namespace, and operation level. Custom roles are defined in YAML and applied via the Admin API.

### Role Assignment
Users are invited by email. Pending invitations expire after 7 days. An Admin or Owner can resend or revoke invitations at any time from **Settings → Team**.
