# Security Guidelines

This document outlines security best practices for deploying and operating the Twilio Voice Agent.

## Environment Variables & Secrets

### Never Commit Secrets
- **NEVER** commit `.env` files to version control
- The `.gitignore` file excludes `.env` by default
- Use `.env.example` as a template (contains no real secrets)

### Secret Management
- Store secrets in environment variables, not in code
- Use a secrets manager in production (AWS Secrets Manager, HashiCorp Vault, etc.)
- Rotate API keys regularly
- Use separate API keys for development and production

### Required Secrets
| Variable | Description | Sensitivity |
|----------|-------------|-------------|
| `TWILIO_ACCOUNT_SID` | Twilio account identifier | High |
| `TWILIO_AUTH_TOKEN` | Twilio authentication token | **Critical** |
| `DEEPGRAM_API_KEY` | Deepgram API key | High |
| `CARTESIA_API_KEY` | Cartesia API key | High |
| `GROQ_API_KEY` | Groq API key | High |

## Logging

### Safe Logging Defaults
This application implements safe logging practices:

1. **No Secret Logging**: API keys and tokens are never logged
2. **Structured Logging**: Uses `structlog` for consistent, parseable logs
3. **Log Levels**: Production should use `INFO` or `WARNING` level
4. **Sensitive Data Masking**: Phone numbers and personal data are masked in logs

### Log Level Guidelines
- `DEBUG`: Development only, may contain verbose data
- `INFO`: Normal operation, safe for production
- `WARNING`: Potential issues, no sensitive data
- `ERROR`: Failures, includes context but no secrets
- `CRITICAL`: System failures requiring immediate attention

## Network Security

### TLS/SSL
- **Always use HTTPS/WSS in production**
- Twilio requires WSS (WebSocket Secure) for media streams
- Use a reverse proxy (nginx, Cloudflare) for TLS termination
- Minimum TLS version: 1.2

### Firewall Rules
- Only expose port 7860 (or your configured port)
- Restrict inbound connections to Twilio IP ranges if possible
- Use network segmentation in production

### Twilio Webhook Validation
- Validate incoming requests are from Twilio
- Use Twilio's request validation in production
- Implement rate limiting on endpoints

## API Security

### Groq/LLM
- Use the minimum required permissions
- Monitor API usage for anomalies
- Set up billing alerts

### Deepgram
- Use streaming-only keys if possible
- Disable unused features
- Monitor transcription volume

### Cartesia
- Use voice-specific keys if available
- Monitor synthesis usage

## Call Security

### Data Handling
- Call audio is processed in real-time and not stored by default
- Transcripts may be logged (configurable)
- Extracted data should be handled per your privacy policy

### PII Considerations
- Phone numbers are logged with partial masking
- Names and personal info from extraction should be handled carefully
- Implement data retention policies

## Deployment Checklist

### Pre-Production
- [ ] All secrets stored securely (not in code)
- [ ] `.env` file excluded from version control
- [ ] TLS/SSL configured
- [ ] Log level set to `INFO` or higher
- [ ] Error reporting configured (no secrets in reports)
- [ ] Rate limiting enabled

### Production
- [ ] Secrets manager integration
- [ ] Monitoring and alerting configured
- [ ] Regular security updates scheduled
- [ ] Backup and recovery tested
- [ ] Incident response plan documented

## Incident Response

### If Secrets Are Exposed
1. **Immediately rotate** the exposed credentials
2. Review access logs for unauthorized usage
3. Update all deployments with new credentials
4. Audit for any data breach

### Reporting Security Issues
If you discover a security vulnerability, please:
1. Do NOT create a public GitHub issue
2. Email security concerns to your security team
3. Provide detailed reproduction steps
4. Allow time for a fix before disclosure

## Least Privilege Principle

### Twilio
- Use API keys with minimal required permissions
- Separate keys for different environments
- Restrict phone number access

### Infrastructure
- Run the application as a non-root user
- Use read-only file systems where possible
- Limit container capabilities in Docker

## Compliance Considerations

### Phone Calls
- Inform callers they're speaking with an AI (where legally required)
- Implement call recording consent flows if recording
- Follow TCPA/GDPR/local regulations

### Data Retention
- Define and implement data retention policies
- Provide data deletion mechanisms
- Document data flows for compliance audits

---

## Quick Reference

```bash
# Check for secrets in code (should return nothing)
grep -r "TWILIO_AUTH_TOKEN\|API_KEY" --include="*.py" src/ server/

# Verify .env is gitignored
git status .env  # Should show as ignored

# Check log level in production
grep LOG_LEVEL .env  # Should be INFO or WARNING
```

**Remember**: Security is everyone's responsibility. When in doubt, ask!
