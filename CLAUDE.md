# Claude Code Guide - Truss Annotation Intelligent Function Layer (AIFL)

This repository contains annotation-related intelligent functions including automated annotation and image services.

## Quick Reference

```bash
# Install dependencies
npm install

# Copy utils to all services
node scripts/copy-utils.js

# Generate templates for a service
node scripts/generate-service-templates.js services/<service-name>
```

## Services

| Service | Runtime | Purpose |
|---------|---------|---------|
| automated-annotation | python3.11 | Automated image annotation |
| image-service | nodejs18.x | Image processing and management |

## Python Services

The `automated-annotation` service uses Python:
- Handler: `index.py`
- Dependencies: `layer-requirements.txt`
- Runtime: `python3.11`

### Python-specific config.json
```json
{
  "deployment": {
    "runtime": "python3.11",
    ...
  }
}
```

## Node.js Services

The `image-service` uses Node.js:
- Handler: `index.js`
- Runtime: `nodejs18.x`

## Shared Utils

Utils are copied but Python services may not use all of them:
- `response.js`
- `structured-logger.js`
- `parameter-validator.js`
- `secrets-manager.js`

## Workflow

1. Edit `config.json` for endpoint changes
2. Run `node scripts/generate-service-templates.js services/<name>`
3. Edit `index.js` or `index.py` for logic
4. Run `node scripts/copy-utils.js` if utils changed
5. Commit and push
