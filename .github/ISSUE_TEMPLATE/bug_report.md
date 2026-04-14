---
name: Bug Report
about: Something is broken or not working as expected
title: "[BUG] "
labels: bug
assignees: ''
---

## Describe the Bug

A clear and concise description of what the bug is.

## Steps to Reproduce

1. Start services: MongoDB / vLLM / semantic chunking (which ones?)
2. Run command: `...`
3. See error

## Expected Behaviour

What you expected to happen.

## Actual Behaviour

What actually happened. Include the **full error traceback**:

```
paste error here
```

## Environment

| Item | Value |
|------|-------|
| OS | e.g. Ubuntu 22.04 |
| Python | e.g. 3.10.12 |
| GPU | e.g. A100-PCIE-40GB |
| VRAM | e.g. 40 GB |
| CUDA | e.g. 12.1 |
| vLLM version | e.g. 0.4.3 |
| PyTorch version | e.g. 2.1.0 |

## Service State at Time of Error

- [ ] MongoDB running (`mongosh --eval "db.runCommand({ping:1})"` → `{ok:1}`)
- [ ] vLLM running (`curl http://localhost:8000/v1/models` → lists model)
- [ ] Semantic chunking running (`curl http://localhost:6000/health` → `{"status":"ok"}`)

## Additional Context

Any other context (config.yaml changes, custom PDFs, etc.)
