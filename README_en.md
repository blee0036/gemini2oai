## Overview

This project provides an API adapter that converts Google Gemini API to OpenAI compatible API, featuring:

- Support for text generation, embeddings and model list APIs
- Automatic proxy routing (requires configuration)
- Multi-API key load balancing
- Streaming support

## Quick Start

### Docker Deployment (Recommended)

```bash
docker run -d \
  --name gemini2oai \
  --restart=always \
  -p 3000:3000 \
  -e TOKENS="your_google_api_key1,your_google_api_key2" \
  -e proxy_cidr="2001:db8::/48" \
  -e proxy_user="user" \
  -e proxy_pass="pass" \
  -e proxy_port="7890" \
  ghcr.io/blee0036/gemini2oai:latest
```

### Environment Variables

| Parameter  | Required | Description                               | Example         |
|------------|----------|-------------------------------------------|-----------------|
| TOKENS     | NO       | Default Google API keys (comma-separated) | "key1,key2"     |
| PORT       | No       | Service port (default 3000)               | "8080"          |
| proxy_cidr | No       | IPv6 proxy CIDR block                     | "2001:db8::/48" |
| proxy_user | No       | Proxy username                            | "proxyuser"     |
| proxy_pass | No       | Proxy password                            | "proxypass"     |
| proxy_port | No       | Proxy port                                | "7890"          |

## Verify Service

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
-H "Authorization: Bearer your_google_api_key" \
-H "Content-Type: application/json" \
-d '{
  "model": "gemini-1.5-pro-latest",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```