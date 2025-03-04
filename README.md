# Gemini to OpenAI API 适配器

[English Version](README_en.md)

## 项目简介

本项目实现Google Gemini API到OpenAI API的协议转换适配器，支持以下功能：

- 支持文本生成、Embeddings、模型列表等API
- 自动代理路由（需配置）
- 多API密钥负载均衡
- 流式传输支持

## 快速开始

### Docker部署（推荐）

```bash
docker run -d \
  --name gemini2oai \
  --restart=always \
  -p 3000:3000 \
  -e TOKENS="your_google_api_key1,your_google_api_key2" \
  -e proxy_cidr="2001:db8::/64" \
  -e proxy_user="user" \
  -e proxy_pass="pass" \
  -e proxy_port="7890" \
  ghcr.io/blee0036/gemini2oai:latest
```

### 环境参数说明

| 参数名        | 必填 | 说明                      | 示例                  |
|------------|----|-------------------------|---------------------|
| TOKENS     | 否  | 默认 Google API密钥列表（逗号分隔） | "api_key1,api_key2" |
| PORT       | 否  | 服务监听端口（默认3000）          | "8080"              |
| proxy_cidr | 否  | IPv6代理CIDR段             | "2001:db8::/64"     |
| proxy_user | 否  | 代理用户名                   | "proxyuser"         |
| proxy_pass | 否  | 代理密码                    | "proxypass"         |
| proxy_port | 否  | 代理端口                    | "7890"              |

## 验证服务

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
-H "Authorization: Bearer your_google_api_key" \
-H "Content-Type: application/json" \
-d '{
  "model": "gemini-1.5-pro-latest",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

## 进阶配置

1. 代理配置要求：
    - 需提供IPv6段
    - 需保证每个API密钥对应不同IP出口
    - 代理需支持HTTP Basic认证
