# Gemini2OpenAI 代理服务

## 概述

本服务提供将 Google Gemini API 转换为 OpenAI 兼容格式的代理功能，支持以下特性：

- 完整的 OpenAI API 格式兼容
- 多密钥轮询与负载均衡
- IPv6 CIDR 代理支持
- 流式响应支持
- 自动故障恢复

## 下载安装

```bash
# Linux x86_64
wget https://github.com/blee0036/gemini2oai/releases/download/v<版本>/gemini2oai-v<版本>-linux-amd64 -O gemini2oai

# Linux ARM64
wget https://github.com/blee0036/gemini2oai/releases/download/v<版本>/gemini2oai-v<版本>-linux-arm64 -O gemini2oai

# Windows
https://github.com/blee0036/gemini2oai/releases/download/v<版本>/gemini2oai-v<版本>-windows-amd64.exe
```

请替换 `<版本>` 为实际版本号，访问 [GitHub Releases](https://github.com/blee0036/gemini2oai/releases) 获取最新版本。

## 配置参数

通过环境变量配置服务

| 环境变量                  | 必填 | 说明                              |
|-----------------------|----|---------------------------------|
| GEMINI2OAI_PORT       | 否  | 服务监听端口，默认 3000                  |
| GEMINI2OAI_AUTH_KEY   | 否  | 系统密钥认证头 (与用户自备密钥二选一)            |
| GEMINI2OAI_TOKENS     | 否  | 系统内置密钥列表，多个用逗号分隔                |
| GEMINI2OAI_PROXY_CIDR | 否  | IPv6 CIDR 地址范围，格式：2001:db8::/32 |

密钥优先级：

1. 用户请求头 Authorization: Bearer <用户密钥>
2. 系统内置 TOKENS (当 AUTH_KEY 匹配时)

## 服务部署

### 1. 安装程序（必须 root 权限）

```bash
chmod +x gemini2oai
mv gemini2oai /usr/local/bin/
```

### 2. 创建配置文件

`/etc/gemini2oai.conf`:

```ini
GEMINI2OAI_AUTH_KEY=system_auth_key  # 可选系统认证头
GEMINI2OAI_TOKENS=key1,key2,key3     # 可选系统内置密钥
GEMINI2OAI_PROXY_CIDR=2001:db8::/32  # 可选代理配置
```

### 3. Systemd 服务配置

创建 `/etc/systemd/system/gemini2oai.service`:

```ini
[Unit]
Description=Gemini to OpenAI Proxy
After=network.target

[Service]
User=root
EnvironmentFile=/etc/gemini2oai.conf
ExecStart=/usr/local/bin/gemini2oai
Restart=always
RestartSec=5
StartLimitInterval=0

[Install]
WantedBy=multi-user.target
```

### 4. 启动服务

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now gemini2oai
```

## 使用验证

```bash
# 使用系统密钥
curl -H "Authorization: Bearer system_auth_key" http://localhost:3000/v1/models

# 使用自定义密钥
curl -H "Authorization: Bearer your_gemini_key" http://localhost:3000/v1/models
```

示例请求：

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_gemini_key" \
  -d '{
    "model": "gemini-1.5-pro-latest",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7
  }'
```

## 服务管理

```bash
systemctl status gemini2oai    # 查看状态
journalctl -u gemini2oai -f    # 查看实时日志
systemctl restart gemini2oai   # 重启服务
```

## 注意事项

1. 必须使用 root 用户运行（网络出口绑定需要特权）
2. GEMINI2OAI_PROXY_CIDR 需要完整的 IPv6 CIDR 格式
3. 同时配置 AUTH_KEY 和 TOKENS 时，需 Authorization 头匹配 AUTH_KEY 才会使用系统密钥