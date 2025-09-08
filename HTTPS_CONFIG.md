# HTTPS 配置指南

## 问题描述

当设置 `MCP_PUBLIC_BASE_URL=https://ai.procaas.com:8787` 时出现 "WARNING: Invalid HTTP request received" 错误。

## 解决方案

### 方案1: 使用 HTTP（推荐用于测试）

如果您不需要 HTTPS，请将配置改为 HTTP：

```bash
# .env
MCP_SSE_HOST=0.0.0.0
MCP_SSE_PORT=8787
MCP_PUBLIC_BASE_URL=http://ai.procaas.com:8787
```

### 方案2: 配置 HTTPS

如果您需要 HTTPS，请提供 SSL 证书：

```bash
# .env
MCP_SSE_HOST=0.0.0.0
MCP_SSE_PORT=8787
MCP_PUBLIC_BASE_URL=https://ai.procaas.com:8787
MCP_SSL_CERT_FILE=/path/to/your/certificate.crt
MCP_SSL_KEY_FILE=/path/to/your/private.key
```

### 方案3: 使用反向代理

您也可以使用 Nginx 或 Apache 作为反向代理来处理 HTTPS：

```nginx
# nginx.conf 示例
server {
    listen 443 ssl;
    server_name ai.procaas.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    location / {
        proxy_pass http://127.0.0.1:8787;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

然后配置 MathViz 使用 HTTP：

```bash
# .env
MCP_SSE_HOST=127.0.0.1
MCP_SSE_PORT=8787
MCP_PUBLIC_BASE_URL=https://ai.procaas.com
```

## 健康检查

启动服务器后，您可以访问健康检查端点来验证配置：

```bash
curl http://ai.procaas.com:8787/health
```

或使用 HTTPS：

```bash
curl https://ai.procaas.com:8787/health
```

## 常见问题

1. **端口未开放**: 确保 8787 端口在防火墙中开放
2. **SSL 证书问题**: 确保证书文件路径正确且可读
3. **域名解析**: 确保域名正确解析到服务器 IP
4. **权限问题**: 确保服务器有权限读取证书文件

## 调试步骤

1. 检查服务器日志：
   ```bash
   MCP_LOG_LEVEL=DEBUG uv run python -m server.main
   ```

2. 测试本地连接：
   ```bash
   curl http://localhost:8787/health
   ```

3. 检查端口是否监听：
   ```bash
   netstat -tlnp | grep 8787
   ```
