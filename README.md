# MathViz MCP Server

一个基于 Model Context Protocol (MCP) 的服务：
- 接收题干
- 使用 LLM 生成 matplotlib 绘图代码
- 在 Docker 沙箱中执行代码并产出图片
- 通过本地静态服务返回可访问链接

## 目录结构

```
MathViz/
  Dockerfile
  requirements.txt
  server/
    __init__.py
    main.py
  public/
    images/  # 运行时生成
```

## 依赖安装

- Python 3.10+
- Docker (Windows 下建议 Docker Desktop)
- 可选：FastAPI/uvicorn（已在 requirements.txt 内）

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

如遇到 Pydantic 相关错误（如 `Unable to apply constraint 'host_required'`），请确保使用的是一个全新虚拟环境，并按照 `requirements.txt` 固定版本安装，这里已将 `pydantic<2.10` 与 `annotated-types<0.7` 锁定以规避兼容性问题。

## 构建 Docker 运行镜像（推荐）

```bash
# 在 MathViz 目录下
docker build -t mathviz-runner:latest .
```

运行时使用该镜像：

- 环境变量：`MCP_DOCKER_IMAGE=mathviz-runner:latest`
- 关闭运行期安装：`MCP_DOCKER_INSTALL_DEPS=false`

## 必要环境变量（建议通过 .env 文件提供）

- `OPENAI_API_KEY`: OpenAI/兼容接口的 API Key
- `OPENAI_BASE_URL`: 可选，自定义 OpenAI 兼容网关地址
- `OPENAI_MODEL`: 可选，默认 `gpt-4o-mini`
- `MCP_SSE_HOST`: 可选，SSE 和静态资源服务器 host，默认为 `MCP_ASSETS_HOST` 的值或 `127.0.0.1`
- `MCP_SSE_PORT`: 可选，SSE 和静态资源服务器端口，默认为 `MCP_ASSETS_PORT` 的值或 `8787`
- `MCP_DOCKER_IMAGE`: 可选，Docker 镜像名，默认 `python:3.11-slim`
- `MCP_DOCKER_INSTALL_DEPS`: 可选，`true|false`，默认自动判断
- `MCP_ENV_FILE`: 可选，指定环境变量文件路径；若未设置则读取项目根目录 `.env`
- `MCP_LOG_LEVEL`: 可选，控制日志级别，可选值为 `DEBUG`, `INFO`, `WARNING`, `ERROR`，默认 `INFO`

## 环境变量文件示例

在项目根目录创建 `.env`（或自定义路径并通过 `MCP_ENV_FILE` 指定）：

```dotenv
OPENAI_API_KEY=你的Key
# 如有自建网关：
# OPENAI_BASE_URL=https://你的OpenAI兼容网关
OPENAI_MODEL=gpt-4o-mini
MCP_DOCKER_IMAGE=mathviz-runner:latest
MCP_DOCKER_INSTALL_DEPS=false
# 可自定义 SSE 和静态服务监听地址：
# MCP_SSE_HOST=127.0.0.1
# MCP_SSE_PORT=8787
# 可调整日志级别
# MCP_LOG_LEVEL=DEBUG
```

也可以使用其他路径的 env 文件并设置：

```bash
set MCP_ENV_FILE=E:\\secrets\\mathviz.env
```

## 运行 MCP 服务

MathViz 服务器支持两种 MCP 传输协议：

- **SSE (Server-Sent Events)**: 推荐用于网络客户端，如网页、VS Code 扩展等。
- **stdio (Standard I/O)**: 用于本地客户端，如 Claude Desktop App。

服务器会优先尝试以 SSE 模式启动。如果缺少必要的 HTTP 库（如 `fastapi`, `uvicorn`），它会自动降级到 stdio 模式。

### 以 SSE 模式运行

这是默认和推荐的运行方式。服务器会启动一个 HTTP 服务，同时提供 SSE 端点和静态图片托管。

```bash
# 在 MathViz 目录
python -m server.main
```

- **SSE 端点**:
  - `GET /sse`: 客户端通过此端点建立 SSE 连接。
  - `POST /messages`: 客户端向此端点发送 JSON-RPC 消息。
- **静态图片**:
  - `GET /images/{image-id}.png`: 访问生成的图片。
- **监听地址**: 由环境变量 `MCP_SSE_HOST` 和 `MCP_SSE_PORT` 控制，默认为 `127.0.0.1:8787`。

### 以 stdio 模式运行

如果环境中未安装 `fastapi` 和 `uvicorn`，服务器会自动回退到 stdio 模式。你也可以通过卸载这些包来强制使用 stdio。

```bash
# 确保 fastapi 和 uvicorn 未安装
pip uninstall fastapi uvicorn
# 运行服务器
python -m server.main
```

## 日志记录

服务器会将运行日志输出到标准错误流（stderr），以便于调试和监控。日志内容包括服务器启动、SSE 连接、工具调用、Docker 执行等关键事件。

可以通过 `MCP_LOG_LEVEL` 环境变量控制日志的详细程度。例如，要查看更详细的 `DEBUG` 级别日志：

```bash
# Windows (CMD)
set MCP_LOG_LEVEL=DEBUG
python -m server.main

# Windows (PowerShell)
$env:MCP_LOG_LEVEL="DEBUG"
python -m server.main

# Linux / macOS
MCP_LOG_LEVEL=DEBUG python -m server.main
```

## 工具

- `render_plot(problem: str)`
  - 入参：题干文本
  - 出参：`{"url": <图片链接>, "note": <信息>}` 或 `{"error": <错误信息>}`

## Windows 注意事项

- Docker 路径挂载在 Linux 容器下会被转换为 `/e/...` 的格式，代码中已处理。
- 请确保 Docker Desktop 共享了对应的盘符（Settings -> Resources -> File Sharing）。

## 安全说明

- 运行容器时禁用网络、限制 CPU/内存/进程数。
- 对 LLM 生成代码做了关键字拦截与强制使用 Agg 后端，并默认追加 `savefig` 调用。
