import os
import re
import sys
import json
import uuid
import shlex
import asyncio
import tempfile
import threading
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple

from mcp.server.fastmcp import FastMCP
try:
    from mcp.server.sse import SseServerTransport  # legacy SSE transport
except Exception:
    SseServerTransport = None  # type: ignore


# HTTP server (SSE + static images)
try:
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    from starlette.routing import ASGIRoute
except Exception:
    FastAPI = None  # type: ignore
    StaticFiles = None  # type: ignore
    uvicorn = None  # type: ignore
    ASGIRoute = None  # type: ignore


# Optional OpenAI-compatible client
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


SERVER_NAME = "MathViz Code Runner"
mcp = FastMCP(SERVER_NAME)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = PROJECT_ROOT / "public"
IMAGES_DIR = PUBLIC_DIR / "images"

_http_server_started = False
_http_server_lock = threading.Lock()

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # type: ignore


def load_env_file() -> None:
    """Load environment variables from a dedicated env file.

    Priority: MCP_ENV_FILE path > PROJECT_ROOT/.env. Variables already set in
    process env will not be overridden.
    """
    # Prefer explicit file path
    env_path = os.environ.get("MCP_ENV_FILE")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(PROJECT_ROOT / ".env")

    for candidate in candidates:
        try:
            if candidate and Path(candidate).exists():
                if load_dotenv:
                    load_dotenv(dotenv_path=str(candidate), override=False)
                else:
                    # Minimal parser fallback
                    for line in Path(candidate).read_text(encoding="utf-8").splitlines():
                        raw = line.strip()
                        if not raw or raw.startswith("#"):
                            continue
                        if "=" in raw:
                            key, value = raw.split("=", 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            os.environ.setdefault(key, value)
                break
        except Exception:
            # Ignore parse/load errors to not block server start
            continue


# Load env file before reading any env-based config
load_env_file()

STATIC_HOST = os.environ.get("MCP_ASSETS_HOST", "127.0.0.1")
STATIC_PORT = int(os.environ.get("MCP_ASSETS_PORT", "8787"))

# Reuse the same host/port for SSE endpoints
SSE_HOST = os.environ.get("MCP_SSE_HOST", STATIC_HOST)
SSE_PORT = int(os.environ.get("MCP_SSE_PORT", str(STATIC_PORT)))


def ensure_dirs() -> None:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def create_http_app() -> Optional[object]:
    if FastAPI is None or StaticFiles is None:
        return None

    ensure_dirs()
    app = FastAPI(title=f"{SERVER_NAME} (SSE MCP Server)")
    app.mount("/images", StaticFiles(directory=str(IMAGES_DIR), html=False), name="images")

    # Legacy SSE transport endpoints
    if SseServerTransport is not None and ASGIRoute is not None:
        sse_transport = SseServerTransport("/messages")

        async def handle_sse(scope, receive, send):
            async with sse_transport.connect_sse(scope, receive, send) as streams:
                # FastMCP implements the same run signature as low-level Server
                await mcp.run(streams[0], streams[1], mcp.create_initialization_options())

        async def handle_messages(scope, receive, send):
            await sse_transport.handle_post_message(scope, receive, send)

        # Mount ASGI endpoints using Starlette ASGIRoute (ASGI app signature)
        app.router.routes.append(ASGIRoute("/sse", handle_sse))
        app.router.routes.append(ASGIRoute("/messages", handle_messages, methods=["POST"]))

    return app


def start_http_server_if_needed() -> None:
    global _http_server_started

    if FastAPI is None or uvicorn is None or StaticFiles is None:
        # HTTP server deps missing; links will be file:// URLs
        return

    with _http_server_lock:
        if _http_server_started:
            return

        app = create_http_app()
        if app is None:
            return

        def run_server() -> None:
            config = uvicorn.Config(app=app, host=SSE_HOST, port=SSE_PORT, log_level="warning")
            server = uvicorn.Server(config)
            server.run()

        thread = threading.Thread(target=run_server, name="mcp-sse-http", daemon=True)
        thread.start()
        _http_server_started = True


def to_docker_mount_path(local_path: Path) -> str:
    """Return a path suitable for Docker -v source on current OS.

    On Windows, Docker accepts native paths like E:\\path. On Unix, use POSIX.
    """
    p = local_path.resolve()
    if os.name == "nt":
        return str(p)
    return p.as_posix()


def extract_code_from_markdown(text: str) -> str:
    # Prefer ```python ... ``` blocks
    match = re.search(r"```python\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: any triple backtick
    match = re.search(r"```\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text.strip()


def sanitize_code(code: str) -> Tuple[bool, Optional[str]]:
    banned_substrings = [
        "import os",
        "import sys",
        "import subprocess",
        "from os",
        "from sys",
        "from subprocess",
        "open(",  # arbitrary file I/O
        "__import__",
        "eval(",
        "exec(",
        "socket",
        "requests",
        "urllib",
        "shutil",
        "pathlib.Path.open",
        "input(",
    ]
    lowered = code.lower()
    for token in banned_substrings:
        if token in lowered:
            return False, f"Detected banned token: {token}"
    return True, None


def wrap_code_for_matplotlib(code: str, output_filename: str) -> str:
    header = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import math\n"
        "\n"
        f"OUTPUT_PATH = r'{output_filename}'\n"
    )

    footer_lines = []
    # Ensure the script saves a figure; if not, we append a default save
    if re.search(r"savefig\s*\(", code) is None:
        footer_lines.append("plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight')")
    footer_lines.append("plt.close('all')")
    footer = "\n".join(footer_lines) + "\n"

    return f"{header}\n{code}\n\n{footer}"


def run_code_in_docker(workdir: Path, script_name: str = "main.py") -> Tuple[bool, str]:
    mount = to_docker_mount_path(workdir)
    image = os.environ.get("MCP_DOCKER_IMAGE", "python:3.11-slim")
    install_deps = os.environ.get("MCP_DOCKER_INSTALL_DEPS")
    needs_install = bool(install_deps == "true" or image == "python:3.11-slim")

    if needs_install:
        inner_cmd = "pip -q install matplotlib numpy >/dev/null 2>&1 && python -u main.py"
    else:
        inner_cmd = "python -u main.py"

    args = [
        "docker", "run", "--rm",
        "-v", f"{mount}:/work",
        "-w", "/work",
        "--network", "none",
        "--pids-limit", "128",
        "--cpus", "1",
        "-m", "512m",
        image,
        "bash", "-lc", inner_cmd,
    ]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=180)
        ok = proc.returncode == 0
        logs = proc.stdout + "\n" + proc.stderr
        return ok, logs
    except subprocess.TimeoutExpired:
        return False, "Execution timed out in Docker"
    except Exception as e:
        return False, f"Docker execution error: {e}"


def move_file_across_drives(src: Path, dst: Path) -> None:
    """Move file from src to dst safely across different drives.

    On Windows, Path.replace (os.replace) fails across drives. Fallback to copy2+unlink.
    """
    try:
        src.replace(dst)
    except OSError:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        try:
            if src.exists():
                src.unlink()
        except Exception:
            # Best-effort cleanup; ignore failures
            pass


def call_llm_for_code(problem: str) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Please install dependencies.")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    base_url = os.environ.get("OPENAI_BASE_URL")  # optional, for compatible endpoints
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)  # type: ignore

    system_prompt = (
        "你是一个将数学/物理题目转为可视化代码的助手。"
        "请输出纯 Python 代码，使用 matplotlib 和 numpy 实现绘图。"
        "严格约束：\n"
        "- 必须使用 matplotlib 非交互后端（我们会注入 Agg）\n"
        "- 不允许网络访问、文件读取写入（除保存图像）\n"
        "- 避免 plt.show()，最后保存图片到变量 OUTPUT_PATH 所指路径\n"
        "- 代码尽量自包含，可运行，包含必要的示例数据/函数\n"
        "- 中文字体优先使用文泉驿正黑、文泉驿微米黑和Noto CJK字体\n"
        "- transparent 参数设置为 True\n"
        "- 不要在图片中显示题干和答案\n"
    )
    user_prompt = (
        f"题干：\n{problem}\n\n"
        "要求：生成 matplotlib 绘图代码，最终调用 plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight') 保存图片。"
    )

    completion = client.chat.completions.create(  # type: ignore
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = completion.choices[0].message.content or ""
    return extract_code_from_markdown(content)


def generate_image_from_problem(problem: str) -> Tuple[str, str]:
    """Generate an image from problem statement.

    Returns: (url, logs)
    """
    ensure_dirs()
    start_http_server_if_needed()

    # 1) LLM -> code
    code = call_llm_for_code(problem)

    ok, reason = sanitize_code(code)
    if not ok:
        raise RuntimeError(f"生成的代码不安全：{reason}")

    # 2) Wrap + run in Docker
    image_id = str(uuid.uuid4()).replace("-", "")
    output_filename = "out.png"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        script_path = tmp_path / "main.py"
        wrapped = wrap_code_for_matplotlib(code, output_filename)
        script_path.write_text(wrapped, encoding="utf-8")

        ok, logs = run_code_in_docker(tmp_path)
        out_file = tmp_path / output_filename
        if not ok:
            raise RuntimeError(f"Docker 运行失败：\n{logs}")
        if not out_file.exists():
            raise RuntimeError(f"未找到输出图片：{output_filename}\n日志：\n{logs}")

        # 3) Move to public and build URL (support cross-drive move on Windows)
        target_file = IMAGES_DIR / f"{image_id}.png"
        move_file_across_drives(out_file, target_file)

    # Prefer HTTP URL if server is running and deps installed; otherwise file URL
    if FastAPI is not None and uvicorn is not None:
        url = f"http://{SSE_HOST}:{SSE_PORT}/images/{image_id}.png"
    else:
        url = Path(target_file).resolve().as_uri()

    return url, "生成成功"


@mcp.tool()
async def render_plot(problem: str) -> dict:
    """根据题干生成 matplotlib 图像并返回可访问链接。

    参数：
    - problem: 题干文本
    返回：{"url": 链接, "note": 信息}
    """
    try:
        loop = asyncio.get_event_loop()
        url, note = await loop.run_in_executor(None, generate_image_from_problem, problem)
        return {"url": url, "note": note}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # If FastAPI/uvicorn available, run HTTP server hosting both SSE endpoints and static images
    if FastAPI is not None and uvicorn is not None and StaticFiles is not None:
        app = create_http_app()
        assert app is not None
        uvicorn.run(app, host=SSE_HOST, port=SSE_PORT, log_level="info")
    else:
        # Fallback: run stdio (for environments without HTTP stack)
        mcp.run()


