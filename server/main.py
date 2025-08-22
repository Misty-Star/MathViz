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
import logging
import time
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
except Exception:
    FastAPI = None  # type: ignore
    StaticFiles = None  # type: ignore
    uvicorn = None  # type: ignore


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
# When running main FastAPI/uvicorn, prevent background HTTP from starting again
_primary_http_mode = False
_cleanup_started = False
_cleanup_lock = threading.Lock()

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

# Public URL configuration for external access
# This allows binding to 0.0.0.0 while returning proper public URLs
PUBLIC_HOST = os.environ.get("MCP_PUBLIC_HOST")  # Optional: override for public URLs
PUBLIC_PORT = os.environ.get("MCP_PUBLIC_PORT")  # Optional: override for public port
# Alternative: full public URL base (e.g., "https://your-domain.com" or "http://1.2.3.4:8787")
PUBLIC_BASE_URL = os.environ.get("MCP_PUBLIC_BASE_URL")

# Image export format and cleanup configuration
_allowed_formats = {"png", "svg", "jpg", "jpeg", "pdf"}
IMAGE_FORMAT = os.environ.get("MCP_IMAGE_FORMAT", "png").lower().strip()
if IMAGE_FORMAT not in _allowed_formats:
    IMAGE_FORMAT = "png"

IMAGES_RETENTION_DAYS = int(os.environ.get("MCP_IMAGES_RETENTION_DAYS", "7"))
IMAGES_CLEAN_INTERVAL_SEC = int(os.environ.get("MCP_IMAGES_CLEAN_INTERVAL_SEC", "3600"))

# Optional: write images directly into public/images from inside Docker
DIRECT_SAVE_IMAGES = os.environ.get("MCP_DIRECT_SAVE_IMAGES", "false").strip().lower() == "true"
DOCKER_IMAGES_MOUNT = "/images"

# Configure logging (stderr only, safe for stdio transport)
logger = logging.getLogger("mathviz")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
_level = os.environ.get("MCP_LOG_LEVEL", "INFO").upper()
try:
    logger.setLevel(getattr(logging, _level, logging.INFO))
except Exception:
    logger.setLevel(logging.INFO)
logger.info(
    "MathViz starting with HTTP host=%s port=%s (SSE host=%s port=%s)",
    STATIC_HOST,
    STATIC_PORT,
    SSE_HOST,
    SSE_PORT,
)


def ensure_dirs() -> None:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def create_http_app() -> Optional[object]:
    if FastAPI is None or StaticFiles is None:
        logger.warning("FastAPI/StaticFiles not available; HTTP/SSE server disabled")
        return None

    ensure_dirs()
    app = FastAPI(title=f"{SERVER_NAME} (SSE MCP Server)")
    app.mount("/images", StaticFiles(directory=str(IMAGES_DIR), html=False), name="images")
    logger.info("Mounted static images at /images -> %s", IMAGES_DIR)

    # Mount FastMCP's official SSE Starlette app to avoid transport mismatches
    try:
        starlette_sse_app = mcp.sse_app()
        app.mount("/", starlette_sse_app)
        logger.info("SSE endpoints enabled: GET /sse, POST /messages")
    except Exception as e:
        logger.warning("Failed to mount SSE app: %s", e)

    return app


def start_http_server_if_needed() -> None:
    global _http_server_started
    global _primary_http_mode

    # If the primary HTTP server is already running (main uvicorn), do nothing
    if _primary_http_mode:
        return

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
            logger.info("Starting HTTP server on http://%s:%s", SSE_HOST, SSE_PORT)
            config = uvicorn.Config(app=app, host=SSE_HOST, port=SSE_PORT, log_level="info")
            server = uvicorn.Server(config)
            server.run()

        thread = threading.Thread(target=run_server, name="mcp-sse-http", daemon=True)
        thread.start()
        _http_server_started = True

    # Ensure background cleanup is running regardless of HTTP/SSE availability
    start_cleanup_thread_if_needed()


def cleanup_old_images(retention_days: int) -> int:
    """Delete images in IMAGES_DIR older than retention_days.

    Returns the number of files removed.
    """
    ensure_dirs()
    now_ts = time.time()
    cutoff = now_ts - max(retention_days, 0) * 24 * 3600
    removed = 0
    try:
        for file in IMAGES_DIR.iterdir():
            if not file.is_file():
                continue
            # Only consider known image extensions
            suffix = file.suffix.lower().lstrip(".")
            if suffix not in _allowed_formats:
                continue
            try:
                if file.stat().st_mtime < cutoff:
                    file.unlink(missing_ok=True)  # type: ignore[arg-type]
                    removed += 1
            except Exception:
                # Ignore individual file errors
                continue
    except Exception as e:
        logger.warning("Image cleanup scan failed: %s", e)
    if removed:
        logger.info("Image cleanup removed %s old files", removed)
    return removed


def _cleanup_worker() -> None:
    while True:
        try:
            cleanup_old_images(IMAGES_RETENTION_DAYS)
        except Exception as e:
            logger.warning("Cleanup worker error: %s", e)
        time.sleep(max(IMAGES_CLEAN_INTERVAL_SEC, 60))


def start_cleanup_thread_if_needed() -> None:
    global _cleanup_started
    with _cleanup_lock:
        if _cleanup_started:
            return
        thread = threading.Thread(target=_cleanup_worker, name="images-cleanup", daemon=True)
        thread.start()
        _cleanup_started = True


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
            logger.warning("Code rejected by sanitizer: token=%s", token)
            return False, f"Detected banned token: {token}"
    return True, None


def wrap_code_for_matplotlib(code: str, output_filename: str) -> str:
    header = (
        "import sys\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "from matplotlib.figure import Figure\n"
        "import numpy as np\n"
        "import math\n"
        "\n"
        f"OUTPUT_PATH = r'{output_filename}'\n"
        "\n"
        "# Add debug prints to help diagnose issues\n"
        "print(f'Starting execution, output path: {OUTPUT_PATH}', file=sys.stderr)\n"
        "\n"
        "# Force all save operations to write to OUTPUT_PATH, preserving common kwargs\n"
        "_ORIG_PLT_SAVEFIG = plt.savefig\n"
        "def _FORCED_SAVEFIG(*args, **kwargs):\n"
        "    kwargs.setdefault('dpi', 200)\n"
        "    kwargs.setdefault('bbox_inches', 'tight')\n"
        "    kwargs.setdefault('transparent', True)\n"
        "    print(f'Saving plot to: {OUTPUT_PATH}', file=sys.stderr)\n"
        "    return _ORIG_PLT_SAVEFIG(OUTPUT_PATH, **kwargs)\n"
        "plt.savefig = _FORCED_SAVEFIG\n"
        "\n"
        "_ORIG_FIG_SAVEFIG = Figure.savefig\n"
        "def _FORCED_FIG_SAVEFIG(self, *args, **kwargs):\n"
        "    kwargs.setdefault('dpi', 200)\n"
        "    kwargs.setdefault('bbox_inches', 'tight')\n"
        "    kwargs.setdefault('transparent', True)\n"
        "    print(f'Saving figure to: {OUTPUT_PATH}', file=sys.stderr)\n"
        "    return _ORIG_FIG_SAVEFIG(self, OUTPUT_PATH, **kwargs)\n"
        "Figure.savefig = _FORCED_FIG_SAVEFIG\n"
        "\n"
        "# Wrap execution in try-catch for better error reporting\n"
        "try:\n"
    )

    footer_lines = []
    # Ensure the script saves a figure; if not, we append a default save
    if re.search(r"savefig\s*\(", code) is None:
        footer_lines.append("    print('No savefig found in code, adding default save', file=sys.stderr)")
        footer_lines.append("    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight', transparent=True)")
    footer_lines.append("")
    footer_lines.append("except Exception as e:")
    footer_lines.append("    print(f'ERROR during execution: {e}', file=sys.stderr)")
    footer_lines.append("    import traceback")
    footer_lines.append("    traceback.print_exc(file=sys.stderr)")
    footer_lines.append("    sys.exit(1)")
    footer_lines.append("")
    footer_lines.append("import os")
    footer_lines.append("if os.path.exists(OUTPUT_PATH):")
    footer_lines.append("    print(f'File saved successfully: {OUTPUT_PATH} (size: {os.path.getsize(OUTPUT_PATH)} bytes)', file=sys.stderr)")
    footer_lines.append("else:")
    footer_lines.append("    print(f'ERROR: File not created: {OUTPUT_PATH}', file=sys.stderr)")
    footer_lines.append("    sys.exit(1)")
    footer_lines.append("plt.close('all')")
    footer = "\n".join(footer_lines) + "\n"

    return f"{header}\n\n{chr(10).join('    ' + line for line in code.split(chr(10)))}\n\n{footer}"


def run_code_in_docker(workdir: Path, script_name: str = "main.py") -> Tuple[bool, str]:
    mount = to_docker_mount_path(workdir)
    image = os.environ.get("MCP_DOCKER_IMAGE", "python:3.11-slim")
    install_deps = os.environ.get("MCP_DOCKER_INSTALL_DEPS")
    needs_install = bool(install_deps == "true" or image == "python:3.11-slim")

    logger.info(
        "Running code in Docker image=%s install_deps=%s", image, str(needs_install)
    )

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
    ]

    # Optionally mount images directory for direct save
    if DIRECT_SAVE_IMAGES:
        images_mount = to_docker_mount_path(IMAGES_DIR)
        args.extend(["-v", f"{images_mount}:{DOCKER_IMAGES_MOUNT}"])

    args.extend([
        image,
        "bash", "-lc", inner_cmd,
    ])
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=180)
        ok = proc.returncode == 0
        logs = proc.stdout + "\n" + proc.stderr
        if ok:
            logger.info("Docker execution finished successfully")
        else:
            logger.error("Docker execution failed (code=%s)", proc.returncode)
        return ok, logs
    except subprocess.TimeoutExpired:
        logger.error("Docker execution timed out")
        return False, "Execution timed out in Docker"
    except Exception as e:
        logger.exception("Docker execution error: %s", e)
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
    logger.info("LLM call completed: model=%s base_url=%s", model_name, base_url or "default")
    return extract_code_from_markdown(content)


def generate_image_from_problem(problem: str) -> Tuple[str, str]:
    """Generate an image from problem statement.

    Returns: (url, logs)
    """
    ensure_dirs()
    start_http_server_if_needed()

    # 1) LLM -> code
    logger.info("Generating code for problem (len=%s)", len(problem))
    code = call_llm_for_code(problem)

    ok, reason = sanitize_code(code)
    if not ok:
        raise RuntimeError(f"生成的代码不安全：{reason}")

    # 2) Wrap + run in Docker
    image_id = str(uuid.uuid4()).replace("-", "")
    ext = "jpeg" if IMAGE_FORMAT == "jpg" else IMAGE_FORMAT

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        script_path = tmp_path / "main.py"
        # Decide where the script should save: temp file or mounted images dir
        if DIRECT_SAVE_IMAGES:
            # Container path where images dir is mounted
            output_path_in_container = f"{DOCKER_IMAGES_MOUNT}/{image_id}.{ext}"
            target_file = IMAGES_DIR / f"{image_id}.{ext}"
        else:
            output_path_in_container = f"out.{ext}"
            target_file = IMAGES_DIR / f"{image_id}.{ext}"

        wrapped = wrap_code_for_matplotlib(code, output_path_in_container)
        script_path.write_text(wrapped, encoding="utf-8")

        ok, logs = run_code_in_docker(tmp_path)
        if not ok:
            raise RuntimeError(f"Docker 运行失败：\n{logs}")
        
        if DIRECT_SAVE_IMAGES:
            # Image should already be written to target_file via Docker mount
            if not target_file.exists():
                raise RuntimeError(
                    f"未在公共目录找到输出图片：{target_file.name}\n日志：\n{logs}"
                )
            logger.info("Image generated directly: id=%s path=%s", image_id, target_file)
        else:
            # Image saved inside temp workdir; move into public/images
            out_file = tmp_path / output_path_in_container
            if not out_file.exists():
                raise RuntimeError(
                    f"未找到输出图片：{output_path_in_container}\n日志：\n{logs}"
                )
            move_file_across_drives(out_file, target_file)
            logger.info("Image generated: id=%s path=%s", image_id, target_file)

    # Generate URL for accessing the image
    if FastAPI is not None and uvicorn is not None:
        # Determine the public URL to return
        if PUBLIC_BASE_URL:
            # Use full base URL if provided (e.g., "https://domain.com" or "http://1.2.3.4:8787")
            base_url = PUBLIC_BASE_URL.rstrip('/')
            url = f"{base_url}/images/{image_id}.{ext}"
        elif PUBLIC_HOST:
            # Use separate public host/port if provided
            public_port = int(PUBLIC_PORT) if PUBLIC_PORT else SSE_PORT
            url = f"http://{PUBLIC_HOST}:{public_port}/images/{image_id}.{ext}"
        else:
            # Default: use the same host/port as server binding
            url = f"http://{SSE_HOST}:{SSE_PORT}/images/{image_id}.{ext}"
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
        logger.info("Tool render_plot invoked")
        # Make sure background services (HTTP and cleanup) are running
        start_http_server_if_needed()
        start_cleanup_thread_if_needed()
        loop = asyncio.get_event_loop()
        url, note = await loop.run_in_executor(None, generate_image_from_problem, problem)
        logger.info("render_plot success: %s", url)
        return {"url": url, "note": note}
    except Exception as e:
        logger.exception("render_plot error: %s", e)
        return {"error": str(e)}


if __name__ == "__main__":
    # If FastAPI/uvicorn available, run HTTP server hosting both SSE endpoints and static images
    if FastAPI is not None and uvicorn is not None and StaticFiles is not None:
        app = create_http_app()
        assert app is not None
        _primary_http_mode = True
        # Start background cleanup in primary HTTP mode as well
        start_cleanup_thread_if_needed()
        uvicorn.run(app, host=SSE_HOST, port=SSE_PORT, log_level="info")
    else:
        # Fallback: run stdio (for environments without HTTP stack)
        logger.info("Starting server in stdio mode")
        start_cleanup_thread_if_needed()
        mcp.run()


