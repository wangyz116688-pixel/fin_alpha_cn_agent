"""Start the AlphaAgent API and frontend together for local demonstrations."""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def wait_port(port: int, timeout: int = 40) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as sock:
            sock.settimeout(.4)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(.4)
    return False


def npm_command() -> list[str]:
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if not npm and Path("D:/npm.cmd").exists():
        npm = "D:/npm.cmd"
    if not npm:
        raise RuntimeError("未找到 npm，请先安装 Node.js 22 或更高版本。")
    return ["cmd.exe", "/d", "/s", "/c", npm, "run", "dev"] if os.name == "nt" else [npm, "run", "dev"]


def main() -> int:
    python = sys.executable
    child_env = os.environ.copy()
    bundled_node = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "node" / "bin"
    if bundled_node.exists():
        child_env["PATH"] = f"{bundled_node}{os.pathsep}{child_env.get('PATH', '')}"
    if importlib.util.find_spec("fastapi") and importlib.util.find_spec("uvicorn"):
        api_cmd = [python, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"]
    else:
        api_cmd = [python, str(ROOT / "backend" / "demo_server.py")]
        print("提示：当前环境未安装 FastAPI，已启用内置兼容服务；接口与页面功能不受影响。")
    processes = [
        subprocess.Popen(api_cmd, cwd=ROOT),
        subprocess.Popen(npm_command(), cwd=ROOT / "frontend", env=child_env),
    ]
    try:
        if not wait_port(8000): raise RuntimeError("策略服务启动超时。")
        if not wait_port(3000): raise RuntimeError("前端服务启动超时。")
        print("\nAlphaAgent 已启动：http://127.0.0.1:3000")
        print("按 Ctrl+C 可同时关闭前端与后端。\n")
        webbrowser.open("http://127.0.0.1:3000")
        while all(process.poll() is None for process in processes):
            time.sleep(1)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        for process in processes:
            if process.poll() is None: process.terminate()
        for process in processes:
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
