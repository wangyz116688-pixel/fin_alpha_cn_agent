"""Zero-install compatibility server for the local demo API.

FastAPI remains the primary API surface. This small standard-library server makes
the demo runnable on machines where the optional web dependencies are not yet
installed, while using the exact same demo service implementation.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


SERVICE_PATH = Path(__file__).with_name("services") / "demo.py"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SPEC = importlib.util.spec_from_file_location("alphaagent_demo_service", SERVICE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载 Demo Chat 服务。")
demo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = demo
SPEC.loader.exec_module(demo)


def envelope(data=None, message="操作成功", success=True):
    return {"success": success, "message": message, "data": data, "timestamp": datetime.now(timezone.utc).isoformat()}


class Handler(BaseHTTPRequestHandler):
    server_version = "AlphaAgentDemo/1.0"

    def _send(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/demo/overview":
                self._send(200, envelope(demo.get_overview(), "策略概览加载成功"))
                return
            if path.startswith("/api/demo/jobs/"):
                job = demo.get_job(path.rsplit("/", 1)[-1])
                if job is None:
                    self._send(404, envelope(None, "任务不存在或服务已重启。", False))
                else:
                    self._send(200, envelope(job, "任务状态加载成功"))
                return
            self._send(404, envelope(None, "接口不存在", False))
        except Exception as exc:
            self._send(500, envelope(None, str(exc), False))

    def do_POST(self):
        if urlparse(self.path).path != "/api/demo/chat":
            self._send(404, envelope(None, "接口不存在", False)); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            message = str(payload.get("message", "")).strip()
            if not message:
                self._send(400, envelope(None, "message 不能为空", False)); return
            result = demo.chat(message, payload.get("date"))
            self._send(200, envelope(result, "消息处理成功"))
        except ValueError as exc:
            self._send(400, {"detail": str(exc)})
        except Exception as exc:
            self._send(500, {"detail": str(exc)})

    def log_message(self, fmt, *args):
        print(f"[Demo API] {self.address_string()} {fmt % args}")


if __name__ == "__main__":
    host, port = "127.0.0.1", 8000
    print(f"AlphaAgent Demo API: http://{host}:{port}/api/demo/overview")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
