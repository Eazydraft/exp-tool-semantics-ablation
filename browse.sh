#!/usr/bin/env bash
# Serve the trajectory browser at http://localhost:8765/ (only browser/ and runs/ are exposed).
cd "$(dirname "$0")"
PORT="${PORT:-8765}"
echo "Trajectory browser: http://localhost:$PORT/"
python3 - "$PORT" <<'PY'
import sys, http.server, functools

class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        path = path.split("?")[0]
        if path in ("", "/"):
            path = "/browser/index.html"
        if not (path.startswith("/browser/") or path.startswith("/runs/")):
            return "/nonexistent"
        return super().translate_path(path)
    def log_message(self, *a):
        pass

http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), functools.partial(Handler, directory=".")).serve_forever()
PY
