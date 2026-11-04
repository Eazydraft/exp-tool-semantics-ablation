"""Serve the trajectory browser: `./run.sh browse` -> http://localhost:8765/

Static files only, bound to localhost, and only browser/ and runs/ are reachable (the browser fetches
runs/main/index.json and the trajectory files by relative path).
"""

import http.server
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def translate_path(self, path):
        path = path.split("?", 1)[0].split("#", 1)[0]
        if path in ("", "/"):
            path = "/browser/index.html"
        if not path.startswith(("/browser/", "/runs/")):
            return str(ROOT / "browser" / "forbidden")  # a path that does not exist: 404
        return super().translate_path(path)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"Trajectory browser: http://localhost:{port}/  (Ctrl-C to stop)")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
