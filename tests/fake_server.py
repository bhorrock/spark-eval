"""A tiny OpenAI-compatible server for testing the runner without a GPU.

mode=good answers tool requests with a proper tool_calls array.
mode=leaky writes Qwen-style XML into content instead, which is what a
missing or wrong --tool-call-parser looks like from the client side.
"""
from __future__ import annotations

import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODE = "good"


def reply_for(body: dict) -> dict:
    last = next((m for m in reversed(body["messages"]) if m["role"] == "user"), {"content": ""})
    text = last["content"].split("---")[-1]
    tools = body.get("tools") or []
    wants_call = tools and body.get("tool_choice") != "none" and body["messages"][-1]["role"] == "user"
    if wants_call:
        name = tools[0]["function"]["name"]
        num = re.findall(r"\d+", text)
        args = {"record_type": "salesorder", "id": int(num[-1]) if num else 1}
        if MODE == "leaky":
            return {"content": f"<tool_call>\n<function={name}>\n<parameter=id>{args['id']}</parameter>\n</function>\n</tool_call>", "finish": "stop"}
        return {"content": "", "tool_calls": [{"id": "call_x", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}], "finish": "tool_calls"}
    if body.get("response_format"):
        return {"content": json.dumps({"name": "x", "ports": [1, 2], "key": "STC-14338", "priority": "high"}), "finish": "stop"}
    if body.get("max_tokens", 99) <= 8 and not body.get("ignore_eos"):
        return {"content": "Rain falls", "finish": "length"}
    return {"content": "51 ready 9.9 karlaak-staging ZXQ-END 1, 2, 3, 4, 5 Pending Fulfillment", "finish": "stop"}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._json({})
        elif self.path == "/version":
            self._json({"version": "fake-0.0"})
        elif self.path == "/v1/models":
            self._json({"data": [{"id": "fake", "max_model_len": 131072}]})
        elif self.path == "/metrics":
            data = b'# HELP x\nvllm:prefix_cache_hits_total{model_name="fake"} 12.0\nvllm:prefix_cache_queries_total{model_name="fake"} 40.0\n'
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        else:
            self._json({}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            return self._json({"count": max(1, len(body["prompt"]) // 4)})
        rep = reply_for(body)
        usage = {"prompt_tokens": sum(len(str(m.get("content") or "")) for m in body["messages"]) // 4, "completion_tokens": 12}
        if not body.get("stream"):
            msg = {"role": "assistant", "content": rep["content"], "tool_calls": rep.get("tool_calls")}
            return self._json({"choices": [{"message": msg, "finish_reason": rep["finish"]}], "usage": usage})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = []
        for piece in re.findall(r".{1,6}", rep["content"], re.S):
            chunks.append({"choices": [{"delta": {"content": piece}}]})
        for tc in rep.get("tool_calls") or []:
            chunks.append({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": tc["id"], "function": {"name": tc["function"]["name"], "arguments": ""}}]}}]})
            for piece in re.findall(r".{1,5}", tc["function"]["arguments"]):
                chunks.append({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": piece}}]}}]})
        chunks.append({"choices": [{"delta": {}, "finish_reason": rep["finish"]}]})
        chunks.append({"choices": [], "usage": usage})
        for c in chunks:
            self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


def serve(port: int, mode: str = "good") -> ThreadingHTTPServer:
    global MODE
    MODE = mode
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    serve(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "good")
    threading.Event().wait()
