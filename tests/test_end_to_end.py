import json
import tempfile
import unittest
from pathlib import Path

from spark_eval import checks, cli, filler
from spark_eval.client import ChatResult
from tests import fake_server


class EndToEnd(unittest.TestCase):
    def run_cli(self, port, mode, gates, extra=()):
        srv = fake_server.serve(port, mode)
        tmp = tempfile.mkdtemp()
        try:
            code = cli.main(["run", "--label", mode, "--base-url", f"http://127.0.0.1:{port}/v1", "--model", "fake",
                             "--results", tmp, "--gates", gates, "--keep-going", "--tool-contexts", "0,2000",
                             "--longctx-contexts", "2000", *extra])
        finally:
            srv.shutdown()
            srv.server_close()
        run = sorted(Path(tmp).glob("*/*/*"))[-1]
        return code, {p.stem: json.loads(p.read_text()) for p in run.glob("*.json")}

    def test_good_server_passes_smoke(self):
        _, out = self.run_cli(18741, "good", "bringup,smoke")
        self.assertTrue(out["bringup"]["passed"])
        self.assertEqual(out["smoke"]["failures"], [], [r for r in out["smoke"]["rows"] if r["status"] != "pass"])
        self.assertEqual(out["manifest"]["server_version"], "fake-0.0")

    def test_leaky_parser_is_counted_as_malformed(self):
        code, out = self.run_cli(18742, "leaky", "tools")
        self.assertEqual(code, 1)
        self.assertGreater(out["tools"]["by_context"]["0"]["malformed_rate"], 0.5)
        self.assertIn("2000", out["tools"]["by_context"])
        self.assertEqual(set(out["tools"]["by_stream"]), {"blocking", "streamed"})

    def test_artifact_changes_the_result_folder(self):
        from spark_eval import manifest
        from spark_eval.client import Client
        d = Path(tempfile.mkdtemp())
        (d / "r.yaml").write_text("model: a/b\ncontainer: img@sha256:1\nmodel_revision: abc\n")
        (d / "parser.py").write_text("v1")
        c = Client("http://127.0.0.1:9/v1", "x")
        h1 = manifest.build(c, "t", "lab", d / "r.yaml", None, None, None, d, None, [d / "parser.py"])
        (d / "parser.py").write_text("v2")
        h2 = manifest.build(c, "t", "lab", d / "r.yaml", None, None, None, d, None, [d / "parser.py"])
        self.assertNotEqual(h1["recipe"]["sha256"], h2["recipe"]["sha256"])
        self.assertEqual(h1["image"], "img@sha256:1")

    def test_perf_runs(self):
        _, out = self.run_cli(18743, "good", "perf", ["--mode", "exclusive", "--input-lens", "500", "--concurrency", "1,2",
                                                       "--requests-per-worker", "2", "--probe-tokens", "500"])
        self.assertEqual(len(out["perf"]["cells"]), 2)
        self.assertEqual(out["perf"]["cells"][1]["errors"], 0)
        self.assertIsNotNone(out["perf"]["prefix_cache_probe"]["ttft_warm"])


class Units(unittest.TestCase):
    TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}}}]

    def call(self, args):
        return ChatResult(tool_calls=[{"function": {"name": "f", "arguments": args}}], finish_reason="tool_calls")

    def test_malformed(self):
        self.assertEqual(checks.malformed_reasons(self.call('{"id": 3}'), self.TOOLS), [])
        self.assertIn("not JSON", checks.malformed_reasons(self.call('{"id": 3'), self.TOOLS)[0])
        self.assertIn("schema", checks.malformed_reasons(self.call('{"id": "3"}'), self.TOOLS)[0])
        self.assertIn("leaked", checks.malformed_reasons(ChatResult(content="<think>hm</think> hi"), [])[0])

    def test_filler_places_needles_and_is_seeded(self):
        a = filler.build(800, seed=1, needles=[(0.5, "NEEDLE-X")])
        self.assertIn("NEEDLE-X", a)
        self.assertEqual(a, filler.build(800, seed=1, needles=[(0.5, "NEEDLE-X")]))
        self.assertNotEqual(a, filler.build(800, seed=2, needles=[(0.5, "NEEDLE-X")]))

    def test_exec_check(self):
        ok = "```python\ndef f(x):\n    return x + 1\n```"
        self.assertIsNone(checks.run_exec({"lang": "python", "test": "assert f(1) == 2"}, ok))
        self.assertIn("failed", checks.run_exec({"lang": "python", "test": "assert f(1) == 3"}, ok))


if __name__ == "__main__":
    unittest.main()
