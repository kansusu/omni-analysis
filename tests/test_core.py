"""Run with:  python -m unittest discover -s tests -v   (standard library only, no GPU)."""
import ast
import importlib.util
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = load("run_qwen3_omni_multi")
viz = load("visualize_omni_outputs")
sim = load("omni_sim")

HEAD = 'scene = Scene()\np1 = Person(scene, "p1")\np2 = Person(scene, "p2")\n'


class SafetyFilter(unittest.TestCase):
    def kept(self, snippet):
        code, _, _ = runner.sim_safety(HEAD + snippet)
        return snippet.splitlines()[0] in code

    def test_unsafe_statements_are_removed(self):
        for bad in ("import os\n", "x = Person.do.__globals__\n", "while True:\n    pass\n",
                    "f = lambda: __builtins__['open']\n", "s = '{0.__init__.__globals__}'.format(p1)\n",
                    "T = type('T', (object,), {})\n", "scene.run()\n", "@exec\nclass A: pass\n",
                    "class A(Person):\n    def f(self):\n        open('/etc/passwd')\n"):
            self.assertFalse(self.kept(bad), bad)

    def test_decorator_removal_leaves_no_orphan_line(self):
        code, _, _ = runner.sim_safety(HEAD + "@exec\nclass A: pass\n")
        self.assertNotIn("@", code)

    def test_safe_code_is_kept(self):
        for ok in ("p1.say(0, 2, 'hello', to=p2)\n", "class B(Person):\n    role = 'builder'\n",
                   "from dataclasses import dataclass\n"):
            self.assertTrue(self.kept(ok), ok)


class Postprocess(unittest.TestCase):
    def test_py_fence_is_stripped(self):
        code, note = runner.postprocess("python", "x\n```py\nscene = Scene()\n```")
        self.assertIn("scene = Scene()", code)
        self.assertNotIn("```", code)

    def test_broken_lines_are_salvaged_not_fatal(self):
        raw = "```python\nscene = Scene()\np1 = Person(scene, 'p1')\np1.say(0, 1, 'That's good')\np1.say(2, 3, 'ok')\n```"
        code, note = runner.postprocess("python", raw)
        self.assertTrue(note.startswith("SALVAGED"))
        self.assertIn("'ok'", code)
        ast.parse(code)

    def test_no_scene_is_reported(self):
        _, note = runner.postprocess("python", "```python\nx = 1\n```", True)
        self.assertIn("NO SCENE", note)

    def test_truncated_svg_is_repaired(self):
        out, note = runner.postprocess("svg", "```svg\n<svg xmlns='http://www.w3.org/2000/svg'><g><rect/>", True)
        self.assertTrue(note.startswith("REPAIRED"))
        self.assertTrue(out.strip().endswith("</svg>"))


class Simulation(unittest.TestCase):
    def build(self):
        s = sim.Scene()
        p1, p2, p3 = (sim.Person(s, i) for i in ("p1", "p2", "p3"))
        blocks = sim.Thing(s, "blocks")
        for p in (p1, p2, p3):
            p.look(0, 10, at=blocks)          # joint attention of all three
        p1.look(12, 16, at=p2)
        p2.look(13, 15, at=p1)                # mutual gaze
        p1.gesture(2, 4, "points at blocks")  # target taken from the text
        p3.gesture(3, 5, "points at blocks")
        return s

    def test_derives_joint_and_mutual_events(self):
        kinds = {(type(e).__name__, tuple(sorted(m.id for m in e.members)))
                 for e in self.build().derive_group_events()}
        self.assertIn(("JointAttention", ("p1", "p2", "p3")), kinds)
        self.assertIn(("MutualGaze", ("p1", "p2")), kinds)
        self.assertIn(("JointGesture", ("p1", "p3")), kinds)

    def test_aliases_exist(self):
        p = sim.Person(sim.Scene(), "p1")
        p.gaze(0, 1)
        p.point(0, 1, at=None)

    def test_run_exports_json(self):
        import contextlib, io, json
        s = self.build()
        path = tempfile.mktemp(suffix=".json")
        with contextlib.redirect_stdout(io.StringIO()):
            s.run(clip_s=20, argv=["--json", path])
        data = json.load(open(path))
        self.assertTrue(data["derived_group_events"])
        self.assertEqual(data["participants"][0]["class"], "Person")


class Visualizer(unittest.TestCase):
    def test_python_parser_survives_bad_quoting(self):
        txt = ("participants = [{'id': 'p1', 'description': 'a'}]\nevents = [\n"
               "{'start': 0.0, 'end': 1.0, 'participant': 'p1', 'modality': 'speech', 'content': 'That's good', 'target': None},\n"
               "{'start': 1.0, 'end': 2.0, 'participant': 'p1', 'modality': 'gaze', 'content': 'looks', 'target': None},\n")
        info = viz.parse_python(txt)
        self.assertEqual(len(info["events"]), 2)

    def test_amr_parser_tolerates_broken_graphs(self):
        root, info = viz.parse_amr_graph("(:ARG0 (b / build-01 :ARG1 (b / block-01")
        self.assertIsNotNone(root)
        self.assertGreater(info["missing_close"], 0)

    def test_invisible_svg_paths_do_not_count_as_drawn(self):
        svg = "<svg xmlns='http://www.w3.org/2000/svg'>" + '<path d="M0 0" opacity=".01"/>\n' * 50 + "</svg>"
        self.assertEqual(viz.svg_info(svg)["drawn"], 0)


if __name__ == "__main__":
    unittest.main()
