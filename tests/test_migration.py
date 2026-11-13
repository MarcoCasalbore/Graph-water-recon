import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.maintenance.migrate_workspace import atomic_write, member, rebase, verify


class MigrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.old = "/old/project"
        values = {
            "final_water/config.json": json.dumps({"path": self.old + "/data/net3", "score": 0.125,
                                                  "other": "/old/project-other/file", "note": "from /old/project"}),
            "final_water/rows.csv": f"run_dir,mse\n{self.old}/run,0.012500\n",
            "data/values.bin": "unchanged numerical artifact",
        }
        entries = []
        for name, value in values.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value.encode())
            entries.append({"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        self.manifest = {"schema_version": 1, "source_project_root": self.old, "files": entries}
        self.path = self.root / "manifest.json"
        self.path.write_text(json.dumps(self.manifest))

    def test_preview_and_apply_preserve_values(self):
        before = (self.root / "final_water/config.json").read_bytes()
        self.assertEqual(rebase(self.root, self.path)["files_to_rebase"], 2)
        self.assertEqual((self.root / "final_water/config.json").read_bytes(), before)
        rebase(self.root, self.path, apply=True)
        value = json.loads((self.root / "final_water/config.json").read_text())
        self.assertEqual(value["path"], self.root.as_posix() + "/data/net3")
        self.assertEqual(value["score"], 0.125)
        self.assertEqual(value["other"], "/old/project-other/file")
        self.assertEqual(value["note"], "from /old/project")
        self.assertIn(",0.012500", (self.root / "final_water/rows.csv").read_text())
        verify(self.root, json.loads(self.path.with_name("manifest-relocated.json").read_text()))

    def test_corrupt_transfer_blocks_writes(self):
        (self.root / "data/values.bin").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "verification failed"):
            rebase(self.root, self.path, apply=True)
        self.assertFalse(self.path.with_name("manifest-relocated.json").exists())

    def test_json_numeric_tokens_are_preserved_exactly(self):
        path = self.root / "final_water/precise.json"
        text = '{"path":"/old/project/run", "precise":1.234567890123456789, "small":1e-400, "large":1e400, "/old/project/key":2}'
        path.write_text(text)
        self.manifest["files"].append({"path": "final_water/precise.json", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        self.path.write_text(json.dumps(self.manifest))
        rebase(self.root, self.path, apply=True)
        self.assertEqual(path.read_text(), text.replace('"/old/project/run"', json.dumps(self.root.as_posix() + "/run", ensure_ascii=False)))

    def test_unsafe_and_duplicate_paths_are_rejected(self):
        for name in ["../outside", "/absolute", "C:/outside", "a\\b"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                member(self.root, name)
        self.manifest["files"].append(self.manifest["files"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            verify(self.root, self.manifest)

    def test_partial_failure_restores_originals(self):
        before = (self.root / "final_water/config.json").read_bytes()
        calls = 0

        def fail_once(path, content):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated disk failure")
            atomic_write(path, content)

        with patch("scripts.maintenance.migrate_workspace.atomic_write", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "simulated disk failure"):
                rebase(self.root, self.path, apply=True)
        self.assertEqual((self.root / "final_water/config.json").read_bytes(), before)
        verify(self.root, self.manifest)


if __name__ == "__main__":
    unittest.main()
