"""Verify a transfer and rebase project paths in JSON/CSV artifact metadata."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOTS = {"data", "final_water", "final_electric", "baseline_joint", "results_section_v",
                  "results_section_v_final", "results_section_vi", "outputs", "paper_artifacts"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def member(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe manifest path: {name}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes project: {name}")
    if any(p.is_symlink() for p in (path, *path.parents) if p != root and p.is_relative_to(root)):
        raise ValueError(f"Symlink in manifest path: {name}")
    return path


def verify(root: Path, manifest: dict) -> None:
    if manifest.get("schema_version") != 1 or not manifest.get("files"):
        raise ValueError("Expected a nonempty version-1 transfer manifest")
    seen, errors = set(), []
    for entry in manifest["files"]:
        path = member(root, entry["path"])
        if path in seen:
            raise ValueError(f"Duplicate manifest path: {entry['path']}")
        seen.add(path)
        if not path.is_file() or digest(path) != entry["sha256"]:
            errors.append(entry["path"])
    if errors:
        raise ValueError(f"Transfer verification failed for {len(errors)} files: {errors[:10]}")


def rewrite(value, old: str, new: str):
    if isinstance(value, str):
        if value == old or value.startswith(old + "/"):
            return new + value[len(old):], 1
        return value, 0
    if isinstance(value, list):
        result = []
        changes = 0
        for item in value:
            item, count = rewrite(item, old, new)
            result.append(item)
            changes += count
        return result, changes
    return value, 0


def atomic_write(path: Path, content: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".migration-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def rewrite_json(text: str, old: str, new: str) -> tuple[str, int]:
    json.loads(text)  # Validate syntax; never serialize parsed scientific numbers.
    changes = 0
    key_suffix = re.compile(r"\s*:")

    def replace(match):
        nonlocal changes
        if key_suffix.match(text, match.end()):
            return match.group()  # Object keys are identifiers, not path values.
        value, count = rewrite(json.loads(match.group()), old, new)
        changes += count
        return json.dumps(value, ensure_ascii=False) if count else match.group()

    return re.sub(r'"(?:\\.|[^"\\])*"', replace, text), changes


def rebase(root: Path, manifest_path: Path, apply: bool = False) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify(root, manifest)
    old = manifest["source_project_root"].replace("\\", "/").rstrip("/")
    new = root.resolve().as_posix()
    if not old or old == "/":
        raise ValueError("Invalid source project root")
    updated = copy.deepcopy(manifest)
    planned = []
    for entry in updated["files"]:
        path = member(root, entry["path"])
        if PurePosixPath(entry["path"]).parts[0] not in ARTIFACT_ROOTS or path.suffix not in {".json", ".csv"} or old == new:
            continue
        original = path.read_bytes()
        text = original.decode("utf-8-sig")
        if path.suffix == ".json":
            updated_text, changes = rewrite_json(text, old, new)
            content = updated_text.encode("utf-8")
        else:
            value, changes = rewrite(list(csv.reader(io.StringIO(text, newline=""))), old, new)
            output = io.StringIO(newline="")
            csv.writer(output).writerows(value)
            content = output.getvalue().encode("utf-8")
        if not changes:
            continue
        if original.startswith(b"\xef\xbb\xbf"):
            content = b"\xef\xbb\xbf" + content
        entry["sha256"] = hashlib.sha256(content).hexdigest()
        entry["bytes"] = len(content)
        planned.append((path, original, content))
    output_manifest = manifest_path.with_name("manifest-relocated.json")
    if apply:
        if output_manifest.exists():
            raise FileExistsError(f"Relocation record already exists: {output_manifest}")
        written = []
        try:
            for path, original, content in planned:
                atomic_write(path, content)
                written.append((path, original))
            updated["source_project_root"] = new
            updated["relocated_from"] = old
            atomic_write(output_manifest, (json.dumps(updated, indent=2) + "\n").encode("utf-8"))
        except BaseException:
            for path, original in reversed(written):
                atomic_write(path, original)
            raise
    return {"files_to_rebase": len(planned), "applied": apply, "old_root": old, "new_root": new,
            "verification_manifest": str(output_manifest if apply else manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["verify", "rebase"])
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "migration/manifest.json")
    parser.add_argument("--apply", action="store_true", help="Write changes; otherwise preview only")
    args = parser.parse_args()
    if args.operation == "verify":
        verify(PROJECT_ROOT, json.loads(args.manifest.read_text(encoding="utf-8")))
        print("Transfer checksums verified")
    else:
        print(json.dumps(rebase(PROJECT_ROOT, args.manifest, args.apply), indent=2))


if __name__ == "__main__":
    main()
