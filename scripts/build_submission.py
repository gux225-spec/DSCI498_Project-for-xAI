"""Build complementary code/evidence ZIPs and verify every packaged file hash.

Usage: python scripts/build_submission.py --output DIRECTORY_OUTSIDE_PROJECT
       python scripts/build_submission.py --verify ARCHIVE.zip
Standard library only. Extract both ZIPs into the same parent directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

PREFIX = "DSCI498_Project"
ROOT_FILES = {"README.md", "DATASET_REVIEW.md", ".gitignore", ".gitattributes", "requirements-check.txt",
              "requirements.txt", "requirements-xai.txt"}
CODE_DIRS = {"src", "scripts", "docs", "report", "ads"}
EVIDENCE_DIRS = {"data", "results"}


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def inventory(root):
    selected, excluded = {"code": [], "evidence": []}, []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        parts = rel.parts
        if any(p in {".git", "__pycache__", ".pytest_cache", ".venv", "work"} for p in parts):
            excluded.append({"path": rel.as_posix(), "reason": "environment or generated working file"})
        elif rel.as_posix() == "docs/reference/shap_lime_filter_paper.txt":
            excluded.append({"path": rel.as_posix(), "reason": "third-party full-text working reference; bibliography and citations retained"})
        elif parts[0] == "results" and "sim" in parts:
            excluded.append({"path": rel.as_posix(), "reason": "openEMS intermediate port/time-series output; response.npz and summary.json retained"})
        elif parts[0] in EVIDENCE_DIRS:
            selected["evidence"].append((rel, path))
        elif parts[0] in CODE_DIRS or (len(parts) == 1 and parts[0] in ROOT_FILES):
            selected["code"].append((rel, path))
        else:
            excluded.append({"path": rel.as_posix(), "reason": "outside release allowlist"})
    return selected, excluded


def verify(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP entries")
        if any(PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts for n in names):
            raise ValueError("Unsafe ZIP member path")
        manifests = [n for n in names if n.startswith(PREFIX+"/manifests/") and n.endswith("_manifest.json")]
        if len(manifests) != 1:
            raise ValueError("Expected one package manifest")
        manifest_name = manifests[0]
        manifest = json.loads(archive.read(manifest_name))
        expected = {PREFIX+"/"+f["path"]: f for f in manifest["files"]}
        if set(names) != set(expected) | {manifest_name}:
            raise ValueError("ZIP content differs from its manifest")
        for name, record in expected.items():
            h, count = hashlib.sha256(), 0
            with archive.open(name) as stream:
                for data in iter(lambda: stream.read(1024*1024), b""):
                    h.update(data)
                    count += len(data)
            if h.hexdigest() != record["sha256"] or count != record["bytes"]:
                raise ValueError("Content hash/size mismatch: "+name)
    return {"archive": path.name, "files": len(expected), "verified": True}


def build(root, output):
    selected, excluded = inventory(root)
    for required in ["README.md", "docs/decision_log.md", "scripts/verify_submission.py", "results/submission_audit/verification.json"]:
        if not (root/required).is_file():
            raise ValueError("Required release file missing: "+required)
    audit = json.loads((root/"results/submission_audit/verification.json").read_text(encoding="utf-8"))
    if audit["counts"]["fail"]:
        raise ValueError("Resolve failing audit checks before packaging")
    expected = json.loads((root/"data/filter_supplement_v2/artifact_hashes.json").read_text(encoding="utf-8"))
    evidence_names = {rel.as_posix() for rel, _ in selected["evidence"]}
    missing = [name for name in expected if "data/filter_supplement_v2/"+name not in evidence_names]
    if missing:
        raise ValueError("Dataset release files excluded: "+str(missing))
    output.mkdir(parents=True, exist_ok=True)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "common_directory": PREFIX,
               "instructions": "Extract both archives into the same parent directory; their DSCI498_Project folders merge.",
               "audit_status": audit["status"], "parts": [], "excluded_files": excluded,
               "external_prerequisites": ["ADS 2026 and a suitable license for new circuit simulations",
                   "openEMS/CSXCAD for new full-wave simulations", "Original native ADS workspaces outside this project are not included"],
               "report_status": "Compiled final report and AI-use reflection are included when present; original draft and preparation notes are retained as history."}
    for part in ["code", "evidence"]:
        archive_path = output/f"DSCI498_{part}.zip"
        print(f"Building {part}: {len(selected[part])} files", flush=True)
        manifest = {"part": part, "created_utc": summary["created_utc"], "common_directory": PREFIX,
                    "companion_archive": f"DSCI498_{'evidence' if part == 'code' else 'code'}.zip", "files": []}
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
            for rel, source in selected[part]:
                compression = zipfile.ZIP_STORED if source.suffix.lower() == ".npz" else zipfile.ZIP_DEFLATED
                archive.write(source, PREFIX+"/"+rel.as_posix(), compress_type=compression)
                manifest["files"].append({"path": rel.as_posix(), "bytes": source.stat().st_size, "sha256": file_hash(source)})
            archive.writestr(PREFIX+f"/manifests/{part}_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
        result = verify(archive_path)
        result.update(bytes=archive_path.stat().st_size, sha256=file_hash(archive_path),
                      unpacked_bytes=sum(f["bytes"] for f in manifest["files"]))
        summary["parts"].append(result)
        print(json.dumps(result), flush=True)
    (output/"submission_package_index.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    (output/"SHA256SUMS.txt").write_text("\n".join(f"{p['sha256']}  {p['archive']}" for p in summary["parts"])+"\n", encoding="ascii")
    print("Both packages verified against their embedded file manifests.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--output", type=Path)
    actions.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify.resolve()), indent=2))
        return
    root, output = args.root.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        parser.error("Output must be outside the project to prevent recursive packaging")
    build(root, output)


if __name__ == "__main__":
    main()
