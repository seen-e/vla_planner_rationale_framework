"""Build immutable, provenance-checked Planner inputs from accepted Stage3 versions."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def correction_digest(path):
    return hashlib.sha256(json.dumps(read(path), ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def normalize(raw, manifest, duration):
    rows = []
    samples = manifest.get("samples", [])
    for pos, source in enumerate(raw["subtasks"]):
        row = copy.deepcopy(source)
        sid = row.get("subtask_index", pos)
        if type(sid) is not int:
            raise ValueError("Noninteger subtask identity")
        if "start_time_sec" in row and "end_time_sec" in row:
            start, end = row["start_time_sec"], row["end_time_sec"]
        else:
            a, b = row.get("start_sample_index"), row.get("end_sample_index_exclusive")
            if not (type(a) is int and type(b) is int and 0 <= a < b <= len(samples)):
                raise ValueError("Invalid sample boundaries; cannot infer from nominal FPS")
            start = samples[a]["timestamp_sec"] if a else 0.0
            end = samples[b]["timestamp_sec"] if b < len(samples) else duration
            row["original_sample_reference_map"] = [
                {"sample_index": i, "frame_index": samples[i]["frame_index"],
                 "timestamp_sec": samples[i]["timestamp_sec"]}
                for i in range(max(0, a - 1), min(len(samples), b + 1))]
        if not (0 <= start < end <= duration + 1e-6):
            raise ValueError("Invalid subtask interval")
        if rows and abs(rows[-1]["end_sec"] - start) > 1e-6:
            raise ValueError("Noncontiguous Stage3 timeline")
        row.update(subtask_index=sid, start_sec=start, end_sec=end)
        rows.append(row)
    if not rows or len({r["subtask_index"] for r in rows}) != len(rows):
        raise ValueError("Empty timeline or duplicate identities")
    if abs(rows[0]["start_sec"]) > 1e-6 or abs(rows[-1]["end_sec"] - duration) > 1e-6:
        raise ValueError("Stage3 does not cover the episode")
    return rows


def prepare(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError("Input snapshot already exists; choose a new output directory")
    audit_root, correction_root = Path(args.audit_root), Path(args.correction_root)
    standards = read(args.standards)["episodes"]
    corrected = read(correction_root / "results.json")
    selected = {}
    if args.scope == "all_accepted":
        with ThreadPoolExecutor(max_workers=32) as pool:
            for eid, audit in pool.map(lambda eid: (eid, read(audit_root / eid / "final.json")), sorted(standards)):
                if audit.get("episode_id") != eid:
                    raise ValueError("Audit episode mismatch")
                if audit.get("final_status") == "normal":
                    selected[eid] = "original_audit_pass"
    for eid, status in corrected.items():
        if status.get("status") in ("corrected", "reannotated"):
            selected[eid] = "corrected_verification_pass"
        elif eid in selected:
            # A later unresolved correction must not be silently replaced by stale input.
            del selected[eid]

    def build(item):
        eid, kind = item
        original_dir = Path(args.annotations_root) / eid
        source = original_dir / "final.json"
        sample_file = original_dir / "sample_manifest.json"
        manifest = read(sample_file)
        meta_file = Path(args.metadata_root) / eid / "metadata.json"
        meta = read(meta_file)["source"]
        if meta["episode_id"] != eid or meta["last_source_frame"] != meta["first_source_frame"] + meta["length"] - 1:
            raise ValueError(f"{eid}: invalid video mapping")
        if manifest["frame_count"] != meta["length"] or abs(manifest["fps"] - meta["fps"]) > 1e-6:
            raise ValueError(f"{eid}: sample manifest and metadata disagree")
        if Path(meta["source_video"]).stat().st_size != meta["source_size"]:
            raise ValueError(f"{eid}: source video changed")
        audit_file = audit_root / eid / "final.json"
        if kind == "corrected_verification_pass":
            d = correction_root / eid
            source = d / "final_corrected_stage3.json"
            verification = read(d / "verification.json")
            review = read(d / "final_review.json")
            status = corrected[eid]
            if verification["verification"] != "pass" or review["stage3_verdict"] != "correct":
                raise ValueError(f"{eid}: correction not accepted")
            if correction_digest(source) != status["final_sha256"] or correction_digest(d / "final_review.json") != status["review_sha256"]:
                raise ValueError(f"{eid}: corrected file hash mismatch")
            audit_file = d / "final_review.json"
        raw = read(source)
        if raw.get("episode_id", eid) != eid:
            raise ValueError("Annotation identity mismatch")
        rows = normalize(raw, manifest, meta["length"] / meta["fps"])
        if kind == "original_audit_pass":
            snapshot = read(audit_root / eid / "round1/input.json")
            payload = snapshot.get("payload", snapshot.get("input"))
            if payload is None:
                raise ValueError(f"{eid}: audit payload not found")
            audited = payload["final_annotation"]["subtasks"]
            if len(audited) != len(rows):
                raise ValueError("Audited subtask count changed")
            for old, current in zip(audited, rows):
                for key in ("subtask_index", "segment_type", "segment_summary", "arm_1", "arm_2"):
                    if old.get(key) != current.get(key):
                        raise ValueError(f"{eid}: audited Stage3 changed: {key}")
        return {"episode_id": eid, "task": standards[eid]["task"],
                "video_path": meta["source_video"], "first_source_frame": meta["first_source_frame"],
                "episode_frame_count": meta["length"], "source_fps": meta["fps"],
                "time_basis": "episode_absolute_seconds_half_open", "subtasks": rows,
                "source_provenance": {"selection": kind, "stage3_path": str(source),
                    "stage3_sha256": sha(source), "review_path": str(audit_file),
                    "review_sha256": sha(audit_file), "metadata_path": str(meta_file),
                    "metadata_sha256": sha(meta_file), "sample_manifest_path": str(sample_file),
                    "sample_manifest_sha256": sha(sample_file)}}

    with ThreadPoolExecutor(max_workers=32) as pool:
        episodes = list(pool.map(build, sorted(selected.items())))
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps({"episodes": episodes}, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = {k: sum(e["source_provenance"]["selection"] == k for e in episodes)
              for k in sorted(set(selected.values()))}
    summary = {"episode_count": len(episodes), "subtask_count": sum(len(e["subtasks"]) for e in episodes),
               "selection_counts": counts, "scope": args.scope,
               "excluded_corrections": {k: v["status"] for k, v in corrected.items() if k not in selected}}
    (output / "selection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    for name in ["annotations-root", "audit-root", "correction-root", "metadata-root", "standards", "output"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--scope", choices=["all_accepted", "corrected_accepted"], default="all_accepted")
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
