#!/usr/bin/env python3
"""SPEC-22 RA21 (H152) -- replay a real Claude Code transcript corpus through the coverage gate.

Walks ``~/.claude/projects/**/*.jsonl`` (override with ``--root``), extracts every state-mutating
tool call (Bash / Edit / Write / NotebookEdit) with its ``cwd``, runs it through the oracle coverage
gate and the denylist status quo (:mod:`verisim.realagent.cc_corpus`), and writes a privacy-safe
aggregate: rates, per-tool counts, and the top auto-prompted verbs. **Raw command strings and file
paths never leave this process** -- only derived :class:`~verisim.realagent.cc_corpus.Record`s are
aggregated, so ``RESULTS.csv``/``summary.json`` are safe to commit.

Usage::

    python bench/cc_corpus/extract.py                 # ~/.claude/projects -> bench/cc_corpus/
    python bench/cc_corpus/extract.py --root DIR --out OUTDIR --limit N
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path

# allow running as a plain script (no install) by adding src/ to the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from verisim.realagent.cc_corpus import (
    CorpusStats,
    Record,
    aggregate_by_group,
    cc_corpus_verdict,
    missed_harm,
    project_breakdown_verdict,
    record_for_call,
    summarize_projects,
    write_csv,
    write_project_csv,
)

MUTATING = {"Bash", "Edit", "Write", "NotebookEdit"}


def iter_project_records(root: str, limit: int | None = None) -> Iterator[tuple[str, Record]]:
    """Yield (project, Record) per state-mutating tool call. Project = top dir under root."""
    files = sorted(glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True))
    n = 0
    for fpath in files:
        project = os.path.relpath(fpath, root).split(os.sep)[0]
        try:
            with open(fpath, encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError:
            continue
        for line in lines:
            try:
                ev = json.loads(line)
            except (ValueError, TypeError):
                continue
            cwd = str(ev.get("cwd") or "")
            msg = ev.get("message") or {}
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not (isinstance(block, dict) and block.get("type") == "tool_use"):
                    continue
                name = block.get("name")
                if name not in MUTATING:
                    continue
                tool_input = block.get("input")
                if not isinstance(tool_input, dict):
                    continue
                yield project, record_for_call(str(name), tool_input, cwd)
                n += 1
                if limit is not None and n >= limit:
                    return


def iter_records(root: str, limit: int | None = None) -> Iterator[Record]:
    """Yield a privacy-safe Record per state-mutating tool call across transcripts under root."""
    for _project, record in iter_project_records(root, limit):
        yield record


def summarize(stats: CorpusStats, project_breakdown: dict[str, object]) -> dict[str, object]:
    harms = missed_harm()
    verdict = cc_corpus_verdict(stats, harms)
    verdict["by_tool"] = dict(stats.by_tool)
    verdict["top_offsurface_denylist_verbs"] = stats.top_offsurface_verbs.most_common(12)
    verdict["top_onsurface_verbs"] = stats.top_onsurface_verbs.most_common(12)
    verdict["harm_detail"] = {h.arm: {"missed_harm": h.missed_harm, "missed": h.missed}
                              for h in harms}
    verdict["project_breakdown"] = project_breakdown
    return verdict


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.expanduser("~/.claude/projects"),
                    help="transcript root (default: ~/.claude/projects)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent),
                    help="output directory for RESULTS.csv + summary.json")
    ap.add_argument("--limit", type=int, default=None, help="cap number of calls (smoke runs)")
    ap.add_argument("--min-calls", type=int, default=20,
                    help="min calls for a project to enter the cross-project distribution")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.root):
        print(f"transcript root not found: {args.root}", file=sys.stderr)
        return 2

    per_project = aggregate_by_group(iter_project_records(args.root, args.limit))
    stats = CorpusStats()
    for s in per_project.values():
        stats.n += s.n
        stats.by_tool.update(s.by_tool)
        stats.coverage_ask += s.coverage_ask
        stats.denylist_ask += s.denylist_ask
        stats.denylist_ask_offsurface += s.denylist_ask_offsurface
        stats.top_offsurface_verbs.update(s.top_offsurface_verbs)
        stats.top_onsurface_verbs.update(s.top_onsurface_verbs)
    if stats.n == 0:
        print("no state-mutating tool calls found", file=sys.stderr)
        return 1

    harms = missed_harm()
    out = Path(args.out)
    csv_path = write_csv(stats, harms, str(out / "RESULTS.csv"))
    proj_csv = write_project_csv(per_project, str(out / "BY_PROJECT.csv"))
    pb = project_breakdown_verdict(summarize_projects(per_project, min_calls=args.min_calls))
    summary = summarize(stats, pb)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    p25p75 = pb["offsurface_density_p25_p75"]
    assert isinstance(p25p75, tuple)
    p25, p75 = float(p25p75[0]), float(p25p75[1])
    print(f"replayed {stats.n:,} real state-mutating tool calls from {args.root}")
    print(f"  off-surface (auto-approved) density : {stats.offsurface_density:.4f}")
    print(f"  coverage gate prompt rate           : {stats.coverage_prompt_rate:.4f}")
    print(f"  denylist prompt rate                : {stats.denylist_prompt_rate:.4f}")
    print(f"  denylist fatigue (off-surface) rate : {stats.denylist_fatigue_rate:.4f}")
    print(f"  coverage missed-harm (arsenal)      : {summary['coverage_missed_harm']:.4f}")
    print(f"  denylist missed-harm (arsenal)      : {summary['denylist_missed_harm']:.4f}")
    print(f"--- cross-project generalization ({pb['n_projects_kept']} projects "
          f">= {args.min_calls} calls) ---")
    print(f"  off-surface density   median        : {pb['offsurface_density_median']:.4f}")
    print(f"  off-surface density   min           : {pb['offsurface_density_min']:.4f}")
    print(f"  off-surface density   p25/p75       : {p25:.4f} / {p75:.4f}")
    print(f"  projects below 0.95 (security-heavy): {pb['projects_below_95']}")
    print(f"  stable central tendency             : {pb['stable_central_tendency']}")
    print(f"wrote {csv_path}, {proj_csv}, and {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
