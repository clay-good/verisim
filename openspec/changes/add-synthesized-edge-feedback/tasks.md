# Tasks — Contract-mediated synthesized-edge feedback

## 1. Discrepancy detection
- [x] Implement a diff of a `RuntimeTrace` against the static `CodeGraph`: runtime calls with no
      matching static edge → candidate synthesized edges. (`detect_candidates` in
      `src/verisim/bridge/feedback.py`)
- [x] (Secondary) Detect runtime paths that violate a declared architectural invariant →
      candidate findings. (`detect_findings` in `src/verisim/bridge/feedback.py`: a
      `LayerInvariant` is a one-directional forbidden layer dependency by repo-relative file-path
      prefix; each resolved runtime invocation entrypoint→callee that crosses it becomes an
      evidence-bearing `RuntimeFinding` landing in the payload's `findings[]` slot, deduped by
      `(invariant, caller, callee)`. `build_feedback_payload(..., invariants=...)` populates the
      slot; with no invariants it stays empty. The slot/schema/`FeedbackPayload.findings` reserved
      by the primary change are now filled — no format change. A finding is reported even when the
      edge is already static, since the invariant is about the runtime *path*, not edge novelty —
      what distinguishes it from a missed-edge candidate.)
- [x] Deduplicate against the static graph, including pre-existing `synthesized` edges.
      (`static_pairs` set is built from all `graph.edges`; tested by
      `test_pre_existing_synthesized_edge_is_not_re_proposed`)

## 2. Node resolution
- [x] Map runtime exec/call sites to OpenLore node ids by (file, symbol). (`_resolve_exec_to_node`
      tries the program then each arg — so an interpreter run resolves the script, not the
      interpreter; `_resolve_file_to_node` anchors by `file_path`, internal nodes only.)
- [x] Drop candidates that do not resolve to a known node (precision over recall); count drops.
      (`detect_candidates` returns `(candidates, dropped)`; degraded traces and external leaves never
      anchor.)

## 3. Payload contract
- [x] Define `verisim-feedback-v1` JSON schema in the style of
      `OpenLore/schemas/openlore-manifest-v1.json` (version, fixture source sha, edges[], findings[]).
      (`schemas/verisim-feedback-v1.json`)
- [x] Each edge carries `callerId`, `calleeId`/`calleeName`, `file`, `line`, `kind`, `callType`,
      `confidence: 'synthesized'`, `synthesizedBy: 'verisim-runtime'`, and the justifying trace
      evidence (`CandidateEdge.to_payload_dict`).
- [x] Implement the idempotent payload producer (stable ordering, no duplicates).
      (`build_feedback_payload` dedups by `(callerId, calleeId)` and sorts; tested by
      `test_payload_is_idempotent_and_dedupes_across_traces`.)

## 4. Local validator (ingest stand-in)
- [x] Implement a Verisim-side validator: payload ↔ schema, and edges ↔ fixture node set.
      (`validate_payload`)
- [x] Reject payloads with unresolved nodes or wrong provenance labels. (unknown-node and
      mislabel rejection tested.)

## 5. Cross-repo contract doc + verification
- [x] Document the OpenLore-side ingest interface this payload targets (inputs, validation,
      additive write through the existing synthesized-edge path, rejection rules).
      (`docs/openlore-ingest-contract.md`)
- [x] Test: a known runtime-only dynamic dispatch in a fixture produces exactly one candidate
      edge with correct provenance and evidence.
      (`test_runtime_only_dynamic_dispatch_becomes_one_candidate`)
- [x] Test: an edge already present in the static graph produces no candidate (idempotency).
      (`test_already_known_edge_is_not_re_proposed`)
- [x] Test: a candidate that cannot resolve to a node is dropped, not emitted.
      (`test_unresolvable_invocation_is_dropped_and_counted`)
- [x] Test: the validator accepts a well-formed payload and rejects malformed/over-claiming ones.
      (`test_validator_*`)
