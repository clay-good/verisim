# Tasks — Contract-mediated synthesized-edge feedback

## 1. Discrepancy detection
- [ ] Implement a diff of a `RuntimeTrace` against the static `CodeGraph`: runtime calls with no
      matching static edge → candidate synthesized edges.
- [ ] (Secondary) Detect runtime paths that violate a declared architectural invariant →
      candidate findings.
- [ ] Deduplicate against the static graph, including pre-existing `synthesized` edges.

## 2. Node resolution
- [ ] Map runtime exec/call sites to OpenLore node ids by (file, symbol).
- [ ] Drop candidates that do not resolve to a known node (precision over recall); count drops.

## 3. Payload contract
- [ ] Define `verisim-feedback-v1` JSON schema in the style of
      `OpenLore/schemas/openlore-manifest-v1.json` (version, fixture source sha, edges[], findings[]).
- [ ] Each edge carries `callerId`, `calleeId`/`calleeName`, `file`, `line`, `kind`, `callType`,
      `confidence: 'synthesized'`, `synthesizedBy: 'verisim-runtime'`, and the justifying trace id.
- [ ] Implement the idempotent payload producer (stable ordering, no duplicates).

## 4. Local validator (ingest stand-in)
- [ ] Implement a Verisim-side validator: payload ↔ schema, and edges ↔ fixture node set.
- [ ] Reject payloads with unresolved nodes or wrong provenance labels.

## 5. Cross-repo contract doc + verification
- [ ] Document the OpenLore-side ingest interface this payload targets (inputs, validation,
      additive write through the existing synthesized-edge path, rejection rules).
- [ ] Test: a known runtime-only dynamic dispatch in a fixture produces exactly one candidate
      edge with correct provenance and evidence.
- [ ] Test: an edge already present in the static graph produces no candidate (idempotency).
- [ ] Test: a candidate that cannot resolve to a node is dropped, not emitted.
- [ ] Test: the validator accepts a well-formed payload and rejects malformed/over-claiming ones.
