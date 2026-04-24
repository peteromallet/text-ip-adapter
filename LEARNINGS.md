# Experiment 001 — Learnings

Source of truth for what we learned from the text-ip-adapter / Gemma-3-4B / 971-pair / 2000-step run. These get folded into the prompt-adapters knowledge base under `experiments/2026-04-text-001-gemma3-adapter/LEARNINGS.md` by the scaffolding build.

## Architecture findings (what held up)

- **Prefix K/V injection into a frozen Gemma-3-4B via patched attention forward actually works.** Monkey-patching `Gemma3Attention.forward` for target layers (17–32) lets us prepend projector-produced `(P, num_kv_heads, head_dim)` tensors to content K/V on seq-dim after RoPE. No explosion, no NaN; loss drops smoothly.
- **GQA-aware shapes matter at runtime.** Gemma-3-4B: 8 attention heads / 4 KV heads / head_dim 256. Projector output must be `(B, P, 4, 256)`, not 8. Reading `config.num_key_value_heads` at runtime (not hardcoding) caught this.
- **Zero-init projector output = training starts as a no-op on the frozen base.** Step-0 adapter output matches step-0 no_ref output, confirming the mechanism begins inert and only affects behavior as the projector learns.
- **RoPE on content K only (prefix K unrotated) is the right choice.** Prepending after `apply_rotary_pos_emb` for content K lets prefix positions act like positionless controllers. No signs this is wrong.
- **`use_cache=False` is required during generate** while KV-cache + prefix interaction is unfixed (attention mask size mismatch on decode steps). Slow but correct. Future work: fix properly.

## Architecture findings (what's weak / open)

- **Encoder bottleneck of 16 queries × 2560 dims probably underspecifies style.** At step 2000 on 971 pairs, T1 (reference discrimination) is strongly PASS but T3 (style-feature carryover) is FAIL. Diagnosis: encoder produces *different* latents per reference but not latents that encode *style-shaped* differences. Either widen to 32–64 queries, add a contrastive objective, or both.
- **Gemma-3 loads as `Gemma3ForConditionalGeneration` (multimodal wrapper).** Text tower is at `base.model.language_model.layers`, not `base.model.layers`. `_find_decoder_layers` and `_base_text_model` had to be path-aware. The vision tower (~400 MB) is loaded into memory but unused — waste for a text-only experiment. Future: drop `vision_tower` + `multi_modal_projector` at load time.

## Data findings

- **971 pairs / 4 registers is too small to produce reliable style-matching signal.** Adapter discriminates refs (T1 PASS) but doesn't style-match them (T3 FAIL). Dataset-scaling prescription from the style-transfer literature (10×–100× more) is probably right.
- **Rule-based instruction generator produces noise.** Strings like `"Compose a brief piece on the theme of act and iii"` are meaningless — the theme extraction pulls stopwords or section headers. Noisy instructions force the adapter to compensate via the prefix channel, which pollutes the style signal. **This is the cheapest thing to fix and the next thing we're fixing in experiment 002.**
- **Author-paired splits with author disjointness across train/val/test works.** `tests/test_register_splits.py` enforces no `(register, author)` overlap across splits. Validated at 40+ poets, no leaks.
- **Three-layer content-leak firewall works.** T4 leak at step 2000: 0% of adapter outputs share any 5-gram with the provided reference. Target-memorization also 0% at 10-gram granularity. No evidence the adapter copy-pastes.
- **Register diversity expanded 1 → 4** (poetry, essay, speech, screenplay); pg19 and reddit still blocked by HF `trust_remote_code` deprecation. Screenplay URL-encoding bug found and fixed. Miller Center speech fetcher works.
- **Gutenberg scraping pitfalls:** (a) `langston_hughes` book_id 60902 yields Bellingrath Gardens text, not Hughes — bad ID, and Hughes isn't PD until 2038 anyway. (b) `john_keats` book_id 2422 yielded <3 valid docs after length filtering. (c) Miller Center license posture was `fair_use_review`, not pure PD; operator review pending.

## Training / diagnostic findings

- **Reference discrimination emerges between step 600 and step 1100.** At step 600, adapter and adapter_swap outputs were essentially identical; by step 1100, 3-gram Jaccard drops to ~0. Sharp phase transition rather than smooth ramp.
- **T2 (vs prompted_baseline) peaked at step 1500 (67% win rate) and regressed to 50% (tie) at step 2000.** Two explanations: (a) small-n LLM-judge noise (only 4 probes), (b) late-stage overfitting / cosine schedule hurting generalization. Needs more probes for signal, and worth running eval at intermediate checkpoints.
- **Surface-features-based T3 is a weak signal on small data.** `mean_advantage` hovered near 0 (advantage of own-ref-match over swap-ref-match). Better style metric might be a learned style classifier or LLM-judge directly on style-match. Current surface metrics (line length, archaic rate, em-dashes, TTR) don't vary enough between 4-register outputs.
- **Loss curve was "SLOW" throughout (14.8% drop, still improving at end).** Cosine schedule pulled LR to near-zero by step 1990, so model didn't plateau from convergence — it ran out of learning rate. Longer schedule + more steps would probably help if data were bigger.
- **Claude (sonnet-4-5) as an LLM-judge is cheap and fast** — ~$0.001/judgment, subsecond latency. Worth using as a primary evaluation signal when probe count scales up.

## Infra / ops findings

- **`pip install -e .` on the pod requires `[tool.hatch.metadata] allow-direct-references = true`** because `runpod-lifecycle @ file:../runpod-lifecycle` is a direct reference, now under `[project.optional-dependencies].infra`. Main pod install skips it.
- **`setsid bash -c '... > log 2>&1' < /dev/null > /dev/null 2>&1 &` is the reliable pattern** for backgrounding long jobs over SSH without paramiko hanging on stdout drain.
- **RunPod pod `open_ssh_client`** gives us direct paramiko access — used in `ssh_sync.py` for SFTP uploads. This resolved the gate v2 concern about the sibling package's SSH surface.
- **One pod (RTX 4090 @ $0.69/hr) handles the full Gemma-3-4B training + data fetch + probe gen comfortably.** Total experiment 001 compute: ~$0.83 for ~1.5 hours.

## Decision-matrix state at end of experiment 001

| Test | Result | Interpretation |
|---|---|---|
| T1 discrimination | **PASS** | Adapter reads the reference. |
| T2 vs prompted_baseline | **TIE** (n=4) | No clear win over prompting at 2000 steps. |
| T3 style carryover | **FAIL** | Outputs differ per ref but not in style-shaped ways. |
| T4 memorization | PASS | No target copy-paste. |
| T4 reference leak | PASS | No reference copy-paste. |
| T5 loss curve | SLOW | 14.8% drop, trained to completion. |

## Prescription ordering (cheapest → most expensive)

1. **Experiment 002 — upgrade rule-based instructions to LLM-generated** (~$5, ~1 hr code + retrain). Likely highest value per dollar: removes the biggest noise source and may push T3 from FAIL → at least WEAK without any architectural change.
2. **Experiment 003 — contrastive loss on encoder outputs.** Half-day of code; same data. Directly attacks the "encoder doesn't extract style" diagnosis.
3. **Experiment 004 — scale data 10× across 6 registers** (fix pg19 via alternative source, fix reddit ditto). 1–2 days of ingest work; then retrain.
4. Only after the above three: consider architectural changes (wider encoder, Flamingo-style per-layer cross-attention, LoRA on upper layers).

## What would flip the project

- T2 (vs prompting) going to PASS consistently after ANY of the above → the approach is validated and we can move to capability tests (α-blending, strength-dial).
- T2 staying TIE/FAIL after all three above → the reference-channel advantage is real but marginal at this model scale; rethink or accept as "context-efficient prompting" rather than "fundamentally better conditioning."
