"""Multi-register split tests.

After a stubbed multi-register ingest, ensure no (register, author_key)
appears in both train and test splits. Settled Decision: author-split
mandatory, register-stratified.
"""

from text_ip_adapter.data.pairing import make_pairs, split_by_author


def _stub_multi_register_records():
    registers = ["poetry", "prose_fiction", "speech", "essay", "screenplay", "reddit"]
    records = []
    for reg in registers:
        # Give each register a good number of authors so 80/10/10 is meaningful.
        for j in range(12):
            author = f"{reg}_author_{j}"
            for k in range(4):
                records.append({
                    "doc_id": f"{author}_{k}",
                    "author": author,
                    "text": f"doc {k} by {author} " + (f"{reg}_word " * 40),
                    "source": "stub",
                    "register": reg,
                    "license": "public_domain",
                })
    return records


def test_no_cross_register_pairs():
    recs = _stub_multi_register_records()
    pairs = make_pairs(recs, max_pairs_per_author=3)
    assert pairs, "should emit pairs"
    # Every pair's ref/target come from the same register.
    # That's implicit in make_pairs (it buckets by (register, author)), but
    # we still assert it here as a contract.
    for p in pairs:
        # author was keyed with register prefix by the stub.
        assert p["author"].startswith(p["register"] + "_author_")


def test_no_author_leaks_across_splits():
    recs = _stub_multi_register_records()
    pairs = make_pairs(recs, max_pairs_per_author=3)
    splits = split_by_author(pairs)

    train_keys = {(p["register"], p["author"]) for p in splits["train"]}
    val_keys = {(p["register"], p["author"]) for p in splits["val"]}
    test_keys = {(p["register"], p["author"]) for p in splits["test"]}

    assert not (train_keys & test_keys), "train and test share (register, author)"
    assert not (train_keys & val_keys), "train and val share (register, author)"
    assert not (val_keys & test_keys), "val and test share (register, author)"


def test_every_register_in_every_split():
    recs = _stub_multi_register_records()
    pairs = make_pairs(recs, max_pairs_per_author=3)
    splits = split_by_author(pairs)

    for split_name, split_pairs in splits.items():
        regs_here = {p["register"] for p in split_pairs}
        # Each register should appear in every split given 12 authors/register.
        assert len(regs_here) == 6, (
            f"split={split_name} only covers registers={regs_here}"
        )


def test_stub_ingest_dispatch_uses_registry():
    """Ensure the registry dispatch mechanism is wired up correctly."""
    from text_ip_adapter.data.sources import REGISTRY, cache_dir_for
    from pathlib import Path

    expected = {"poetry", "prose_fiction", "speech", "essay", "screenplay", "reddit"}
    assert set(REGISTRY.keys()) == expected
    # Cache dirs are register-specific and under data/raw/.
    for reg in expected:
        p = cache_dir_for(reg, Path("/tmp/fake_root"))
        assert "raw" in p.parts
