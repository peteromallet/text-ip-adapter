from text_ip_adapter.data.instructions import make_instruction
from text_ip_adapter.data.pairing import make_pairs, split_by_author


def _stub_records(register: str = "poetry"):
    records = []
    for i, author in enumerate([f"{register}_author_{j}" for j in range(30)]):
        for k in range(4):
            records.append({
                "doc_id": f"{author}_{k}",
                "author": author,
                "text": f"doc {k} by {author} " + ("word " * 50),
                "source": "stub",
                "register": register,
                "license": "public_domain",
            })
    return records


def test_pairing_groups_by_author():
    recs = _stub_records()
    pairs = make_pairs(recs, max_pairs_per_author=3)
    assert pairs, "should emit pairs"
    for p in pairs:
        assert p["ref_doc_id"] != p["target_doc_id"]
        assert p["ref_doc_id"].startswith(p["author"])
        assert p["target_doc_id"].startswith(p["author"])
        assert p["register"] == "poetry"


def test_author_disjoint_splits():
    recs = _stub_records()
    pairs = make_pairs(recs, max_pairs_per_author=3)
    splits = split_by_author(pairs)
    train_a = {(p["register"], p["author"]) for p in splits["train"]}
    val_a = {(p["register"], p["author"]) for p in splits["val"]}
    test_a = {(p["register"], p["author"]) for p in splits["test"]}
    assert not (train_a & test_a)
    assert not (train_a & val_a)
    assert not (val_a & test_a)
    assert train_a and val_a and test_a


def test_record_schema_has_register_field():
    recs = _stub_records(register="essay")
    pairs = make_pairs(recs, max_pairs_per_author=2)
    for p in pairs:
        assert "register" in p
        assert p["register"] == "essay"


def test_make_instruction_register_aware():
    text = "The quick brown fox jumps over the lazy dog. The dog sleeps in the sun."
    poem = make_instruction(text, register="poetry")
    speech = make_instruction(text, register="speech")
    essay = make_instruction(text, register="essay")
    # Different templates; each should contain a theme word.
    assert any(w in poem.lower() for w in ("poem", "verse", "stanza", "piece"))
    assert any(w in speech.lower() for w in ("speech", "address", "remarks", "statement"))
    assert "essay" in essay.lower() or "piece" in essay.lower()
