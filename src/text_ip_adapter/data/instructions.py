from __future__ import annotations

import random
import re
from collections import Counter

_STOPWORDS = set(
    """a an and the of in on for for to with from by at as is are was were be been
    being have has had do does did but or not no nor so if then than that which
    who whom whose this these those there here it its i you he she we they them
    my your his her our their me us him her thee thy thou ye oh ah upon into
    over under out up down off very just all any some more most much many any
    each every about through during before after above below between above also
    only own same other such only too will would could should may might must
    can shall yet now ever never still one two three four five six seven eight
    nine ten like unto hath doth would'st o said says say get got go goes gone
    want wants wanted make makes made take takes took give gives gave new old
    good bad big small high low long short way ways thing things people lot lots
    know knew think thought feel feels felt time times day days year years""".split()
)

# Templates per register. Still rule-based; LLM upgrade deferred.
_TEMPLATES_BY_REGISTER: dict[str, list[str]] = {
    "poetry": [
        "Write a short poem about {theme}.",
        "Compose a brief piece on the theme of {theme}.",
        "Write a verse about {theme}.",
        "Create a short poem exploring {theme}.",
        "Write a few stanzas about {theme}.",
    ],
    "prose_fiction": [
        "Write a scene in which {theme} drives the action.",
        "Write a passage of fiction about {theme}.",
        "Compose a short narrative section centered on {theme}.",
        "Write a paragraph describing {theme}.",
        "Tell a short story excerpt involving {theme}.",
    ],
    "speech": [
        "Give a brief speech on {theme}.",
        "Deliver a short public address about {theme}.",
        "Write remarks to be delivered on the subject of {theme}.",
        "Compose a short speech about {theme}.",
        "Write a brief statement on {theme} for a public audience.",
    ],
    "essay": [
        "Write a short essay arguing about {theme}.",
        "Compose a brief essay on {theme}.",
        "Write a reflective essay about {theme}.",
        "Write a short argumentative piece concerning {theme}.",
        "Draft an essay exploring {theme}.",
    ],
    "screenplay": [
        "Write a scene where {theme}.",
        "Write a short screenplay scene involving {theme}.",
        "Compose a screenplay scene about {theme}.",
        "Write a scene that depicts {theme}.",
        "Draft a short screenplay scene focused on {theme}.",
    ],
    "reddit": [
        "Write a short post about {theme}.",
        "Write a short online post discussing {theme}.",
        "Write a casual post about {theme}.",
        "Compose a brief post on {theme}.",
        "Write a short forum post about {theme}.",
    ],
}

# Default fallback.
_DEFAULT_TEMPLATES = _TEMPLATES_BY_REGISTER["poetry"]


def extract_theme(text: str) -> str:
    tokens = re.findall(r"[A-Za-z]{3,}", text.lower())
    tokens = [t for t in tokens if t not in _STOPWORDS]
    if not tokens:
        return "life"
    counts = Counter(tokens)
    top = [w for w, _ in counts.most_common(3)]
    return " and ".join(top[:2]) if len(top) >= 2 else top[0]


def make_instruction(
    target_text: str,
    register: str | None = None,
    seed: int = 0,
) -> str:
    """Rule-based target-only instruction generation.

    register-aware: picks a template family per register. Default to poetry
    templates if the register is unknown (keeps legacy callers unchanged).
    """
    rng = random.Random(seed ^ hash(target_text[:50]))
    theme = extract_theme(target_text)
    templates = _TEMPLATES_BY_REGISTER.get(register or "", _DEFAULT_TEMPLATES)
    template = rng.choice(templates)
    return template.format(theme=theme)
