from text_ip_adapter.data.ingest_speeches import _extract_president_and_transcript, _president_slug_from_url


def test_president_slug_from_miller_url_uses_speech_date():
    url = (
        "https://millercenter.org/the-presidency/presidential-speeches/"
        "september-12-1962-address-space-effort"
    )
    assert _president_slug_from_url(url) == "kennedy"


def test_president_slug_from_miller_url_handles_second_trump_term():
    url = (
        "https://millercenter.org/the-presidency/presidential-speeches/"
        "february-1-2026-example-speech"
    )
    assert _president_slug_from_url(url) == "trump"


def test_extract_president_ignores_global_nav_president_links():
    html = """
    <html>
      <body>
        <nav><a href="/president/washington">George Washington</a></nav>
        <article>
          <div class="field-item">
            This transcript is long enough to be selected as page content.
            It should not inherit the global navigation's first president link.
            The parser should leave president metadata empty unless it appears
            in page-local attributes. This sentence pads the fixture beyond the
            transcript length threshold used by the extractor so that the test
            isolates president extraction rather than transcript extraction.
            This sentence pads the fixture beyond the transcript length
            threshold again, with enough words and characters to pass.
          </div>
        </article>
      </body>
    </html>
    """
    pres, transcript = _extract_president_and_transcript(html)
    assert pres is None
    assert transcript is not None
