"""Tests for the grounding rules — the parts that decide what a reader is told.

Everything here is pure: no index, no model, no network. The retrieval policy
that needs the corpus is measured by eval_retrieval.py instead, because a test
that depends on 2,602 embedded chunks tells you the corpus changed, not that the
code broke.
"""

import pytest
from langchain_core.documents import Document

import rag


# --- refusal detection -------------------------------------------------------
#
# The reserved sentence means "the corpus does not answer this". Whether it is
# the answer or merely appears in one is the difference between showing the
# reader citations and hiding them.


def test_refusal_at_the_start_is_a_refusal():
    assert rag.is_refusal("This is out of my knowledge.\n\nThe sources cover KYC [1].")


def test_bold_refusal_is_a_refusal():
    assert rag.is_refusal(rag.NO_CONTEXT)


def test_refusal_tacked_onto_an_answer_is_not_a_refusal():
    # The model breaking its own prompt rule must not suppress the citations of
    # an answer that plainly rests on sources.
    answer = "Section 8 requires reasonable security safeguards [1].\n\nThis is out of my knowledge."
    assert not rag.is_refusal(answer)


def test_cited_answer_is_not_a_refusal():
    assert not rag.is_refusal("Banks must update KYC records periodically [1][2].")


# --- stray refusal removal ---------------------------------------------------


def test_trailing_refusal_is_stripped():
    answer = "Section 8 requires safeguards [1].\n\nThis is out of my knowledge."
    assert rag.strip_stray_refusal(answer) == "Section 8 requires safeguards [1]."


def test_bold_trailing_refusal_is_stripped():
    answer = "Section 8 requires safeguards [1].\n\n**This is out of my knowledge.**"
    assert rag.strip_stray_refusal(answer) == "Section 8 requires safeguards [1]."


def test_a_real_refusal_survives_intact():
    answer = "This is out of my knowledge.\n\nThe sources cover digital lending [1]."
    assert rag.strip_stray_refusal(answer) == answer


def test_clean_answer_is_untouched():
    answer = "Rule 4 requires notice within 72 hours [2]."
    assert rag.strip_stray_refusal(answer) == answer


def test_stripping_never_returns_nothing():
    # An answer that is only the marker in some unexpected shape is still worth
    # showing; blanking the reply would be a worse failure than leaving it.
    assert rag.strip_stray_refusal("> this is out of my knowledge").strip()


# --- citation validation -----------------------------------------------------


def test_markers_are_parsed_deduplicated_and_sorted():
    cited, invalid, uncited = rag.validate_citations("a [3] b [1] c [3]", 4)
    assert cited == [1, 3]
    assert invalid == []
    assert uncited == [2, 4]


def test_marker_past_the_supplied_range_is_invalid():
    # The prompt forbids inventing source numbers, but a prompt is a request.
    _, invalid, _ = rag.validate_citations("as required [9]", 6)
    assert invalid == [9]


def test_answer_without_markers_cites_nothing():
    cited, invalid, uncited = rag.validate_citations("no markers here", 2)
    assert (cited, invalid, uncited) == ([], [], [1, 2])


# --- input checks ------------------------------------------------------------


def test_blank_question_is_rejected():
    assert rag.check_input("   ")


def test_oversized_question_is_rejected_not_truncated():
    # Truncating makes it a different question, and answering that confidently
    # would be worse than declining.
    assert rag.check_input("x" * (rag.MAX_INPUT_CHARS + 1))


def test_ordinary_question_passes():
    assert rag.check_input("What does section 8 require?") is None


# --- retrieval diversity -----------------------------------------------------


def _doc(framework, breadcrumb):
    return Document(page_content="", metadata={"framework_id": framework, "breadcrumb": breadcrumb})


def test_diversify_caps_repeats_of_one_section():
    # The real failure this exists for: one long section winning every slot.
    scored = [(_doc("RBI", "Ch V > 38 Periodic Updation"), 0.2 + i / 100) for i in range(6)]
    kept = rag.diversify(scored, k=6, max_per_section=2)
    assert len(kept) == 2


def test_diversify_keeps_nearest_first_and_fills_from_other_sections():
    scored = [
        (_doc("RBI", "Ch V > 38"), 0.20),
        (_doc("RBI", "Ch V > 38"), 0.22),
        (_doc("RBI", "Ch V > 38"), 0.24),
        (_doc("RBI", "Ch VIII > 50"), 0.30),
        (_doc("SEBI", "Ch II > 4"), 0.31),
    ]
    kept = rag.diversify(scored, k=4, max_per_section=2)
    assert [distance for _, distance in kept] == [0.20, 0.22, 0.30, 0.31]


def test_diversify_stops_at_k():
    scored = [(_doc("F", f"section {i}"), 0.1 * i) for i in range(10)]
    assert len(rag.diversify(scored, k=3)) == 3


# --- reference grouping ------------------------------------------------------


def _source(n, framework="RBI KYC Directions", breadcrumb=None, pages="p39"):
    return {
        "n": n,
        "framework_name": framework,
        "breadcrumb": breadcrumb or f"{framework} > Chapter V > 38 Updation",
        "pages": pages,
        "source_url": "https://example.test/doc",
    }


def test_repeats_of_one_location_become_one_reference():
    groups = rag.group_references([_source(1), _source(2), _source(4)])
    assert len(groups) == 1
    place, numbers = groups[0]
    assert numbers == [1, 2, 4]


def test_grouping_strips_the_document_name_from_the_location():
    (place, _), = rag.group_references([_source(1)])
    assert place["location"] == "Chapter V > 38 Updation"


def test_distinct_locations_stay_separate():
    other = _source(3, breadcrumb="RBI KYC Directions > Chapter VIII > 50 Alerts", pages="p52")
    assert len(rag.group_references([_source(1), other])) == 2


# --- framework detection -----------------------------------------------------


@pytest.fixture
def frameworks(monkeypatch):
    """A stand-in for the indexed frameworks, so these tests need no index."""
    monkeypatch.setattr(
        rag,
        "_framework_tokens",
        lambda: {
            "DPDP_ACT_2023": ({"dpdp"}, {"act"}),
            "DPDP_RULES_2025": ({"dpdp"}, {"rules"}),
            "GDPR_2018": ({"gdpr"}, set()),
            "IT_ACT_2000": ({"it"}, {"act"}),
            "RBI_KYC_MASTER_DIRECTIONS": ({"rbi", "kyc"}, {"master", "directions"}),
            "RBI_CYBERSECURITY_FRAMEWORK": ({"rbi", "cybersecurity"}, {"framework"}),
        },
    )


def test_the_instrument_word_separates_two_frameworks_of_one_name(frameworks):
    assert rag.detect_framework("penalties under the DPDP Act") == "DPDP_ACT_2023"
    assert rag.detect_framework("what the DPDP Rules require") == "DPDP_RULES_2025"


def test_a_bare_shared_name_picks_neither(frameworks):
    assert rag.detect_framework("what does DPDP say about consent") is None


def test_a_comparison_is_left_unscoped(frameworks):
    assert rag.detect_framework("compare DPDP Act and GDPR breach timelines") is None


def test_more_naming_words_win(frameworks):
    assert rag.detect_framework("RBI KYC periodic updation") == "RBI_KYC_MASTER_DIRECTIONS"


def test_a_question_naming_nothing_is_unscoped(frameworks):
    assert rag.detect_framework("what are the password complexity rules") is None


def test_a_short_token_needs_capitals(frameworks):
    # "it" is a pronoun far more often than it is the Information Technology Act.
    assert rag.detect_framework("is it mandatory to encrypt data") is None
    assert rag.detect_framework("under the IT Act, what is section 43A") == "IT_ACT_2000"
