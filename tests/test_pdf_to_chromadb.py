"""Tests for the chunker's structure rules — no PDFs, no network.

Each case is a shape that broke a real document while the pipeline was built:
footnote numbers read as sections, a contents list read as the Act, margin
notes landing mid-sentence. The PDFs themselves are checked by `pdf_to_chromadb.py
chunk`'s validation and by eval_retrieval.py; these pin the rules.
"""

import pdf_to_chromadb as P


def _lines(*texts, page=1):
    return [(page, text) for text in texts]


def _sections(lines, strategy):
    return [h.number for h in P.find_headings(lines, strategy) if h.kind == "section"]


# --- gazette sections -----------------------------------------------------------


def test_gazette_sections_and_titles():
    lines = _lines(
        "1. Short title and commencement.—(1) This Act may be called…",
        "(2) It shall come into force…",
        "2. Definitions.—In this Act,—",
    )
    headings = [h for h in P.find_headings(lines, "gazette") if h.kind == "section"]
    assert [(h.number, h.title) for h in headings] == [
        ("1", "Short title and commencement"),
        ("2", "Definitions"),
    ]


def test_a_numbered_list_inside_a_section_is_not_new_sections():
    # Section 40's list restarts at 1; only a plausible next number is a heading.
    lines = _lines(
        "39. Allotment.—(1) No allotment shall be made…",
        "40. Securities to be dealt with.—(1) Every company…",
        "1. the first item;",
        "2. the second item;",
        "41. Global depository receipt.—A company may…",
    )
    assert _sections(lines, "gazette") == ["39", "40", "41"]


def test_inserted_and_bracketed_sections_keep_the_sequence():
    # "3A" is an inserted section; "[270." is how an amended one is printed.
    lines = _lines(
        "3. Formation of company.—(1) A company may be formed…",
        "3A. Members severally liable.—If at any time…",
        "4. Memorandum.—(1) The memorandum…",
        "[5.Articles.—(1) The articles…",
    )
    assert _sections(lines, "gazette") == ["3", "3A", "4", "5"]


def test_numbering_resumes_after_omitted_sections():
    lines = _lines("10. Ten.—x", "13. Thirteen.—x")  # 11 and 12 omitted
    assert _sections(lines, "gazette") == ["10", "13"]


# --- other styles ---------------------------------------------------------------


def test_articles_take_their_title_from_the_next_line():
    lines = _lines("Article 5", "Principles relating to processing", "1. Personal data shall be…")
    (heading,) = [h for h in P.find_headings(lines, "article") if h.kind == "section"]
    assert (heading.number, heading.title) == ("Article 5", "Principles relating to processing")


def test_decimal_clauses_carry_their_depth():
    lines = _lines("7 Support", "7.5 Documented information", "7.5.2", "Creating and updating", "When creating…")
    headings = [h for h in P.find_headings(lines, "decimal") if h.kind == "section"]
    assert [(h.number, h.level) for h in headings] == [("7", 1), ("7.5", 2), ("7.5.2", 3)]
    assert headings[-1].title == "Creating and updating"


def test_a_sentence_is_not_a_decimal_heading():
    assert not _sections(_lines("4.1 The Regulated Entity shall maintain records."), "decimal")


def test_the_style_with_most_headings_wins():
    lines = _lines(*(f"{n}. Section {n}.—text" for n in range(1, 8)))
    assert P.choose_strategy(lines)[0] == "gazette"


# --- chapters and schedules -----------------------------------------------------


def test_a_schedule_starts_its_own_section_and_ends_numbering():
    lines = _lines(
        "44. Amendments.—In section 14…",
        "THE SCHEDULE",
        "1. Breach of security safeguards | Penalty: up to ₹250 crore",
    )
    headings = P.find_headings(lines, "gazette")
    assert [(h.kind, h.number) for h in headings] == [("section", "44"), ("schedule", "Schedule")]


def test_prose_mentioning_a_schedule_is_not_one():
    assert not P.find_headings(_lines("Schedule to the Constitution applies."), "gazette")


def test_chapter_title_is_read_from_the_next_capitalised_line():
    (chapter,) = [h for h in P.find_headings(_lines("CHAPTER II", "OBLIGATIONS OF DATA FIDUCIARY"), "gazette")]
    assert chapter.title == "Obligations Of Data Fiduciary"


# --- print furniture --------------------------------------------------------------


def test_a_contents_list_is_dropped_when_its_numbers_come_back():
    toc = [f"{n}. Title of section {n}." for n in range(1, 16)]
    body = []
    for n in range(1, 16):
        body += [f"{n}. Title of section {n}.—(1) Body text of section {n}.", "More body text.", "Even more."]
    lines = _lines(*toc, *body)
    kept = P._drop_toc(lines, {})
    assert [text for _, text in kept] == body


def test_a_short_run_of_headings_is_not_a_contents_list():
    lines = _lines(*(f"{n}. Title {n}." for n in range(1, 5)), *(f"{n}. Title {n}.—body" for n in range(1, 5)))
    assert len(P._drop_toc(lines, {})) == len(lines)


def test_margin_note_becomes_the_section_title():
    items = [(100.0, 140.0, ["26. The Chairperson shall exercise the following powers"])]
    notes = [(98.0, "Powers of"), (107.0, "Chairperson.")]
    assert P._attach_margin_titles(items, notes) == 1
    assert items[0][2] == ["26. Powers of Chairperson.—The Chairperson shall exercise the following powers"]


def test_a_section_that_already_has_a_title_is_left_alone():
    items = [(100.0, 140.0, ["149. Company to have Board of Directors.—(1) Every company"])]
    assert P._attach_margin_titles(items, [(100.0, "Board.")]) == 0


# --- reflow and breadcrumbs -------------------------------------------------------


def test_reflow_joins_wrapped_lines_and_keeps_clause_breaks():
    text = P.reflow([
        "4. (1) A person may process the personal data of a Data",
        "Principal only in accordance with the provisions of this Act,—",
        "(a) for which the Data Principal has given her consent; or",
        "(b) for certain legitimate uses.",
    ])
    assert text.splitlines() == [
        "4. (1) A person may process the personal data of a Data Principal only in "
        "accordance with the provisions of this Act,—",
        "(a) for which the Data Principal has given her consent; or",
        "(b) for certain legitimate uses.",
    ]


def test_reflow_rejoins_hyphenated_words():
    assert P.reflow(["the obliga-", "tion of the Board"]) == "the obligation of the Board"


def test_breadcrumb_runs_framework_chapter_parents_section():
    section = {"number": "7.5.2", "title": "Creating and updating", "chapter": None,
               "parents": ["7 Support", "7.5 Documented information"]}
    assert P.breadcrumb("ISO/IEC 27001:2022", section) == (
        "ISO/IEC 27001:2022 > 7 Support > 7.5 Documented information > 7.5.2 Creating and updating"
    )


def test_page_lookup_follows_the_offsets():
    offsets = [(0, 3), (120, 4), (300, 5)]
    assert [P._page_at(offsets, at) for at in (0, 119, 120, 450)] == [3, 3, 4, 5]


# --- gazette tables without ruling lines -------------------------------------------


class _Page:
    """Just enough of a PyMuPDF page: positioned words and a rect."""

    def __init__(self, words):
        import pymupdf

        self._words, self.rect = words, pymupdf.Rect(0, 0, 595, 842)

    def get_text(self, kind, sort=False):
        assert kind == "words"
        return self._words


def _w(x0, x1, y, text):
    return (x0, y, x1, y + 9, text, 0, 0, 0)


# Word positions taken from the DPDP Act's penalty schedule (page 21).
SCHEDULE = [
    _w(122, 133, 154, "Sl."), _w(134, 148, 154, "No."),
    _w(165, 193, 154, "Breach"), _w(195, 203, 154, "of"), _w(205, 246, 154, "provisions"),
    _w(426, 456, 154, "Penalty"),
    _w(130, 141, 172, "(1)"), _w(271, 282, 172, "(2)"), _w(436, 447, 172, "(3)"),
    _w(132, 139, 196, "1."), _w(165, 193, 196, "Breach"), _w(196, 204, 196, "in"),
    _w(206, 246, 196, "observing"), _w(405, 423, 196, "May"), _w(425, 451, 196, "extend"),
    _w(165, 182, 208, "safeguards."), _w(405, 438, 208, "hundred"), _w(442, 456, 208, "and"),
    _w(460, 477, 208, "fifty"),
    _w(132, 139, 232, "2."), _w(165, 193, 232, "Breach"), _w(196, 204, 232, "of"),
    _w(206, 246, 232, "notice."), _w(405, 423, 232, "Two"), _w(425, 451, 232, "hundred"),
    _w(130, 200, 300, "UPLOADED"), _w(205, 450, 300, "BY-THE-PRESS-ACROSS-ALL-COLUMNS"),
]


def test_a_gazette_table_is_rebuilt_row_by_row_from_its_column_numbers():
    _, lines = P._column_number_table(_Page(SCHEDULE))
    assert lines == [
        "Sl. No.: 1. | Breach of provisions: Breach in observing safeguards. | Penalty: May extend hundred and fifty",
        "Sl. No.: 2. | Breach of provisions: Breach of notice. | Penalty: Two hundred",
    ]


def test_sub_section_numbers_in_prose_are_not_a_column_row():
    prose = [_w(120, 135, 100, "(1)"), _w(140, 300, 100, "Where"), _w(310, 325, 100, "(2)"),
             _w(132, 139, 130, "1."), _w(132, 139, 150, "2.")]
    assert P._column_number_table(_Page(prose)) is None


# --- clause splitting -------------------------------------------------------------
#
# A record is one clause by the document's own numbering. Limits are shrunk here
# so a few lines exercise what a 2,000-token section does in the real corpus.


import pytest

from vectorstore import token_length


@pytest.fixture
def small_limits(monkeypatch):
    monkeypatch.setattr(P, "CLAUSE_MAX_TOKENS", 40)
    monkeypatch.setattr(P, "CLAUSE_MIN_TOKENS", 10)


def _parts(body):
    return [body[a:b].strip() for a, b, _ in P.split_clauses(body)]


SECTION = (
    "8. General obligations.—\n"
    "(1) A Data Fiduciary shall be responsible for complying with the provisions of this Act.\n"
    "(2) A Data Fiduciary may engage a Data Processor only under a valid contract.\n"
    "Provided that the contract shall bind the Data Processor to these obligations.\n"
    "(3) Where personal data is likely to be—\n"
    "(a) used to make a decision that affects the Data Principal; or\n"
    "(b) disclosed to another Data Fiduciary, the data shall be complete and accurate.\n"
)


def test_a_section_splits_at_its_numbering_whatever_its_size(small_limits):
    body = (
        "(1) A Data Fiduciary shall maintain the accuracy of personal data it processes.\n"
        "(2) A Data Fiduciary shall erase personal data once its purpose is served.\n"
    )
    assert [p[:3] for p in _parts(body)] == ["(1)", "(2)"]


def test_tiny_numbered_pieces_are_merged_back_together():
    body = "(1) A short rule.\n(2) Another short rule.\n"
    assert _parts(body) == [body.strip()]


def test_a_section_without_numbering_stays_whole():
    body = "This guideline applies to every regulated entity.\nIt takes effect at once.\n"
    assert _parts(body) == [body.strip()]


def test_sections_split_at_their_numbering_and_rejoin_exactly(small_limits):
    spans = P.split_clauses(SECTION)
    assert "".join(SECTION[a:b] for a, b, _ in spans) == SECTION
    assert [p[:3] for p in _parts(SECTION)] == ["8. ", "(2)", "(3)", "(a)", "(b)"]


def test_a_proviso_stays_with_the_rule_it_qualifies(small_limits):
    (two,) = [p for p in _parts(SECTION) if p.startswith("(2)")]
    assert two.endswith("bind the Data Processor to these obligations.")


def test_sub_clauses_carry_the_stem_they_complete(small_limits):
    spans = P.split_clauses(SECTION)
    contexts = {SECTION[a:b].strip()[:3]: ctx for a, b, ctx in spans}
    assert contexts["(b)"].endswith("is likely to be—")
    # A heading is not a stem: the breadcrumb already says where (2) sits.
    assert contexts["(2)"] == ""


def test_fragments_merge_into_their_neighbour(small_limits):
    parts = _parts(SECTION)
    # "8. General obligations.—" alone is under the minimum; it opens (1)'s part.
    assert parts[0].startswith("8. General obligations.—\n(1)")


def test_unnumbered_text_splits_at_lines_within_the_limit(small_limits):
    body = "\n".join(f"Row {n} | control objective text for testing purposes here" for n in range(12))
    parts = _parts(body)
    assert len(parts) > 1
    assert all(token_length(p) <= P.CLAUSE_MAX_TOKENS for p in parts)
    assert all(p.startswith("Row ") for p in parts)
