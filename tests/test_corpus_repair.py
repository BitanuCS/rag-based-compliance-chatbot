"""Tests for the ingest-time repairs.

The DPDP fixture below is the real chunk, abridged: the same column-major table
dump, the same heading printed after the table it names, the same trailing press
imprint. It is reproduced here rather than read from the corpus so the tests say
what shape they handle, and keep saying it if the corpus is revised.
"""

import corpus_repair as cr

DPDP_SCHEDULE_CHUNK = {
    "chunk_id": "DPDP_ACT_2023-0045-c01",
    "framework_name": "Digital Personal Data Protection Act, 2023",
    "section_id": "DPDP_ACT_2023-s44",
    "section_title": "Section 44",
    "breadcrumb": "Digital Personal Data Protection Act, 2023 > Chapter IX > 44 Section 44",
    "text": (
        "(3) In section 8 of the Right to Information Act, 2005, for clause (j), the "
        'following clause shall be substituted, namely:- "(j) information which relates '
        'to personal information;".\n'
        "21 of 2000.\n"
        "27 of 2008.\n"
        "Breach of provisions of  this Act or rules made thereunder (2) Breach in observing "
        "the obligation of Data Fiduciary to take reasonable security safeguards under "
        "sub-section (5) of section 8.\n"
        "Breach in observing the obligation to give notice of a personal data breach under "
        "sub-section (6) of section 8.\n"
        "Breach in observance of additional obligations in relation to children under section 9.\n"
        "Sl. No.\n"
        "(1) 1.\n"
        "2. 3.\n"
        "Penalty (3) May extend to two hundred and fifty crore rupees.\n"
        "May extend to two hundred crore rupees.\n"
        "May extend to two hundred crore rupees.\n"
        "THE  SCHEDULE [See section 33 (1)] ----\n"
        "DR. REETA VASISHTA, Secretary to the Govt. of  India.\n"
        "UPLOADED BY THE MANAGER, GOVERNMENT OF INDIA PRESS, MINTO ROAD.\n"
    ),
}

# The Companies Act front matter lists its schedules by name. Splitting here would
# file the Act's own enacting words under "Schedule I".
CONTENTS_CHUNK = {
    "chunk_id": "COMPANIES_ACT_2013-0001-c01",
    "framework_name": "Companies Act, 2013",
    "section_title": "Preamble",
    "breadcrumb": "Companies Act, 2013 > 0 Preamble",
    "text": (
        "469. Power of Central Government to make rules.\n"
        "470. Power to remove difficulties.\n"
        "SCHEDULE I.\n"
        "SCHEDULE II.\n"
        "SCHEDULE III.\n"
        "THE COMPANIES ACT, 2013 ACT NO. 18 OF 2013 An Act to consolidate and amend the "
        "law relating to companies.\n"
    ),
}

ORDINARY_CHUNK = {
    "chunk_id": "IT_ACT_2000-0007-c02",
    "framework_name": "Information Technology Act, 2000",
    "section_title": "Section 43A",
    "breadcrumb": "Information Technology Act, 2000 > Chapter IX > 43A",
    "text": "Where a body corporate is negligent in implementing reasonable security practices.",
}


def _schedule_record(chunk):
    return next((entry for entry in cr.repair(chunk) if entry[0] == "#schedule"), None)


# --- what gets repaired ------------------------------------------------------


def test_ordinary_chunk_passes_through_untouched():
    assert cr.repair(ORDINARY_CHUNK) == [("", ORDINARY_CHUNK["text"], {})]


def test_contents_listing_is_not_treated_as_a_schedule():
    assert len(cr.repair(CONTENTS_CHUNK)) == 1


def test_lowercase_mentions_are_not_headings():
    # "Power to amend Schedule." is a marginal note, not the schedule itself.
    match, _ = cr.find_schedule_heading("Power to amend Schedule.\nSchedule to the Constitution.")
    assert match is None


def test_schedule_is_lifted_into_its_own_record():
    records = cr.repair(DPDP_SCHEDULE_CHUNK)
    assert [suffix for suffix, _, _ in records] == ["", "#schedule"]


# --- how the lifted record is labelled ---------------------------------------


def test_lifted_schedule_gets_its_own_breadcrumb():
    _, _, overrides = _schedule_record(DPDP_SCHEDULE_CHUNK)
    assert overrides["breadcrumb"] == (
        "Digital Personal Data Protection Act, 2023 > THE SCHEDULE [See section 33 (1)]"
    )
    # The chapter it was printed after is not the chapter it belongs to.
    assert "Chapter IX" not in overrides["breadcrumb"]


def test_lifted_schedule_is_flagged_as_a_repair():
    _, _, overrides = _schedule_record(DPDP_SCHEDULE_CHUNK)
    assert overrides["repair"] == "schedule"
    assert overrides["section_number"] == "SCHEDULE"


def test_long_headings_are_trimmed_for_the_breadcrumb():
    heading = "SCHEDULE VII (See section 135) Activities which may be included by companies " \
              "in their Corporate Social Responsibility Policies"
    trimmed = cr.trim_title(heading)
    assert len(trimmed) <= cr.TITLE_LIMIT + 1
    assert trimmed.startswith("SCHEDULE VII")


# --- the table itself --------------------------------------------------------


def test_columns_are_zipped_back_into_rows():
    _, table, _ = _schedule_record(DPDP_SCHEDULE_CHUNK)
    rows = [line for line in table.splitlines() if line[:1].isdigit()]
    assert len(rows) == 3
    # The mapping from breach to amount is the entire content of the table.
    assert "security safeguards under sub-section (5) of section 8" in rows[0]
    assert "two hundred and fifty crore" in rows[0]
    assert "children under section 9" in rows[2]
    assert "two hundred crore" in rows[2]


def test_the_table_leaves_the_body_behind():
    body = cr.repair(DPDP_SCHEDULE_CHUNK)[0][1]
    assert "Right to Information Act" in body
    assert "crore" not in body


def test_press_imprint_is_dropped():
    body = cr.repair(DPDP_SCHEDULE_CHUNK)[0][1]
    assert "UPLOADED BY" not in body
    assert "REETA VASISHTA" not in body


def test_unrecognised_table_shape_is_left_alone():
    # Two columns, no serial column: not a shape this understands, so it must not
    # guess. Corrupting text that was fine is worse than leaving it unrepaired.
    assert cr.reflow_schedule_table("Penalty (3) fifty crore rupees.\nAnother line.") is None


def test_mismatched_column_lengths_are_left_alone():
    text = (
        "Breach A (2) first breach.\nsecond breach.\n"
        "Sl. No.\n(1) 1.\n2. 3.\n"  # three serials
        "Penalty (3) ten rupees.\ntwenty rupees.\n"  # two penalties
        "THE SCHEDULE [See section 1]\n"
    )
    assert cr.reflow_schedule_table(text) is None
