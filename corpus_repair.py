"""Repairs applied to corpus chunks on the way into the index.

The corpus arrives pre-chunked from a PDF extraction pipeline this project does
not own, and that pipeline has a blind spot: a schedule or annexure printed at
the back of an Act does not become its own section. It is swallowed by whatever
section chunk happens to be last, and it keeps that section's breadcrumb.

That matters because the breadcrumb is embedded. The DPDP Act's penalty
schedule — the text that says a security-safeguards breach may cost ₹250 crore —
sits inside a chunk labelled `Chapter IX > 44 Section 44`, so a search for "the
maximum penalty under the DPDP Act" matches nothing: the words the reader used
are in a table whose label talks about amendments to other Acts. The system then
answers, correctly and uselessly, that it cannot find a maximum penalty.

The second half of the same blind spot is that a table is extracted column by
column. Every breach description arrives first, then every serial number, then
every penalty amount — so the mapping from breach to rupee figure, which is the
entire content of the table, is gone by the time it reaches the index.

Both repairs are structural, not keyed to a document: a chunk is repaired only
when its text carries a schedule heading, and a table is reflowed only when its
three columns are found and have the same number of rows. Anything else passes
through untouched, because a repair that fires on a shape it does not understand
would corrupt text that was fine.
"""

import re

# Legal headings are set in capitals, which is what separates the real heading
# "THE  SCHEDULE [See section 33 (1)]" from prose like "Schedule to the
# Constitution." or the marginal note "Power to amend Schedule." Matching
# case-insensitively here would relabel a chunk on the strength of a passing
# mention, which is worse than not repairing it at all.
SCHEDULE_HEADING_RE = re.compile(
    r"^[ \t]*("
    r"(?:THE[ \t]+)?SCHEDULE(?:[ \t]*[-–—]?[ \t]*[IVXLC\d]+)?"
    r"|(?:THE[ \t]+)?ANNEXURE(?:[ \t]*[-–—]?[ \t]*[IVXLC\dA-Z]+)?"
    r"|APPENDIX(?:[ \t]*[-–—]?[ \t]*[IVXLC\dA-Z]+)?"
    r")\b[^\n]*",
    re.MULTILINE,
)

# The serial-number column of a table, on its own line. Its position is the anchor
# the rest of the reflow works outwards from.
SERIAL_HEADER_RE = re.compile(r"^[ \t]*(?:Sl\.?|S\.?|Serial)[ \t]*No\.?[ \t]*$", re.MULTILINE)

# The third column's header. `(3)` is the column number the draftsman prints
# under each heading, and requiring it keeps this from matching the word
# "Penalty" in running text.
PENALTY_HEADER_RE = re.compile(r"^[ \t]*(Penalty|Fine|Amount)[ \t]*\(3\)[ \t]*", re.MULTILINE)

# Trailing print-shop lines: the press imprint and the signing officer are not
# part of the schedule and only dilute it.
TRAILING_NOISE_RE = re.compile(
    r"^(?:UPLOADED BY|MGIPMRND|PRINTED BY|PUBLISHED BY|DR\.|SHRI |MS\. |MR\. ).*$",
    re.MULTILINE,
)


# A heading with nothing after the numeral — "SCHEDULE I." — is usually a table
# of contents line, not the schedule itself. The Companies Act front matter lists
# all seven that way, and splitting the preamble there would file the Act's own
# enacting words under "Schedule I".
BARE_HEADING_RE = re.compile(
    r"^(?:THE[ \t]+)?(?:SCHEDULE|ANNEXURE|APPENDIX)[ \t]*[-–—]?[ \t]*[IVXLC\d]*[ \t]*[.:)]?$",
    re.IGNORECASE,
)

# Long enough to identify the schedule, short enough to stay a breadcrumb. The
# extractor often runs the schedule's first sentence onto the heading line.
TITLE_LIMIT = 80


def _clean_heading(match):
    """The heading text, minus the rule characters PDFs print around titles."""
    return " ".join(match.group(0).split()).rstrip("- —–_.")


def trim_title(heading, limit=TITLE_LIMIT):
    if len(heading) <= limit:
        return heading
    return heading[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-([") + "…"


def find_schedule_heading(text):
    """The schedule/annexure heading in `text`, or (None, None).

    Returns the match rather than a bool: the caller needs both the cleaned-up
    heading string, for the breadcrumb, and its position, because in a
    column-extracted PDF the heading routinely lands *after* the table it names.

    A chunk listing several bare headings is a contents page, so bare headings
    are only trusted when one stands alone.
    """
    matches = [(m, _clean_heading(m)) for m in SCHEDULE_HEADING_RE.finditer(text)]
    matches = [(m, h) for m, h in matches if h]
    if not matches:
        return None, None

    titled = [(m, h) for m, h in matches if not BARE_HEADING_RE.match(h)]
    if titled:
        return titled[0]
    return matches[0] if len(matches) == 1 else (None, None)


def _rows_before(region, count):
    """Split the breach column out of `region`, or (None, None).

    The column header ends with its column number — "…thereunder (2)" — and the
    first row follows on the same line. Which `(2)` is the header is decided by
    counting: the one that leaves exactly `count` rows behind it is the one that
    opened the column. Guessing by position instead would trip over the
    "sub-section (2)" references that legal text is full of.
    """
    for match in re.finditer(r"\(2\)", region):
        rows = [line.strip() for line in region[match.end() :].split("\n") if line.strip()]
        if len(rows) == count:
            line_start = region.rfind("\n", 0, match.start()) + 1
            header = region[line_start : match.start()].strip()
            return rows, (header, line_start)
    return None, None


def reflow_schedule_table(text):
    """Rebuild a column-major table dump into rows, or return None.

    Returns (start, end, table_text): the span of `text` the table occupies and
    the readable replacement. None means the shape was not recognised, and the
    caller must leave the text exactly as it was.
    """
    serial = SERIAL_HEADER_RE.search(text)
    if not serial:
        return None

    penalty = PENALTY_HEADER_RE.search(text, serial.end())
    if not penalty:
        return None

    # Column 1: the serial numbers, printed two or three to a line by the
    # extractor. The count they give is what every other column is checked against.
    serials = re.findall(r"\b(\d+)\.", text[serial.end() : penalty.start()])
    if len(serials) < 2:
        return None

    # Column 3: exactly as many lines as there are serial numbers. Taking a fixed
    # count is what stops the press imprint below the table being read as a penalty.
    tail = [line.strip() for line in text[penalty.end() :].split("\n") if line.strip()]
    penalties = tail[: len(serials)]
    if len(penalties) < len(serials):
        return None

    breaches, header_info = _rows_before(text[: serial.start()], len(serials))
    if not breaches:
        return None
    breach_header, table_start = header_info

    heading_match, heading = find_schedule_heading(text)
    title = heading or "SCHEDULE"

    # The table ends at the last penalty line. Finding it by search rather than
    # arithmetic keeps the span honest when the extractor doubled a newline.
    last = text.find(penalties[-1], penalty.end())
    table_end = last + len(penalties[-1]) if last != -1 else len(text)
    # The heading itself belongs to the table even when it was printed after it.
    if heading_match and heading_match.start() >= table_end:
        table_end = heading_match.end()

    # Column-extracted PDF text carries the original line-breaking as stray double
    # spaces; a table row is one sentence, so it is squeezed flat.
    def flat(value):
        return " ".join(value.split())

    lines = [title, "", f"Sl. No. | {flat(breach_header)} | Penalty"]
    for number, breach, amount in zip(serials, breaches, penalties):
        lines.append(f"{number}. {flat(breach)} — Penalty: {flat(amount)}")

    return table_start, table_end, "\n".join(lines)


def _schedule_breadcrumb(chunk, title):
    """Breadcrumb for a lifted schedule: the document, then the schedule itself.

    The chapter is dropped deliberately. A schedule is not inside the chapter it
    was printed after, and saying it is would put the same wrong label back on
    the text this whole module exists to relabel.
    """
    return f"{chunk['framework_name']} > {title}"


def repair(chunk):
    """A chunk in, the records it should become out.

    Returns a list of (suffix, text, metadata overrides). An unrepaired chunk
    yields a single entry with an empty suffix and no overrides, so callers can
    treat every chunk the same way.
    """
    text = chunk["text"]
    plain = [("", text, {})]

    _, heading = find_schedule_heading(text)
    if not heading:
        return plain

    # A chunk that is already labelled as the schedule needs no relabelling; it
    # may still need its table reflowed in place.
    labelled = "schedule" in (chunk.get("section_title") or "").lower() or "annexure" in (
        chunk.get("section_title") or ""
    ).lower()

    reflowed = reflow_schedule_table(text)
    if reflowed:
        start, end, table = reflowed
        body = (text[:start] + text[end:]).strip()
        body = TRAILING_NOISE_RE.sub("", body)
        # Lifting a span out of the middle leaves the gap behind as blank lines.
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
    else:
        # No table shape to lift, so the heading alone drives the split: whatever
        # follows it is the schedule.
        match, _ = find_schedule_heading(text)
        if match.start() == 0 or labelled:
            return plain
        table = text[match.start() :].strip()
        body = text[: match.start()].strip()

    if not table:
        return plain

    title = trim_title(" ".join(heading.split()))
    overrides = {
        "section_id": f"{chunk.get('section_id', chunk['chunk_id'])}-schedule",
        "section_number": "SCHEDULE",
        "section_title": title,
        "breadcrumb": _schedule_breadcrumb(chunk, title),
        # Flagged, because this record is not a corpus chunk in its own right: it
        # was lifted out of one, and `ingest.py stats` checks the index against
        # the corpus's own per-framework chunk counts.
        "repair": "schedule",
    }

    records = []
    if body:
        records.append(("", body, {}))
    records.append(("#schedule", table, overrides))
    return records
