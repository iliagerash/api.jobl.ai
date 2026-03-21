"""
cleaner.py
──────────
Normalises raw job-board HTML into clean, structured markup and extracts
the application deadline when present.

Keeps all content (company intro, benefits, boilerplate) — only fixes
structure and encoding. Works on EN and FR postings.

Transformations applied
-----------------------
1.  URL-decode injected encoded markup  (e.g. %3Cspan…)
2.  Strip malformed !*!<…> sentinel blocks  (rare job-board artefact)
3.  Unwrap layout-only tags: <span>, <font>, <div>, <table>, <center>, …
4.  Strip all HTML attributes  (style=, class=, id=, …)
5.  Split <strong>/<b> on interior <br>  → sibling bold elements + bare <br>
6.  Unwrap nested <strong><strong> and <b><b>
7.  Merge consecutive sibling bold tags that are both short header fragments
    — but NOT when one text is a substring of the other (parent+child headers)
8.  Promote fully standalone <strong>/<b> to <h3>  (section headers)
9.  Collapse <br>-delimited runs into proper <p> blocks
10. Split <p> blocks containing blank lines (\\n\\n) into sibling <p> elements
11. Promote <strong>/<b> at the START of a <p> to <h3> + <p>
    — skipped for short inline label:value pairs  ("Posting ID: 5064")
12. Fix invalid <h3> nested inside <p>  (hoist h3 out, split content)
13. Unwrap double-nested <p><p>  (lxml re-parse artefact)
14. Wrap bare text nodes in <p>
15. Normalize whitespace within <p>/<li> text nodes  (collapse \\n and spaces)
16. Drop empty block tags
17. Enforce allowed tag set: p h2 h3 h4 ul ol li strong em a

Expiry extraction
-----------------
Recognises these field patterns (EN + FR):
  Application Deadline: March 19, 2026
  Deadline: 2026-03-19
  Closing Date: 31/03/2026
  Unposting Date: Ongoing          → None  (open-ended)
  Date limite pour postuler: 2 avril 2026
  Date de clôture: 2026-04-30
  apply by April 1, 2026           (inline prose)

CleanResult.expiry returns:
  datetime.date  — future deadline found
  "expired"      — deadline has already passed
  None           — not found, or open-ended ("Ongoing")

Public API
----------
    from app.services.cleaner import clean_job_description, CleanResult

    result = clean_job_description(raw_html)
    result.html        # clean HTML fragment
    result.expiry      # date | "expired" | None
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from bs4 import BeautifulSoup, NavigableString, Tag

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRACKING_RE = re.compile(r"^#[A-Z][A-Z0-9\-]+$")

# Matches a NavigableString whose last word is an article/preposition,
# signalling that a following <strong> is inline (mid-sentence) rather than
# a section header.  Used to suppress spurious paragraph-flush in _split_on_brs.
_MID_SENTENCE_ENDS_RE = re.compile(
    r"\b(a|an|the|for|in|of|to|with|as|at|by|on|or|and|from|our|their|"
    r"your|this|that|its|have|has|is|are|was|were|be|been)\s*$",
    re.IGNORECASE,
)

_LAYOUT_TAGS = {
    "span", "font", "div", "center",
    "table", "tbody", "thead", "tfoot", "tr", "td", "th",
    "section", "article", "aside", "header", "footer", "main", "nav",
}

_BLOCK_TAGS = {
    "p", "ul", "ol",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "pre", "figure", "hr",
}

_ALLOWED_TAGS = {"p", "h2", "h3", "h4", "ul", "ol", "li", "strong", "em", "a"}

# ---------------------------------------------------------------------------
# Expiry extraction
# ---------------------------------------------------------------------------

_FR_MONTHS: dict[str, int] = {
    "janvier": 1, "février": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "août": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12,
}

# English month names (full + abbreviated) for Day-Month-Year parsing
_EN_MONTHS: dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# Patterns that introduce a deadline field label (EN + FR, expanded)
_DEADLINE_LABEL_RE = re.compile(
    r"""
    (?:
        application\s+deadline
      | unposting\s+date
      | (?:job\s+)?posting\s+down\s+date
      | closing\s+(?:date|on)
      | close\s+date
      | deadline
      | position\s+closes
      | apply\s+(?:by\s+date|before)
      | posting\s+(?:closes|end\s+date|expiration\s+date)
      | job\s+posting\s+(?:end|expiration)\s+date
      | applications?\s+(?:will\s+)?close
      | applications?\s+due
      | expir(?:y|ation)\s+date
      | application\s+window\b.{0,60}close\b
      | (?:position|job|work)\s+start\s+date
      | (?:hired|offered)\b.{0,120}(?:between|from)\b.{0,120}\band\b
      | open\s+until
      | posted\s+until
      | no\s+later\s+than
      | prior\s+to\b
      | ends\s+on
      | accepted\s+through
      | applications?\s+received\s+by
      | last\s+application\s+date
      | date\s+limite\s+(?:pour\s+)?(?:postuler|de\s+candidature)?
      | date\s+de\s+(?:cl[oô]ture|fermeture)
      | date\s+de\s+fin\s+d[''\u2019]affichage\b
      | date\s+d[''\u2019]affichage\b
      | fin\s+d[e]\s+l[''\u2019]affichage\b
      | fin\s+d[''\u2019]affichage\b
      | p[eé]riode\s+d[''\u2019]inscription\b
      | avant\s+le
      | candidatures?\s+re[cç]ues?\s+jusqu(?:[''\u2019]|\s+)au
      | fermeture\s+du\s+concours
      | [eé]ch[eé]ance\s+(?:de\s+l[''\u2019]affichage\b)?
      | au\s+plus\s+tard\s+le
      | (?:cette\s+offre\s+)?expirer[a]?\s+le
      | offre\s+(?:se\s+termine|valide\s+jusqu(?:[''\u2019]|\s+)au)
    )
    \s*:?\s*
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Inline prose: "apply by April 1, 2026" / "apply online by February 16, 2026"
# Allows up to 3 optional words between "apply" and "by"; captures remainder of line
# so the date can appear on the next line when wrapped in a <span>.
_INLINE_APPLY_BY_RE = re.compile(
    r"apply\s+(?:\w+\s+){0,3}by\s*(.*)",
    re.IGNORECASE,
)

# Inline prose: "submit a résumé by Wednesday, March 18, 2026"
# Note: (.*) — may capture only a weekday ("Wednesday,") when the date is in
# the next text node (span), or be empty when entirely in the next node.
_INLINE_RESUME_BY_RE = re.compile(
    r"r[eé]sum[eé]s?\s+by\s*(.*)",
    re.IGNORECASE,
)

# Inline prose: "Submit your application via our careers' website by 8 March 2026"
#               "Submit your CV by April 30, 2026"
# Uses negative lookahead to stop consuming tokens once "by" is reached.
_INLINE_SUBMIT_BY_RE = re.compile(
    r"submit\w*\s+(?:(?!by[\s,])\S+\s+){0,8}by\s*(.*)",
    re.IGNORECASE,
)

# Inline prose: "posting will close at 11:59 pm MST on March 16, 2026"
#               "job posting will close on April 30, 2026"
#               "this position closes on 2026-03-31"
# Note: (.*)  — may be empty when the date falls in the next text node (span).
_INLINE_CLOSE_ON_RE = re.compile(
    r"(?:(?:job|this)\s+)?(?:posting|position)\s+(?:will\s+)?(?:close[sd]?|expire[sd]?)\b.{0,60}?\bon\b\s*(.*)",
    re.IGNORECASE,
)

# Inline prose: "applications will be accepted until March 13, 2026"
#               "Mattress Firm is accepting applications until: 03/31/2026"
# The date may be split across multiple lines when each word is in its own <strong>.
_INLINE_ACCEPTED_UNTIL_RE = re.compile(
    r"accept(?:ed|ing\s+\w+)\s+until\s*:?\s*(.*)",
    re.IGNORECASE,
)

# Open-ended values — no expiry implied
_OPEN_ENDED_RE = re.compile(r"^\s*(ongoing|until\s+filled|open)\s*$", re.IGNORECASE)

# EN date: "March 19, 2026" / "April 1, 2026"
_EN_DATE_RE = re.compile(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})")
# FR date: "2 avril 2026"
_FR_DATE_RE = re.compile(r"(\d{1,2})\s+([a-zéûôàî]+)\s+(\d{4})", re.IGNORECASE)
# Ordinal date: "22nd March 2026" / "1st April 2026" / "3rd March 2026"
_ORDINAL_DATE_RE = re.compile(r"(\d{1,2})(?:st|nd|rd|th)\s+([A-Za-z]+)\s+(\d{4})", re.IGNORECASE)
# Oracle/ATS abbreviated-month date: "16-MAR-2026"
_DD_MON_YYYY_RE = re.compile(r"(\d{1,2})-([A-Za-z]{3})-(\d{4})")
_MON_ABBR: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_date(text: str, mm_dd_first: bool = False) -> date | None:
    """Parse a date string. Returns None if unparseable or open-ended.

    mm_dd_first: when True, try MM/DD before DD/MM for ambiguous numeric dates.
    """
    text = text.strip()
    if not text or _OPEN_ENDED_RE.match(text):
        return None

    # Numeric YYYY-MM-DD
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # Oracle/ATS DD-MON-YYYY (e.g. 16-MAR-2026)
    m = _DD_MON_YYYY_RE.search(text)
    if m:
        month = _MON_ABBR.get(m.group(2).lower())
        if month:
            try:
                return date(int(m.group(3)), month, int(m.group(1)))
            except ValueError:
                pass

    # Numeric DD/MM/YYYY or MM/DD/YYYY or DD-MM-YYYY
    m = re.search(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", text)
    if not m:
        # Try 2-digit year: M/D/YY or D/M/YY
        m2 = re.search(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2})\b", text)
        if m2:
            a, b, yy = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            year = 2000 + yy
            pairs = [(a, b), (b, a)] if mm_dd_first else [(b, a), (a, b)]
            for month, day in pairs:
                try:
                    return date(year, month, day)
                except ValueError:
                    pass
    if m:
        a, b, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # Order of tries depends on detected format: MM/DD (a=month, b=day) or DD/MM
        pairs = [(a, b), (b, a)] if mm_dd_first else [(b, a), (a, b)]
        for month, day in pairs:
            try:
                return date(year, month, day)
            except ValueError:
                pass

    # Ordinal DD Month YYYY: "22nd March 2026", "1st April 2026"
    m = _ORDINAL_DATE_RE.search(text)
    if m:
        month = _EN_MONTHS.get(m.group(2).lower())
        if month:
            try:
                return date(int(m.group(3)), month, int(m.group(1)))
            except ValueError:
                pass

    # French first (prevents EN parser from misreading "2 avril 2026")
    m = _FR_DATE_RE.search(text)
    if m:
        day = int(m.group(1))
        month_str = m.group(2).lower()
        year = int(m.group(3))
        month = _FR_MONTHS.get(month_str) or _EN_MONTHS.get(month_str)
        if month:
            try:
                return date(year, month, day)
            except ValueError:
                pass
        else:
            # Unknown month name — try dateutil as last resort
            try:
                from dateutil import parser as du
                return du.parse(m.group(0), dayfirst=True).date()
            except Exception:
                pass

    # English via dateutil
    m = _EN_DATE_RE.search(text)
    if m:
        try:
            from dateutil import parser as du
            return du.parse(m.group(0), dayfirst=False).date()
        except Exception:
            pass

    return None


_NUMERIC_DATE_RE = re.compile(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})")

# Month + ordinal day with no year, e.g. "April 12th" / "April 12"
_PARTIAL_EN_DATE_RE = re.compile(
    r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\b(?!\d)",
    re.IGNORECASE,
)


def _parse_partial_date(text: str) -> date | None:
    """Parse a month+day string with no year, inferring current or next year.

    Only used as a fallback after a hard deadline label (e.g. "accepted through
    April 12th") when the full-year parser found nothing.  Skips text that
    already contains a 4-digit year (those should have been handled by
    _parse_date).
    """
    if re.search(r"\d{4}", text):
        return None  # year present — leave to _parse_date
    m = _PARTIAL_EN_DATE_RE.search(text)
    if not m:
        return None
    month = _EN_MONTHS.get(m.group(1).lower())
    if not month:
        return None
    day = int(m.group(2))
    today = date.today()
    for year in (today.year, today.year + 1):
        try:
            d = date(year, month, day)
            if d >= today:
                return d
        except ValueError:
            pass
    return None


def _detect_mm_dd(text: str) -> bool:
    """Return True if unambiguous numeric dates in text use MM/DD format.

    Scans for dates where the first component (a) exceeds 12 (→ must be a day,
    so format is DD/MM — return False) or the second component (b) exceeds 12
    (→ must be a day, so format is MM/DD — return True).  Returns False (DD/MM
    default) when no unambiguous signal is found.
    """
    for m in _NUMERIC_DATE_RE.finditer(text):
        a, b = int(m.group(1)), int(m.group(2))
        if b > 12:
            return True   # second component can't be month → MM/DD
        if a > 12:
            return False  # first component can't be month → DD/MM
    return False  # no signal: default to DD/MM


def _extract_expiry_from_text(full_text: str) -> date | None:
    """Scan plain text for deadline patterns, return first parseable date.

    Returns None for open-ended postings ("Ongoing") and for no match.
    """
    mm_dd = _detect_mm_dd(full_text)
    lines = [l.strip() for l in full_text.splitlines() if l.strip()]

    for i, line in enumerate(lines):
        # Label-based: value may follow on the same line or the next
        m = _DEADLINE_LABEL_RE.search(line)
        if m:
            after = line[m.end():].strip()
            if after:
                if _OPEN_ENDED_RE.match(after):
                    return None
                d = _parse_date(after, mm_dd) or _parse_partial_date(after)
                if d is not None:
                    return d
            if i + 1 < len(lines):
                nxt = lines[i + 1]
                if _OPEN_ENDED_RE.match(nxt):
                    return None
                d = _parse_date(nxt, mm_dd) or _parse_partial_date(nxt)
                if d is not None:
                    return d
                # Date may span two lines, e.g. "Friday," + "March 6, 2026"
                if i + 2 < len(lines):
                    combined = nxt + " " + lines[i + 2]
                    d = _parse_date(combined, mm_dd) or _parse_partial_date(combined)
                    if d is not None:
                        return d

        # Inline prose: "apply by April 1, 2026" / "apply online by February 16, 2026"
        m2 = _INLINE_APPLY_BY_RE.search(line)
        if m2:
            captured = m2.group(1).strip()
            d = _parse_date(captured, mm_dd) if captured else None
            if d is None and i + 1 < len(lines):
                # Date may follow on the next line (span split), possibly after a weekday
                d = _parse_date(captured + " " + lines[i + 1], mm_dd) or _parse_date(lines[i + 1], mm_dd)
            if d is not None:
                return d

        # Inline: "submit a résumé by Wednesday, March 18, 2026"
        m_res = _INLINE_RESUME_BY_RE.search(line)
        if m_res:
            captured = m_res.group(1).strip()
            d = _parse_date(captured, mm_dd) if captured else None
            if d is None and i + 1 < len(lines):
                # Date may follow on the next line (span split), possibly after a weekday
                d = _parse_date(captured + " " + lines[i + 1], mm_dd) or _parse_date(lines[i + 1], mm_dd)
            if d is not None:
                return d

        # Inline: "Submit your application via our careers' website by 8 March 2026"
        m_sub = _INLINE_SUBMIT_BY_RE.search(line)
        if m_sub:
            captured = m_sub.group(1).strip()
            d = _parse_date(captured, mm_dd) if captured else None
            if d is None and i + 1 < len(lines):
                d = _parse_date(captured + " " + lines[i + 1], mm_dd) or _parse_date(lines[i + 1], mm_dd)
            if d is not None:
                return d

        # Inline prose: "posting will close at 11:59 pm MST on March 16, 2026"
        m3 = _INLINE_CLOSE_ON_RE.search(line)
        if m3:
            captured = m3.group(1).strip()
            # Date may be in the next text node (span → separate line)
            if not captured and i + 1 < len(lines):
                captured = lines[i + 1]
            if captured:
                d = _parse_date(captured, mm_dd)
                if d is not None:
                    return d

        # Inline: "applications will be accepted until March 13, 2026"
        # Date may be fragmented across several lines (each word in its own <strong>).
        m4 = _INLINE_ACCEPTED_UNTIL_RE.search(line)
        if m4:
            parts: list[str] = [m4.group(1).strip()]
            for j in range(1, 5):
                if i + j < len(lines):
                    parts.append(lines[i + j])
                candidate = " ".join(p for p in parts if p)
                d = _parse_date(candidate, mm_dd)
                if d is not None:
                    return d

        # "by" alone OR at start of line ("by March 27, 2026...") —
        # get_text(separator="\n") can produce either form depending on whether
        # the date is in a separate inline element (<span class="hl-date">) or
        # plain text.  Only trigger when recent preceding lines contain
        # submission context.
        m_by = re.match(r'^\s*by\s*(.*)', line, re.IGNORECASE)
        if m_by:
            prev_ctx = " ".join(lines[max(0, i - 3):i])
            if re.search(
                r'\bsubmit|send\b.{0,20}\bresume|apply\b|application\b|candidature\b',
                prev_ctx,
                re.IGNORECASE,
            ):
                rest = m_by.group(1).strip()
                d = (_parse_date(rest, mm_dd) or _parse_partial_date(rest)) if rest else None
                if d is None and i + 1 < len(lines):
                    d = _parse_date(lines[i + 1], mm_dd) or _parse_partial_date(lines[i + 1])
                if d is not None:
                    return d

        # French "du [date1] au [date2]" range — the closing date follows "au".
        # Only trigger when a parseable date appears in the 3 preceding lines
        # (confirming this is the second half of a date range, not random prose).
        m_au = re.match(r'^\s*au\s*(.*)', line, re.IGNORECASE)
        if m_au:
            prev_lines = lines[max(0, i - 3):i]
            if any(_parse_date(pl, mm_dd) is not None for pl in prev_lines):
                rest = m_au.group(1).strip()
                d = (_parse_date(rest, mm_dd) or _parse_partial_date(rest)) if rest else None
                if d is None and i + 1 < len(lines):
                    d = _parse_date(lines[i + 1], mm_dd) or _parse_partial_date(lines[i + 1])
                if d is not None:
                    return d

    return None


# Label patterns that identify a start/availability date or a weak consideration
# date (not a hard deadline) — used to classify hl-date spans as low-priority.
_START_DATE_CONTEXT_RE = re.compile(
    r"start\s+date|start(?:ing)?\s*(?:date)?|available\s+(?:from|date)?"
    r"|date\s+de\s+d[eé]but|disponible\s+(?:le|à\s+partir)"
    r"|full\s+consideration(?:\s+date)?"
    r"|begin(?:ning)?\s+on|end(?:ing)?\s+on|start(?:s)?\s+on"
    r"|term\s+(?:start|end)"
    r"|entr[eé]e?\s+en\s+fonction"
    r"|\bdu\s*$",  # French "du [date] au [date]" range — "du" precedes the start date
    re.IGNORECASE,
)

# Label patterns for a job start date or a weak consideration date in plain text.
_START_DATE_LABEL_RE = re.compile(
    r"""
    (?:start\s+date|starting\s+date?|date\s+de\s+d[eé]but
      |full\s+consideration(?:\s+date)?
      |begin(?:ning)?\s+on|end(?:ing)?\s+on|start(?:s)?\s+on
      |term\s+(?:start|end)
      |date\s+d[''\u2019]entr[eé]e\s+en\s+fonction)
    \s*:?\s*
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _hl_date_is_start_date(tag: Tag) -> bool:
    """Return True if an hl-date span is labelled as a job start date.

    Checks preceding siblings within the same parent first.  When the span is
    the sole content of its parent (e.g. <p><span class="hl-date">…</span></p>)
    and has no preceding siblings, falls back to the label in the immediately
    preceding sibling of the parent element (typically an <h3> heading).
    """
    prev = tag.previous_sibling
    while prev is not None:
        if isinstance(prev, NavigableString):
            txt = str(prev).strip()
            if txt:
                return bool(_START_DATE_CONTEXT_RE.search(txt))
            prev = prev.previous_sibling
        elif isinstance(prev, Tag):
            return bool(_START_DATE_CONTEXT_RE.search(prev.get_text()))
        else:
            break
    # Span has no preceding text context within its parent — look at the
    # immediately preceding sibling of the parent (e.g. <h3>Date d'entrée en
    # fonction</h3> before <p><span class="hl-date">…</span></p>).
    parent = tag.parent
    if parent is not None:
        prev_sib = parent.find_previous_sibling(True)
        if prev_sib is not None:
            return bool(_START_DATE_CONTEXT_RE.search(prev_sib.get_text()))
    return False


def _extract_hl_dates(soup: BeautifulSoup) -> tuple[date | None, date | None]:
    """Return (deadline, start_date) from all <span class="hl-date"> tags.

    Inspects the text immediately preceding each span to classify it.
    """
    deadline: date | None = None
    start_date: date | None = None
    for tag in soup.find_all("span", class_="hl-date"):
        d = _parse_date(tag.get_text(strip=True))
        if d is None:
            continue
        if _hl_date_is_start_date(tag):
            if start_date is None:
                start_date = d
        else:
            if deadline is None:
                deadline = d
    return deadline, start_date


def _extract_start_date_from_text(full_text: str) -> date | None:
    """Scan plain text for a job start date label; low-priority expiry fallback."""
    mm_dd = _detect_mm_dd(full_text)
    lines = [l.strip() for l in full_text.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        m = _START_DATE_LABEL_RE.search(line)
        if m:
            after = line[m.end():].strip()
            if after:
                d = _parse_date(after, mm_dd)
                if d is not None:
                    return d
            if i + 1 < len(lines):
                d = _parse_date(lines[i + 1], mm_dd)
                if d is not None:
                    return d
    return None


def extract_expiry(raw_html: str) -> date | Literal["expired"] | None:
    """Extract the application deadline from raw job-posting HTML.

    Parameters
    ----------
    raw_html:
        Raw HTML as fetched from a job board.

    Returns
    -------
    datetime.date
        Future deadline date.
    "expired"
        The deadline has already passed (relative to today).
    None
        No deadline found, or posting is explicitly open-ended.
    """
    soup = BeautifulSoup(raw_html, "lxml")
    text = soup.get_text(separator="\n")
    hl_deadline, hl_start = _extract_hl_dates(soup)
    found = (
        hl_deadline
        or _extract_expiry_from_text(text)
        or _extract_start_date_from_text(text)
        or hl_start
    )
    if found is None:
        return None
    return found if found >= date.today() else "expired"


def extract_expiry_raw(raw_html: str) -> date | None:
    """Like extract_expiry but returns the date even if it has already passed.

    Used for labelling/training data extraction where past deadlines are still
    informative for the reviewer.
    """
    soup = BeautifulSoup(raw_html, "lxml")
    text = soup.get_text(separator="\n")
    hl_deadline, hl_start = _extract_hl_dates(soup)
    return (
        hl_deadline
        or _extract_expiry_from_text(text)
        or _extract_start_date_from_text(text)
        or hl_start
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class CleanResult:
    """Result of clean_job_description()."""

    html: str
    """Clean HTML fragment (p, h2–h4, ul/ol/li, strong, em, a only)."""

    expiry: date | Literal["expired"] | None
    """
    Application deadline:
      datetime.date  — future deadline
      "expired"      — deadline has passed
      None           — not found or open-ended
    """


# ---------------------------------------------------------------------------
# HTML transformation helpers (private)
# ---------------------------------------------------------------------------

def _is_header_fragment(text: str) -> bool:
    text = text.strip()
    if not text or len(text) > 60:
        return False
    if len(text.split()) > 8:
        return False
    if text.endswith((".", "?", "!")):
        return False
    return True


def _is_section_header(text: str) -> bool:
    text = text.strip().rstrip(":")
    if not text or len(text) < 2 or len(text) > 80:
        return False
    if len(text.split()) > 7:
        return False
    if text.endswith((".", "?", "!")):
        return False
    if "@" in text:  # email addresses are not section headers
        return False
    if re.search(r"#\d", text):  # reference codes like #11-26 are not section headers
        return False
    return True


_BLOCK_LAYOUT_TAGS = {"div", "section", "article", "aside", "header", "footer", "main", "nav", "tr"}


def _mark_block_layout_boundaries(body: Tag, soup: BeautifulSoup) -> None:
    """Append a <br> to block-level layout elements that are followed by a sibling
    of the same kind (div/tr/section rows).

    Called before _unwrap_layout_tags so that the visual row/section boundaries
    survive as line-break separators after unwrapping. Targets only elements with
    a following block-layout sibling so lone wrapper divs don't gain spurious breaks.
    """
    for tag in body.find_all(_BLOCK_LAYOUT_TAGS):
        nxt = tag.next_sibling
        while isinstance(nxt, NavigableString) and not nxt.strip():
            nxt = nxt.next_sibling
        if isinstance(nxt, Tag) and nxt.name in _BLOCK_LAYOUT_TAGS:
            tag.append(soup.new_tag("br"))


def _unwrap_layout_tags(body: Tag) -> None:
    for tag in body.find_all(_LAYOUT_TAGS):
        tag.unwrap()


def _strip_all_attributes(body: Tag) -> None:
    for tag in body.find_all(True):
        tag.attrs = {}


def _split_bold_on_br(body: Tag, soup: BeautifulSoup) -> None:
    """Split <strong>/<b> containing <br> into multiple sibling bold elements.

    <strong>A<br>B</strong>  →  <strong>A</strong><br><strong>B</strong>

    This lets _collapse_brs treat the <br> like any other line break, producing
    separate <p> blocks rather than concatenating or silently dropping content.
    Trailing <br> (empty last segment) are not emitted, so <strong>Header<br></strong>
    becomes just <strong>Header</strong> and the header-promotion logic is unaffected.
    """
    for tag in list(body.find_all(["strong", "b", "em"])):
        if not tag.find("br", recursive=False):
            continue
        tag_name = tag.name
        segments: list[list] = []
        current: list = []
        for child in list(tag.children):
            if isinstance(child, Tag) and child.name == "br":
                segments.append(current)
                current = []
            else:
                current.append(child)
        segments.append(current)

        new_nodes: list = []
        for i, seg in enumerate(segments):
            inner = "".join(str(c) for c in seg).strip()
            if inner:
                new_tag = BeautifulSoup(f"<{tag_name}>{inner}</{tag_name}>", "lxml").find(tag_name)
                new_nodes.append(new_tag)
            if i < len(segments) - 1 and inner:
                new_nodes.append(soup.new_tag("br"))

        if not new_nodes:
            # Tag had only whitespace + <br> elements (e.g. <strong> <br><br></strong>).
            # Preserve the <br>s so _collapse_brs can convert them to paragraph breaks;
            # decompose only if there were no <br>s at all.
            br_count = len(segments) - 1  # one <br> between each segment
            if br_count > 0:
                brs = [soup.new_tag("br") for _ in range(br_count)]
                tag.replace_with(brs[0])
                for j, br_tag in enumerate(brs[1:], 1):
                    brs[j - 1].insert_after(br_tag)
            else:
                tag.decompose()
            continue
        tag.replace_with(new_nodes[0])
        for i, node in enumerate(new_nodes[1:], 1):
            new_nodes[i - 1].insert_after(node)


def _unwrap_nested_bold(body: Tag) -> None:
    for outer in body.find_all(["strong", "b"]):
        for inner in outer.find_all(["strong", "b"]):
            inner.unwrap()


# Single-word fragments that cannot stand alone as a section header and should
# always be merged with the adjacent bold fragment.
_INCOMPLETE_HEADER_WORDS = frozenset({
    "why", "how", "what", "who", "when", "where", "which",
    "our", "your", "the", "a", "an",
})


def _merge_consecutive_bold(body: Tag) -> None:
    for tag in body.find_all(["strong", "b"]):
        if tag.parent is None:
            continue
        while True:
            nxt = tag.next_sibling
            while isinstance(nxt, NavigableString) and not nxt.strip():
                nxt = nxt.next_sibling
            if not (isinstance(nxt, Tag) and nxt.name in ("strong", "b")):
                break
            t1 = tag.get_text(strip=True)
            t2 = nxt.get_text(strip=True)
            # Allow merge when t2 ends with sentence-final punctuation: the
            # combined text will end with "." / "?" / "!" and will therefore not
            # be promoted to <h3>, which is exactly what we want (e.g.
            # "<strong>Applications close at … 16th </strong><strong>April 2026.</strong>").
            t2_completes_sentence = t2.endswith((".", "?", "!"))
            if not (_is_header_fragment(t1) and (_is_header_fragment(t2) or t2_completes_sentence)):
                break
            if t1.lower() in t2.lower() or t2.lower() in t1.lower():
                break
            tag.string = t1 + " " + t2
            nxt.decompose()


def _promote_standalone_bold(soup: BeautifulSoup, body: Tag) -> None:
    for tag in body.find_all(["b", "strong"]):
        text = tag.get_text(strip=True)
        if _TRACKING_RE.match(text):
            tag.decompose()
            continue
        parent = tag.parent
        if parent is None:
            continue
        # Don't promote when parent is an inline element (e.g. <a>) or a list
        # item — bold text that fills an entire <li> is list content, not a
        # section header, and replacing the <li> with <h3> destroys the list.
        if parent.name in ("a", "em", "strong", "b", "span", "li"):
            continue
        if parent.get_text(strip=True) == text and _is_section_header(text):
            h3 = soup.new_tag("h3")
            h3.string = text.rstrip(":").strip()
            parent.replace_with(h3)


def _split_label_bold_rows(body: Tag, soup: BeautifulSoup) -> None:
    """Insert <br> before consecutive <b>Label:</b> metadata rows that lack separators.

    Fixes ATS-generated blocks where metadata fields are output as bare inline
    elements with no block-level separator between them, e.g.:
        <b>Work Arrangement:</b> Hybrid <b>Requisition Number:</b> 123 …
    lxml auto-wraps these into one <p>, so no <br> → all fields on one line.

    Only triggers when 2+ label-bolds appear in the same container so that a
    single inline "<b>Note:</b> see below" is left alone.
    """

    def _is_label_bold(node: Any) -> bool:
        if not (isinstance(node, Tag) and node.name in ("b", "strong")):
            return False
        text = node.get_text(strip=True)
        # Require actual content before ":" so a bare <strong>:</strong> is excluded.
        return text.endswith(":") and len(text.rstrip(":").strip()) > 0 and len(text.split()) <= 5

    def _has_preceding_content(tag: Tag) -> bool:
        """Return True if tag is preceded by non-whitespace content that is NOT a <br>."""
        prev = tag.previous_sibling
        while isinstance(prev, NavigableString) and not prev.strip():
            prev = prev.previous_sibling
        if prev is None:
            return False
        if isinstance(prev, Tag) and prev.name == "br":
            return False  # already separated
        return True

    def _process_container(container: Tag) -> None:
        label_bolds = [c for c in list(container.children) if _is_label_bold(c)]
        if len(label_bolds) < 2:
            return
        for lb in label_bolds:
            if _has_preceding_content(lb):
                lb.insert_before(soup.new_tag("br"))
        # Also add <br> before the first non-label <b>/<strong> that immediately
        # follows the last label-bold's value (e.g. "Date: Mar 6 <b>Be You.</b>"),
        # but only if it looks like a section header (not a range dash like <b>-</b>).
        node = label_bolds[-1].next_sibling
        while node is not None:
            if isinstance(node, Tag) and node.name == "br":
                break  # already separated
            if isinstance(node, Tag) and node.name in ("b", "strong") and not _is_label_bold(node):
                if _is_section_header(node.get_text(strip=True)):
                    node.insert_before(soup.new_tag("br"))
                break
            node = node.next_sibling

    _process_container(body)
    for p in list(body.find_all("p")):
        _process_container(p)


def _split_trailing_section_from_label_para(body: Tag, soup: BeautifulSoup) -> None:
    """Split <p> elements that open with a label-bold but contain a non-label bold later.

    After _collapse_brs, catches cases like:
        <p><b>Date:</b>Mar 6, 2026<b>Be You.</b>At Duke…</p>
    and splits at the non-label bold boundary.
    """

    def _is_label_bold(node: Any) -> bool:
        if not (isinstance(node, Tag) and node.name in ("b", "strong")):
            return False
        text = node.get_text(strip=True)
        return text.endswith(":") and len(text.rstrip(":").strip()) > 0 and len(text.split()) <= 5

    for p in list(body.find_all("p")):
        children = list(p.children)
        # Paragraph must start with a label-bold
        first_real = next(
            (c for c in children if not (isinstance(c, NavigableString) and not c.strip())),
            None,
        )
        if not (first_real and _is_label_bold(first_real)):
            continue
        # Find first non-label bold anywhere after the opening label-bold
        past_first = False
        for child in children:
            if child is first_real:
                past_first = True
                continue
            if not past_first:
                continue
            if (
                isinstance(child, Tag)
                and child.name in ("b", "strong")
                and not _is_label_bold(child)
                and _is_section_header(child.get_text(strip=True))
            ):
                child.insert_before(soup.new_tag("br"))
                _split_p_on_brs(p, soup)
                break


def _wrap_orphan_lis(body: Tag, soup: BeautifulSoup) -> None:
    """Wrap <li> elements that are direct body children in a <ul>.

    lxml's HTML parser sometimes places bare <li> tags (without a parent
    <ul>/<ol> in the source) as direct body children rather than creating an
    implicit list. _split_on_brs then puts them in its flush bucket, which
    wraps them in <p>; lxml rejects <li> inside <p>, so the content is lost.
    Running this step first ensures they are inside a <ul> before _collapse_brs.
    """
    while True:
        # Find the first direct-child <li>; process one group per iteration.
        first = next(
            (c for c in body.children if isinstance(c, Tag) and c.name == "li"),
            None,
        )
        if first is None:
            break
        ul = soup.new_tag("ul")
        first.insert_before(ul)
        # Absorb all consecutive <li> siblings into the new <ul>
        node = first
        while node and isinstance(node, Tag) and node.name == "li":
            nxt = node.next_sibling
            node.extract()
            ul.append(node)
            # Skip whitespace-only text nodes between <li> elements
            while nxt and isinstance(nxt, NavigableString) and not nxt.strip():
                nxt = nxt.next_sibling
            node = nxt if (isinstance(nxt, Tag) and nxt.name == "li") else None


def _split_on_brs(container: Tag, soup: BeautifulSoup) -> None:
    bucket: list = []
    new_top: list = []

    def _flush() -> None:
        parts: list[str] = []
        for node in bucket:
            if isinstance(node, NavigableString):
                t = str(node).strip()
                if t:
                    parts.append(t)
            elif isinstance(node, Tag):
                parts.append(str(node))
        if parts:
            frag = BeautifulSoup(f'<p>{" ".join(parts)}</p>', "lxml").find("p")
            if frag:
                new_top.append(frag)

    for child in list(container.children):
        if isinstance(child, Tag) and child.name == "br":
            _flush()
            bucket = []
        elif isinstance(child, Tag) and child.name in _BLOCK_TAGS:
            _flush()
            bucket = []
            new_top.append(child)
        elif (
            isinstance(child, Tag)
            and child.name in ("b", "strong")
            and len(child.get_text(strip=True)) > 4
            and _is_section_header(child.get_text(strip=True))
            and any(isinstance(n, NavigableString) and str(n).strip() for n in bucket)
            and not _MID_SENTENCE_ENDS_RE.search(
                next(
                    (str(n) for n in reversed(bucket)
                     if isinstance(n, NavigableString) and str(n).strip()),
                    "",
                )
            )
        ):
            # Standalone heading bold encountered after non-empty text (e.g. a
            # job-reference code "GOL00555" in the same run) — flush the text
            # first so they become separate paragraphs.
            # Guard: skip if the preceding text ends with an article/preposition
            # (e.g. "opening for a <strong>Role Title</strong>") — the bold is
            # inline, not a section header.
            _flush()
            bucket = [child]
        else:
            bucket.append(child)
    _flush()
    container.clear()
    for node in new_top:
        container.append(node)


def _split_p_on_brs(p: Tag, soup: BeautifulSoup) -> None:
    """Replace a <p> that contains <br> with sibling <p> elements.

    Unlike _split_on_brs (designed for the body container), this function
    inserts the new paragraphs as siblings — avoiding nested <p> that
    _fix_nested_p would later unwrap, concatenating the content.
    """
    segments: list[list] = []
    bucket: list = []
    for child in list(p.children):
        if isinstance(child, Tag) and child.name == "br":
            segments.append(bucket)
            bucket = []
        else:
            bucket.append(child)
    segments.append(bucket)

    new_paras: list[Tag] = []
    for seg in segments:
        parts = [
            str(c) if isinstance(c, Tag) else str(c).strip()
            for c in seg
            if not (isinstance(c, NavigableString) and not str(c).strip())
        ]
        inner = " ".join(p for p in parts if p).strip()
        if inner:
            frag = BeautifulSoup(f"<p>{inner}</p>", "lxml").find("p")
            if frag:
                new_paras.append(frag)

    if not new_paras:
        p.decompose()
    elif len(new_paras) == 1:
        p.replace_with(new_paras[0])
    else:
        p.replace_with(new_paras[0])
        for i, np in enumerate(new_paras[1:], 1):
            new_paras[i - 1].insert_after(np)


def _collapse_brs(body: Tag, soup: BeautifulSoup) -> None:
    _split_on_brs(body, soup)
    for p in list(body.find_all("p")):
        if p.find("br"):
            _split_p_on_brs(p, soup)


def _split_p_on_blank_lines(body: Tag, soup: BeautifulSoup) -> None:
    """Split <p> elements that contain blank lines (\\n\\n) into sibling <p> elements.

    Handles sources that use literal newlines instead of <br> tags to separate
    paragraphs — after unwrapping <div>/<table> layout tags these end up as a
    single large <p> with internal \\n\\n sequences.
    """
    for p in list(body.find_all("p")):
        serialized = "".join(str(c) for c in p.children)
        segments = re.split(r"\n[ \t]*\n", serialized)
        if len(segments) <= 1:
            continue
        new_paras: list[Tag] = []
        for seg in segments:
            inner = seg.strip()
            if inner:
                frag = BeautifulSoup(f"<p>{inner}</p>", "lxml").find("p")
                if frag:
                    new_paras.append(frag)
        if not new_paras:
            p.decompose()
        elif len(new_paras) == 1:
            p.replace_with(new_paras[0])
        else:
            p.replace_with(new_paras[0])
            for i, np_ in enumerate(new_paras[1:], 1):
                new_paras[i - 1].insert_after(np_)


# Unicode characters used as bullet separators in Word/ATS-exported HTML.
# \uf0b7 is the Wingdings private-use-area bullet very common in ATS exports.
_ANY_BULLET_RE = re.compile(r"[•◦◆▪▸▶✓✔●○■□▫▹\uf0b7]")
_BULLET_SPLIT_RE = re.compile(r"\s*[•◦◆▪▸▶✓✔●○■□▫▹\uf0b7]\s*")


def _convert_bullet_chars_to_list(body: Tag, soup: BeautifulSoup) -> None:
    """Convert <p> elements that use Unicode bullet chars as item separators into <ul><li>.

    Handles job descriptions exported from Word/ATS systems where items are
    separated by characters like • ◦ ▪ instead of proper <ul><li> markup.
    Requires at least 2 bullet characters in the paragraph to trigger conversion.
    If the first segment is ALL-CAPS (≥2 words) it is promoted to <h3>.
    """
    for p in list(body.find_all("p")):
        inner_html = "".join(str(c) for c in p.children)
        if len(_ANY_BULLET_RE.findall(inner_html)) < 2:
            continue

        parts = [pt.strip() for pt in _BULLET_SPLIT_RE.split(inner_html) if pt.strip()]
        if len(parts) < 2:
            continue

        replacements: list[Tag] = []

        # Hoist an ALL-CAPS first segment as a heading
        first_text = BeautifulSoup(f"<span>{parts[0]}</span>", "lxml").get_text(strip=True)
        stripped = first_text.strip()
        is_caps_header = (
            len(stripped.split()) >= 2
            and stripped == stripped.upper()
            and stripped.replace(" ", "").isalpha()
        )
        list_parts = parts[1:] if is_caps_header else parts
        if is_caps_header:
            h3 = soup.new_tag("h3")
            h3.string = stripped
            replacements.append(h3)

        if list_parts:
            ul = soup.new_tag("ul")
            for part in list_parts:
                li_soup = BeautifulSoup(f"<li>{part}</li>", "lxml").find("li")
                if li_soup and li_soup.get_text(strip=True):
                    ul.append(li_soup)
            if ul.find("li"):
                replacements.append(ul)

        if replacements:
            p.replace_with(replacements[0])
            for i, repl in enumerate(replacements[1:], 1):
                replacements[i - 1].insert_after(repl)


def _normalize_inline_whitespace(body: Tag) -> None:
    """Collapse whitespace within text nodes inside <p> and <li> elements.

    Removes leading/trailing whitespace and collapses internal \\n and
    multiple spaces to a single space in each text node.
    """
    for tag in body.find_all(["p", "li"]):
        for child in list(tag.children):
            if isinstance(child, NavigableString):
                normalized = re.sub(r"\s+", " ", str(child)).strip()
                if normalized != str(child):
                    child.replace_with(NavigableString(normalized))


def _promote_leading_bold_in_p(soup: BeautifulSoup, body: Tag) -> None:
    for p in list(body.find_all("p")):
        children = [
            c for c in p.children
            if not (isinstance(c, NavigableString) and not c.strip())
        ]
        if not children:
            continue
        first = children[0]
        if not (isinstance(first, Tag) and first.name in ("strong", "b")):
            continue
        header_text = first.get_text(strip=True).rstrip(":").strip()
        if not _is_section_header(header_text):
            continue
        remainder = "".join(
            str(c) for c in p.children if c is not first
        ).strip().lstrip(":").strip()
        is_inline_label = bool(remainder) and (
            (len(header_text.split()) <= 3 and ":" in first.get_text())
            or re.search(r"[\-\u2013\u2014]\s*$", first.get_text())
        )
        if is_inline_label:
            continue
        # If the remainder starts with a lowercase letter the bold is an inline
        # subject continuing into a sentence (e.g. "<strong>Unimax</strong> brings
        # together…"), not a standalone section header.
        remainder_plain = BeautifulSoup(remainder, "lxml").get_text() if remainder else ""
        first_char = remainder_plain.lstrip()[0:1]
        if first_char and (first_char.islower() or first_char in ".,;:)("):
            continue
        h3 = soup.new_tag("h3")
        h3.string = header_text
        if remainder:
            new_p = BeautifulSoup(f"<p>{remainder}</p>", "lxml").find("p")
            p.replace_with(h3)
            h3.insert_after(new_p)
        else:
            p.replace_with(h3)


def _fix_h3_in_p(body: Tag) -> None:
    for p in list(body.find_all("p")):
        h3 = p.find("h3")
        if not h3:
            continue
        before: list = []
        after: list = []
        found = False
        for child in list(p.children):
            if child is h3:
                found = True
            elif not found:
                before.append(child)
            else:
                after.append(child)
        h3.extract()

        def _nodes_to_p(nodes: list):
            text = "".join(
                str(n)
                for n in nodes
                if not (isinstance(n, NavigableString) and not n.strip())
            ).strip()
            if not text:
                return None
            return BeautifulSoup(f"<p>{text}</p>", "lxml").find("p")

        parts = [x for x in [_nodes_to_p(before), h3, _nodes_to_p(after)] if x]
        p.replace_with(parts[0])
        for i, part in enumerate(parts[1:], 1):
            parts[i - 1].insert_after(part)


def _fix_nested_p(body: Tag) -> None:
    for p in body.find_all("p"):
        for inner in p.find_all("p"):
            inner.unwrap()


def _wrap_naked_text(soup: BeautifulSoup, body: Tag) -> None:
    for child in list(body.children):
        if isinstance(child, NavigableString) and child.strip():
            p = soup.new_tag("p")
            child.replace_with(p)
            p.string = child.strip()


def _hoist_h3_from_li(soup: BeautifulSoup, body: Tag) -> None:
    """Restructure lists where <h3> elements are used as section dividers inside <li>.

    Handles the Canadian Job Bank pattern:
      <ul>
        <li><h3>Tasks</h3></li>
        <li>Do thing A</li>
        <li>Do thing B</li>
        <li><h3>Requirements</h3></li>
        <li>Requirement 1</li>
      </ul>

    Transforms into:
      <h3>Tasks</h3>
      <ul><li>Do thing A</li><li>Do thing B</li></ul>
      <h3>Requirements</h3>
      <ul><li>Requirement 1</li></ul>

    Label-only <li> items (text ending with ":" and at most 3 words) are dropped.
    """
    for lst in list(body.find_all(["ul", "ol"])):
        children = list(lst.find_all("li", recursive=False))
        # Only process lists that contain at least one <li><h3>...</h3></li>
        if not any(
            li.find("h3", recursive=False)
            or (li.find("h3") and li.get_text(strip=True) == (li.find("h3") or li).get_text(strip=True))
            for li in children
        ):
            continue

        groups: list[tuple] = []  # (h3_tag | None, [li_tags])
        current_h3 = None
        current_items: list = []

        for li in children:
            h3 = li.find("h3")
            if h3 and li.get_text(strip=True) == h3.get_text(strip=True):
                # li is purely a section header
                groups.append((current_h3, current_items))
                current_h3 = soup.new_tag("h3")
                current_h3.string = h3.get_text(strip=True)
                current_items = []
            else:
                text = li.get_text(strip=True)
                # Drop label-only items like "Education:" or "Expérience:"
                if text.endswith(":") and len(text.split()) <= 4:
                    continue
                current_items.append(li)

        groups.append((current_h3, current_items))

        new_nodes: list = []
        for h3_tag, items in groups:
            if h3_tag is not None:
                new_nodes.append(h3_tag)
            if items:
                new_ul = soup.new_tag(lst.name)
                for item in items:
                    item.extract()
                    new_ul.append(item)
                new_nodes.append(new_ul)

        if not new_nodes:
            lst.decompose()
            continue

        lst.replace_with(new_nodes[0])
        for i, node in enumerate(new_nodes[1:], 1):
            new_nodes[i - 1].insert_after(node)


def _dedup_consecutive_h3(body: Tag) -> None:
    """Remove an <h3> that is immediately followed by an <h3> with identical text.

    Job boards (Workday, Lever, etc.) sometimes emit duplicate consecutive
    headings (e.g. two "Description du poste" blocks). Only same-text pairs
    are removed; consecutive headings with different text (e.g. "Step 1",
    "Step 2", …) are kept intact.
    Empty/whitespace-only sibling elements (e.g. empty <p> left over from
    <br><br> inside the first <strong>) are skipped when looking for the
    next meaningful sibling.
    """
    for h3 in list(body.find_all("h3")):
        # Walk forward past empty siblings (empty tags or whitespace-only text).
        sib = h3.find_next_sibling(True)
        while sib and not sib.get_text(strip=True):
            sib = sib.find_next_sibling(True)
        if (
            sib
            and sib.name == "h3"
            and h3.get_text(strip=True).lower() == sib.get_text(strip=True).lower()
        ):
            h3.decompose()


def _drop_label_before_heading(body: Tag) -> None:
    """Remove a plain-text <p> ending in ':' when immediately followed by <h3>.

    Handles the pattern:
      <p>Main Responsibilities:</p>
      <h3>Responsibilities</h3>
      <ul>...</ul>
    The paragraph is a redundant category label; the heading is the real title.
    Only drops paragraphs with no child elements (pure text or a single inline).
    """
    for p in list(body.find_all("p")):
        text = p.get_text(strip=True)
        if not text.endswith(":"):
            continue
        # Only plain-text or single-strong paragraphs (not rich content)
        if len(list(p.children)) > 2:
            continue
        next_el = p.find_next_sibling()
        if next_el and next_el.name == "h3":
            p.decompose()


def _remove_nav_lists(body: Tag) -> None:
    """Drop <ul>/<ol> lists that are entirely composed of UI navigation items."""
    for lst in list(body.find_all(["ul", "ol"])):
        items = lst.find_all("li", recursive=False)
        if not items:
            continue
        if all(_UI_ARTIFACT_RE.match(li.get_text(strip=True)) for li in items):
            lst.decompose()


def _remove_ui_artifacts(body: Tag) -> None:
    for tag in list(body.find_all(["p", "h2", "h3", "h4"])):
        text = tag.get_text(strip=True)
        if not _UI_ARTIFACT_RE.match(text):
            continue
        # Keep generic section headers (Description / Job Description) when the
        # next sibling is a list — they serve as a real heading in that context.
        next_el = tag.find_next_sibling()
        if re.fullmatch(r"(?:job\s+)?description", text, re.IGNORECASE) and (
            next_el is not None and next_el.name in ("ul", "ol")
        ):
            continue
        tag.decompose()


def _drop_empty_blocks(body: Tag) -> None:
    for tag in body.find_all(["p", "li", "h2", "h3", "h4"]):
        text = tag.get_text(strip=True)
        if not text:
            tag.decompose()
        elif re.fullmatch(r"[\w\s/\-]+:\s*", text):
            # Label with no value, e.g. "Position Title: " left after placeholder removal.
            # Keep if the next sibling is a list — it's a genuine intro label ("We offer:").
            next_el = tag.find_next_sibling()
            if next_el and next_el.name in ("ul", "ol"):
                pass
            else:
                tag.decompose()
        elif tag.name == "p" and re.fullmatch(r"\d{1,4}", text):
            # Lone number, e.g. "09" orphaned after a Grade Level heading is removed
            tag.decompose()


def _enforce_allowed_tags(body: Tag) -> None:
    for tag in body.find_all(True):
        if tag.name not in _ALLOWED_TAGS:
            tag.unwrap()


_MD_BOLD_RE = re.compile(r"\*\*([^\*\n]+?)\*\*")

# Windows-1252 "smart" punctuation stored as C1 control codepoints (U+0080–U+009F).
# Occurs when a Windows-1252 page is mislabelled ISO-8859-1: bytes like \x92 (')
# get decoded as U+0092 and re-encoded in UTF-8 as \xC2\x92, displayed as "Â" +
# an invisible character (e.g. "company's" → "companyÂs").
_C1_FIX_RE = re.compile("\u00c2([\u0080-\u009f])")
_C1_MAP: dict[str, str] = {
    "\u0091": "\u2018",  # '  left single quote
    "\u0092": "\u2019",  # '  right single quote / apostrophe
    "\u0093": "\u201c",  # "  left double quote
    "\u0094": "\u201d",  # "  right double quote
    "\u0095": "\u2022",  # •  bullet
    "\u0096": "\u2013",  # –  en dash
    "\u0097": "\u2014",  # —  em dash
    "\u0099": "\u2122",  # ™  trademark
}

_UI_ARTIFACT_RE = re.compile(
    r"^(apply(?:\s+(?:now|for\s+this\s+(?:job|position|role)))?|back\s+to\s+search\s+results?"
    r"|postulez(?:\s+maintenant)?|accueil|nos\s+offres?|toutes?\s+les\s+offres?"
    r"|retour\s+(?:aux|à\s+la\s+liste\s+des)\s+offres?"
    r"|(?:accueil\s+)?postulez\s+nos\s+offres?"
    r"|share\s+(?:link|on\s+\w+|via\s+\w+)"
    r"|job\s+description|description"
    r"|log\s+in\s+today(?:\s+and)?|log\s*in|sign\s+in|sign\s+up|log\s+out|sign\s+out"
    r"|register\s+now(?:\s+to\s+get\s+started)?|register"
    r"|get\s+real.time\s+job\s+notifications?"
    r"|view\s+pay\s+(?:&|and)\s+facility\s+details?"
    r"|search\s+jobs?|search"
    r"|help|home|menu|skip\s+to\s+(?:main\s+)?content"
    r"|my\s+(?:profile|account|applications?)|saved\s+jobs?"
    r"|contact\s+us|about\s+us|faqs?|privacy\s+policy|terms\s+(?:of\s+(?:use|service))?"
    # Separator lines (e.g. "---...---" or "--- 203 - EEO Job Group...")
    r"|[-=_*\s]{3,}"
    r"|[-=_]{3,}.*"
    # Internal job metadata
    r"|job\s+(?:id|req(?:uisition)?(?:\s+id)?)\s*[:#]?\s*\S+"
    r"|posted\s+on\s*[:\-]?\s*.+"
    r"|grade\s+level\s*(?:\([^)]+\))?\s*:?\s*\S*"
    # EEO job classification codes (e.g. "203 - Entry Professional (EEO Job Group)")
    r"|.*\(eeo(?:[- ]2)?\s+job\s+(?:group|categor\w+)\).*"
    # Job board navigation: "Previous job Next job", "Previous posting Next posting"
    r"|previous\s+(?:job|posting|position)\s+next\s+(?:job|posting|position)"
    # Employee/internal portal prompts
    r"|are\s+you\s+(?:a\s+)?\w[\w\s]*employee\??"
    r"|open\s+(?:my\s+)?\w[\w\s&]*portal"
    r"|current\s+[\w\s]+employees?\s+should\s+apply\b.*"
    # Email-this-job widget
    r"|email\s+this\s+job\s+to(?:\s+a\s+friend)?"
    r"|your\s+email\s+is\s+on\s+its\s+way\.{0,3}"
    r"|email\s+has\s+not\s+(?:been\s+)?sent"
    # Print/save notice
    r"|please\s+print(?:[/\\]save)?\s+this\s+job\s+description.*"
    r")\s*$",
    re.IGNORECASE,
)


def _convert_markdown_bold(src: str) -> str:
    """Convert **text** and *text* to <strong>text</strong>."""
    return _MD_BOLD_RE.sub(lambda m: f"<strong>{m.group(1)}</strong>", src)


def _build_clean_html(raw_html: str) -> str:
    src = urllib.parse.unquote(raw_html)
    # Fix Windows-1252 smart punctuation stored as C1 control codepoints
    # (e.g. "companyÂs" → "company's")
    src = _C1_FIX_RE.sub(lambda m: _C1_MAP.get(m.group(1), ""), src)
    # Strip JSON-style backslash escapes (e.g. \" → ")
    src = re.sub(r'\\(["\'/\*])', r'\1', src)
    src = re.sub(r"!\*!<.*", "", src, flags=re.DOTALL)
    # Remove unfilled template placeholders like [[title]] or {{field_name}}
    src = re.sub(r"\[\[.*?\]\]|\{\{.*?\}\}", "", src)
    # Convert bare URLs (not inside href/src attributes) to <a> links so that
    # document references (privacy notices, benefit guides, etc.) are preserved.
    # Use [^\s<>]+ instead of \S+ so HTML tags immediately following the URL
    # (e.g. https://example.com<br>) are not consumed.
    src = re.sub(
        r'(?<!=")https?://[^\s<>]+',
        lambda m: f'<a href="{m.group(0)}">{m.group(0)}</a>',
        src,
    )
    src = _convert_markdown_bold(src)
    soup = BeautifulSoup(src, "lxml")
    body: Tag = soup.find("body") or soup  # type: ignore[assignment]

    _mark_block_layout_boundaries(body, soup)
    _unwrap_layout_tags(body)
    _strip_all_attributes(body)
    _split_bold_on_br(body, soup)
    _unwrap_nested_bold(body)
    _split_bold_on_br(body, soup)  # second pass: outer strongs that gained <br> after unwrapping
    _merge_consecutive_bold(body)
    _promote_standalone_bold(soup, body)
    _wrap_orphan_lis(body, soup)
    _split_label_bold_rows(body, soup)
    _collapse_brs(body, soup)
    _split_trailing_section_from_label_para(body, soup)
    _split_p_on_blank_lines(body, soup)
    _promote_leading_bold_in_p(soup, body)
    _fix_h3_in_p(body)
    _fix_nested_p(body)
    _wrap_naked_text(soup, body)
    _hoist_h3_from_li(soup, body)
    _dedup_consecutive_h3(body)
    _normalize_inline_whitespace(body)
    _convert_bullet_chars_to_list(body, soup)
    _remove_nav_lists(body)
    _remove_ui_artifacts(body)
    _drop_label_before_heading(body)
    _drop_empty_blocks(body)
    for b_tag in list(body.find_all("b")):
        b_tag.name = "strong"
    _enforce_allowed_tags(body)

    result = str(body)
    result = re.sub(r"<body[^>]*>|</body>", "", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r"[ \t]+", " ", result)
    # Ensure a space where a non-whitespace character is immediately adjacent to
    # an inline open/close tag boundary (e.g. "right<strong>", "end.<strong>")
    # [^\s>] excludes whitespace and '>' so we never insert inside tag markup.
    result = re.sub(r"([^\s>])(<(?:strong|em|a)(?:\s[^>]*)?>)", r"\1 \2", result)
    result = re.sub(r"(</(?:strong|em|a)>)(\w)", r"\1 \2", result)
    # Ensure a space between two adjacent closing/opening inline tags
    # (e.g. "</strong><strong>" → "</strong> <strong>")
    result = re.sub(
        r"(</(?:strong|em|a)>)(<(?:strong|em|a)(?:\s[^>]*)?>)",
        r"\1 \2",
        result,
    )
    return result.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_job_description(raw_html: str) -> CleanResult:
    """Clean raw job-posting HTML and extract the application deadline.

    Parameters
    ----------
    raw_html:
        Raw HTML string as fetched from a job board.

    Returns
    -------
    CleanResult
        .html    — clean HTML fragment
        .expiry  — datetime.date  (future deadline)
                   "expired"      (deadline has passed)
                   None           (not found or open-ended)
    """
    return CleanResult(
        html=_build_clean_html(raw_html),
        expiry=extract_expiry(raw_html),
    )
