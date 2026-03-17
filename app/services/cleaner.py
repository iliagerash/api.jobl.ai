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
10. Promote <strong>/<b> at the START of a <p> to <h3> + <p>
    — skipped for short inline label:value pairs  ("Posting ID: 5064")
11. Fix invalid <h3> nested inside <p>  (hoist h3 out, split content)
12. Unwrap double-nested <p><p>  (lxml re-parse artefact)
13. Wrap bare text nodes in <p>
14. Drop empty block tags
15. Enforce allowed tag set: p h2 h3 h4 ul ol li strong em a

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
from typing import Literal

from bs4 import BeautifulSoup, NavigableString, Tag

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRACKING_RE = re.compile(r"^#[A-Z][A-Z0-9\-]+$")

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

# Patterns that introduce a deadline field label (EN + FR, expanded)
_DEADLINE_LABEL_RE = re.compile(
    r"""
    (?:
        application\s+deadline
      | unposting\s+date
      | closing\s+date
      | close\s+date
      | deadline
      | position\s+closes
      | apply\s+before
      | posting\s+closes
      | applications?\s+close
      | applications?\s+due
      | expiry\s+date
      | open\s+until
      | date\s+limite\s+(?:pour\s+)?(?:postuler|de\s+candidature)?
      | date\s+de\s+(?:cl[oô]ture|fermeture)
      | avant\s+le
      | candidatures?\s+re[cç]ues?\s+jusqu(?:[''\u2019]|\s+)au
      | fermeture\s+du\s+concours
    )
    \s*:?\s*
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Inline prose: "apply by April 1, 2026"
_INLINE_APPLY_BY_RE = re.compile(
    r"apply\s+by\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)

# Open-ended values — no expiry implied
_OPEN_ENDED_RE = re.compile(r"^\s*(ongoing|until\s+filled|open)\s*$", re.IGNORECASE)

# EN date: "March 19, 2026" / "April 1, 2026"
_EN_DATE_RE = re.compile(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})")
# FR date: "2 avril 2026"
_FR_DATE_RE = re.compile(r"(\d{1,2})\s+([a-zéûôàî]+)\s+(\d{4})", re.IGNORECASE)


def _parse_date(text: str) -> date | None:
    """Parse a date string. Returns None if unparseable or open-ended."""
    text = text.strip()
    if not text or _OPEN_ENDED_RE.match(text):
        return None

    # Numeric YYYY-MM-DD
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # Numeric DD/MM/YYYY or DD-MM-YYYY
    m = re.match(r"^(\d{1,2})[/\-](\d{2})[/\-](\d{4})$", text)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass

    # French first (prevents EN parser from misreading "2 avril 2026")
    m = _FR_DATE_RE.search(text)
    if m:
        day = int(m.group(1))
        month = _FR_MONTHS.get(m.group(2).lower())
        year = int(m.group(3))
        if month:
            try:
                return date(year, month, day)
            except ValueError:
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


def _extract_expiry_from_text(full_text: str) -> date | None:
    """Scan plain text for deadline patterns, return first parseable date.

    Returns None for open-ended postings ("Ongoing") and for no match.
    """
    lines = [l.strip() for l in full_text.splitlines() if l.strip()]

    for i, line in enumerate(lines):
        # Label-based: value may follow on the same line or the next
        m = _DEADLINE_LABEL_RE.search(line)
        if m:
            after = line[m.end():].strip()
            if after:
                if _OPEN_ENDED_RE.match(after):
                    return None
                d = _parse_date(after)
                if d is not None:
                    return d
            if i + 1 < len(lines):
                nxt = lines[i + 1]
                if _OPEN_ENDED_RE.match(nxt):
                    return None
                d = _parse_date(nxt)
                if d is not None:
                    return d

        # Inline prose: "apply by April 1, 2026"
        m2 = _INLINE_APPLY_BY_RE.search(line)
        if m2:
            d = _parse_date(m2.group(1))
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
    found = _extract_expiry_from_text(text)
    if found is None:
        return None
    return found if found >= date.today() else "expired"


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
    if not text or len(text) < 2 or len(text) > 60:
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
    if len(text.split()) > 9:
        return False
    if text.endswith((".", "?", "!")):
        return False
    return True


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
    for tag in list(body.find_all(["strong", "b"])):
        if not tag.find("br"):
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
            tag.decompose()
            continue
        tag.replace_with(new_nodes[0])
        for i, node in enumerate(new_nodes[1:], 1):
            new_nodes[i - 1].insert_after(node)


def _unwrap_nested_bold(body: Tag) -> None:
    for outer in body.find_all(["strong", "b"]):
        for inner in outer.find_all(["strong", "b"]):
            inner.unwrap()


def _merge_consecutive_bold(body: Tag) -> None:
    for tag in body.find_all(["strong", "b"]):
        if tag.parent is None:
            continue
        nxt = tag.next_sibling
        while isinstance(nxt, NavigableString) and not nxt.strip():
            nxt = nxt.next_sibling
        if not (isinstance(nxt, Tag) and nxt.name in ("strong", "b")):
            continue
        t1 = tag.get_text(strip=True)
        t2 = nxt.get_text(strip=True)
        if not (_is_header_fragment(t1) and _is_header_fragment(t2)):
            continue
        if t1.lower() in t2.lower() or t2.lower() in t1.lower():
            continue
        # Both are standalone headers — keep separate so each becomes its own <h3>
        if _is_section_header(t1) and _is_section_header(t2):
            continue
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
        if parent.get_text(strip=True) == text and _is_section_header(text):
            h3 = soup.new_tag("h3")
            h3.string = text.rstrip(":").strip()
            parent.replace_with(h3)


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
        is_inline_label = (
            len(header_text.split()) <= 3
            and ":" in first.get_text()
            and len(remainder) < 80
        )
        if is_inline_label:
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


def _remove_ui_artifacts(body: Tag) -> None:
    for tag in list(body.find_all(["p", "h2", "h3", "h4"])):
        if _UI_ARTIFACT_RE.match(tag.get_text(strip=True)):
            tag.decompose()


def _drop_empty_blocks(body: Tag) -> None:
    for tag in body.find_all(["p", "li", "h2", "h3", "h4"]):
        if not tag.get_text(strip=True):
            tag.decompose()


def _enforce_allowed_tags(body: Tag) -> None:
    for tag in body.find_all(True):
        if tag.name not in _ALLOWED_TAGS:
            tag.unwrap()


_MD_BOLD_RE = re.compile(r"\*\*([^\*\n]+?)\*\*|\*([^\*\n]+?)\*")

_UI_ARTIFACT_RE = re.compile(
    r"^(apply(?:\s+(?:now|for\s+this\s+(?:job|position|role)))?|back\s+to\s+search\s+results?"
    r"|postulez(?:\s+maintenant)?|accueil|nos\s+offres?|toutes?\s+les\s+offres?"
    r"|retour\s+(?:aux|à\s+la\s+liste\s+des)\s+offres?"
    r"|(?:accueil\s+)?postulez\s+nos\s+offres?"
    r"|share\s+(?:link|on\s+\w+|via\s+\w+)"
    r"|job\s+description|description"
    r"|log\s+in\s+today(?:\s+and)?"
    r"|register\s+now(?:\s+to\s+get\s+started)?"
    r"|get\s+real.time\s+job\s+notifications?"
    r"|view\s+pay\s+(?:&|and)\s+facility\s+details?"
    r"|search\s+jobs?"
    r")\s*$",
    re.IGNORECASE,
)


def _convert_markdown_bold(src: str) -> str:
    """Convert **text** and *text* to <strong>text</strong>."""
    return _MD_BOLD_RE.sub(lambda m: f"<strong>{m.group(1) or m.group(2)}</strong>", src)


def _build_clean_html(raw_html: str) -> str:
    src = urllib.parse.unquote(raw_html)
    src = re.sub(r"!?\*!<.*", "", src, flags=re.DOTALL)
    src = _convert_markdown_bold(src)
    soup = BeautifulSoup(src, "lxml")
    body: Tag = soup.find("body") or soup  # type: ignore[assignment]

    _unwrap_layout_tags(body)
    _strip_all_attributes(body)
    _split_bold_on_br(body, soup)
    _unwrap_nested_bold(body)
    _merge_consecutive_bold(body)
    _promote_standalone_bold(soup, body)
    _collapse_brs(body, soup)
    _promote_leading_bold_in_p(soup, body)
    _fix_h3_in_p(body)
    _fix_nested_p(body)
    _wrap_naked_text(soup, body)
    _remove_ui_artifacts(body)
    _drop_empty_blocks(body)
    _enforce_allowed_tags(body)

    result = str(body)
    result = re.sub(r"<body[^>]*>|</body>", "", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r"[ \t]+", " ", result)
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
