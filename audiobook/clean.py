"""Text cleanup: page furniture, hyphenation and speech normalization.

Two outputs are produced from the same raw block:

* ``normalize_for_review``  keeps the author's words, only fixing ligatures,
  quotes, whitespace and line-break hyphenation. This is what you read when
  reviewing ``script.json``.
* ``normalize_for_speech``  additionally verbalizes acronyms, numbers,
  identifiers and URLs so the TTS does not have to guess.
"""

from __future__ import annotations

import re
from typing import Iterable
from urllib.parse import urlsplit

from .extract import Block

SOFT_HYPHENS = "\u2010\u00ad"
HARD_HYPHENS = "-\u2011"

# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------


def page_furniture_kind(
    text: str, y0: float, page_height: float, size: float
) -> str | None:
    """Return the kind of page furniture, if any.

    The sample book has no top running header. Page numbers and the running
    chapter title sit at the bottom (y ≈ 610 pt); the PDFDrive watermark sits
    below them. Footnotes are small type above the footer.
    """
    if "it-ebooks.info" in text.lower():
        return "watermark"
    if y0 >= page_height * 0.90 and size <= 9.5:
        return "footer"
    if size <= 8.6 and y0 >= page_height * 0.70:
        return "footnote"
    return None


# ---------------------------------------------------------------------------
# Line joining
# ---------------------------------------------------------------------------


def join_lines(lines: Iterable[str]) -> str:
    """Join wrapped lines, resolving end-of-line hyphenation.

    In this book, a line ending in U+2010 is a typeset break inside a word
    (``com‐`` + ``puting``); an ASCII hyphen at the end is part of a compound
    (``blame-`` + ``free``). Keeping that distinction avoids both
    ``com-puting`` and ``blamefree``.
    """
    result = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not result:
            result = line
            continue
        previous = result[-1]
        if previous in SOFT_HYPHENS:
            result = result[:-1] + line
        elif previous in HARD_HYPHENS:
            result = result + line
        else:
            result = result + " " + line
    return result


def block_text(block: Block) -> str:
    return join_lines(line.speech_text() for line in block.lines)


def join_paragraphs(previous: str, following: str) -> str:
    """Join two blocks split by a page break.

    A paragraph that continues on the next page can end with a typeset soft
    hyphen (``opera‐``) or an ASCII hyphen (``blame-``). The hyphenation rule is
    the same as inside a block, minus the space.
    """
    previous = previous.rstrip()
    following = following.lstrip()
    if not previous:
        return following
    if not following:
        return previous
    if previous.endswith(tuple(SOFT_HYPHENS)):
        return previous[:-1] + following
    if previous.endswith(tuple(HARD_HYPHENS)):
        return previous + following
    return f"{previous} {following}"


# ---------------------------------------------------------------------------
# Review normalization
# ---------------------------------------------------------------------------

_TYPOGRAPHIC = str.maketrans(
    {
        "\u2019": "'",  # ’
        "\u2018": "'",  # ‘
        "\u201a": "'",
        "\u201c": '"',  # “
        "\u201d": '"',  # ”
        "\u201e": '"',
        "\u2013": "-",  # en dash
        "\u2014": ", ",  # em dash: a spoken pause
        "\u2212": "-",  # minus sign
        "\u2026": "...",
        "\u00a0": " ",
        "\u2007": " ",
        "\u202f": " ",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb00": "ff",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u2022": "",  # bullet
        "\u00b7": "",
        "\u25aa": "",
        "\u25e6": "",
        "\u2010": "-",
        "\u00ad": "",
    }
)

_MULTISPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_DUPLICATE_COMMA_RE = re.compile(r",(\s*,)+")
_COMMA_AFTER_STOP_RE = re.compile(r"([.!?])\s*,")
_COMMA_BEFORE_STOP_RE = re.compile(r",\s*([.!?])")


def normalize_for_review(text: str) -> str:
    text = text.translate(_TYPOGRAPHIC)
    text = _MULTISPACE_RE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _DUPLICATE_COMMA_RE.sub(",", text)
    # An em dash right after a full stop (quote formatting) becomes ", ";
    # that comma is noise, not punctuation.
    text = _COMMA_AFTER_STOP_RE.sub(r"\1", text)
    text = _COMMA_BEFORE_STOP_RE.sub(r"\1", text)
    return text.strip(" \t\n")


# ---------------------------------------------------------------------------
# Speech normalization
# ---------------------------------------------------------------------------

# Abbreviations are expanded before sentence splitting so chunking does not
# mistake "e.g." for a sentence boundary.
_ABBREVIATIONS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), replacement)
    for pattern, replacement in (
        (r"\be\.g\.,", "for example,"),
        (r"\be\.g\.", "for example"),
        (r"\bi\.e\.,", "that is,"),
        (r"\bi\.e\.", "that is"),
        (r"\betc\.", "et cetera."),
        (r"\bvs\.", "versus"),
        (r"\bcf\.", "compare"),
        (r"\bapprox\.", "approximately"),
        (r"\bFigs\.", "Figures"),
        (r"\bFig\.", "Figure"),
        (r"\bDr\.", "Doctor"),
        (r"\bMr\.", "Mister"),
        (r"\bMrs\.", "Missus"),
        (r"\bMs\.", "Miss"),
        (r"\bSt\.", "Saint"),
        (r"\bU\.S\.A\.", "U S A"),
        (r"\bU\.S\.", "U S"),
        (r"\bU\.K\.", "U K"),
    )
)

_URL_RE = re.compile(
    r"(?:(?:https?|ftp)://|www\.)[^\s<>\"']+"
    r"|(?<![\w@.])(?:[a-z0-9][a-z0-9-]*\.)+"
    r"(?:com|org|net|io|dev|ai|edu|gov|co|br|info)(?:/[^\s<>\"']*)?",
    re.IGNORECASE,
)

# Initialisms that an American reader pronounces as a word.
WORD_ACRONYMS: dict[str, str] = {
    "UNIX": "UNIX",
    "LINUX": "LINUX",
    "NASA": "NASA",
    "RAM": "RAM",
    "ROM": "ROM",
    "JSON": "jason",
    "YAML": "yaml",
    "GUI": "gooey",
    "GIF": "gif",
    "JPEG": "jpeg",
    "SQL": "sequel",
    "BIOS": "bye-ohs",
    "LAN": "lan",
    "WAN": "wan",
    "RADAR": "radar",
    "LASER": "laser",
    "SCUBA": "scuba",
    "PIN": "pin",
    "RAID": "raid",
    "AIDS": "aids",
    "COVID": "covid",
    "OK": "okay",
}

_PROTECTED_IDENTIFIERS = {
    "iPhone",
    "iPad",
    "iPod",
    "iMac",
    "iOS",
    "macOS",
    "watchOS",
    "tvOS",
    "YouTube",
    "GitHub",
    "GitLab",
    "JavaScript",
    "TypeScript",
    "PostgreSQL",
    "MySQL",
    "OpenStack",
    "Kubernetes",
    "WordPress",
    "OpenAI",
}

_KEBAB_RE = re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+)+\b")
_SNAKE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")
_CAMEL_RE = re.compile(r"\b[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+\b")
_PASCAL_RE = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")
_ACRONYM_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,5})(s?)\b")
_CITATION_RE = re.compile(r"\[[A-Za-z][A-Za-z0-9+.\-]*\d{2,4}\]")
_LABEL_RE = re.compile(
    r"\b(Figure|Table|Chapter|Section|Appendix)\s+(\d+)-(\d+)\b",
    re.IGNORECASE,
)
_EN_RANGE_RE = re.compile(r"(?<=\d)\s*\u2013\s*(?=\d)")
# Thousands separators belong to the number only with three digits after the
# comma; a bare trailing comma is punctuation ("Figure 1, see...").
_NUMBER = r"(?:\d{1,3}(?:,\d{3})*|\d+)"
_PERCENT_RE = re.compile(rf"(?<![\w.])({_NUMBER}(?:\.\d+)?)\s*%")
_MULTIPLIER_RE = re.compile(rf"(?<![\w.])({_NUMBER})\s*[x\u00d7](?![\w])")
_ORDINAL_RE = re.compile(
    rf"(?<![\w.])({_NUMBER})(st|nd|rd|th)\b", re.IGNORECASE
)
_DECADE_RE = re.compile(r"(?<![\w.])(1[89]\d{2}|20\d{2})s\b")
# A trailing full stop is sentence punctuation, not part of the number; a
# period followed by a digit is a version/decimal and is left to those rules.
_YEAR_RE = re.compile(r"(?<![\w.])(1[89]\d{2}|20\d{2})(?![\w]|\.\d)")
_DECIMAL_RE = re.compile(rf"(?<![\w.])({_NUMBER}\.\d+)(?![\w])")
_INTEGER_RE = re.compile(rf"(?<![\w.])({_NUMBER})(?![\w]|\.\d)")
_IP_RE = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b")
_VERSION_RE = re.compile(r"\bv(\d+(?:\.\d+)+)\b", re.IGNORECASE)
_DOTTED_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+){2,})(?![\w])")
_SLASH_RE = re.compile(r"\s*/\s*")

_DIGIT_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)
_TEENS = (
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)
_SCALES = ("", "thousand", "million", "billion")


def normalize_for_speech(text: str) -> str:
    """Turn text into something a TTS model reads naturally in English."""
    text = _EN_RANGE_RE.sub(" to ", text)
    text = normalize_for_review(text)
    text = _CITATION_RE.sub("", text)
    text = _expand_abbreviations(text)
    text = _URL_RE.sub(lambda match: _verbalize_url(match.group(0)), text)
    text = _split_identifiers(text)
    text = _LABEL_RE.sub(
        lambda m: f"{m.group(1)} {m.group(2)} dash {m.group(3)}", text
    )
    text = _ACRONYM_RE.sub(_expand_acronym, text)
    text = _expand_numbers(text)
    text = _clean_punctuation(text)
    return text


def ensure_sentence(text: str) -> str:
    text = text.strip()
    if text and text[-1] not in ".!?;:":
        return text + "."
    return text


def _expand_abbreviations(text: str) -> str:
    for pattern, replacement in _ABBREVIATIONS:
        text = pattern.sub(replacement, text)
    return text


def _verbalize_url(url: str) -> str:
    trailing = ""
    while url and url[-1] in ".,;:!?)\"'":
        trailing = url[-1] + trailing
        url = url[:-1]
    candidate = url if "://" in url else "http://" + url
    parts = urlsplit(candidate)
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    words = " dot ".join(_url_words(label) for label in host.split(".") if label)

    path = parts.path.rstrip("/")
    if path:
        segments = [segment for segment in path.split("/") if segment]
        if segments:
            words += " slash " + " slash ".join(
                _url_words(segment) for segment in segments
            )
    if parts.query:
        words += " query " + _url_words(parts.query.replace("=", " equals "))
    if parts.fragment:
        words += " fragment " + _url_words(parts.fragment)
    return words + trailing


def _url_words(value: str) -> str:
    value = value.replace("%20", " ")
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"\.", " dot ", value)
    value = re.sub(r"[?&=]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _split_identifiers(text: str) -> str:
    text = _KEBAB_RE.sub(lambda m: m.group(0).replace("-", " "), text)
    text = _SNAKE_RE.sub(_split_snake, text)
    text = _CAMEL_RE.sub(_split_camel, text)
    text = _PASCAL_RE.sub(_split_camel, text)
    return text


def _split_snake(match: re.Match[str]) -> str:
    token = match.group(0)
    if token in _PROTECTED_IDENTIFIERS:
        return token
    parts = token.split("_")
    if all(part.isupper() for part in parts if part):
        return " ".join(part.lower() for part in parts)
    return " ".join(parts)


def _split_camel(match: re.Match[str]) -> str:
    token = match.group(0)
    if token in _PROTECTED_IDENTIFIERS:
        return token
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", token)


def _expand_acronym(match: re.Match[str]) -> str:
    token, plural = match.group(1), match.group(2)
    if token in WORD_ACRONYMS:
        return WORD_ACRONYMS[token] + ("s" if plural else "")
    spoken = " ".join(token)
    if plural:
        return spoken + "'s"
    return spoken


def _expand_numbers(text: str) -> str:
    text = _VERSION_RE.sub(lambda m: "version " + _dotted(m.group(1)), text)
    text = _PERCENT_RE.sub(
        lambda m: _number_phrase(m.group(1)) + " percent", text
    )
    text = _MULTIPLIER_RE.sub(
        lambda m: _number_phrase(m.group(1)) + " times", text
    )
    text = _IP_RE.sub(lambda m: _dotted(m.group(1), joiner=" dot "), text)
    text = _DOTTED_RE.sub(lambda m: _dotted(m.group(1)), text)
    text = _ORDINAL_RE.sub(lambda m: _ordinal(int(m.group(1).replace(",", ""))), text)
    text = _DECADE_RE.sub(lambda m: _decade_words(m.group(1)), text)
    text = _YEAR_RE.sub(lambda m: _year_words(int(m.group(1))), text)
    text = _DECIMAL_RE.sub(lambda m: _number_phrase(m.group(1)), text)
    text = _INTEGER_RE.sub(
        lambda m: _int_to_words(int(m.group(1).replace(",", ""))), text
    )
    return text


def _clean_punctuation(text: str) -> str:
    text = re.sub(r"\band/or\b", "and or", text, flags=re.IGNORECASE)
    # Editorial brackets: "[re]integrating" -> "reintegrating".
    text = re.sub(r"\[([A-Za-z]+)\](?=[A-Za-z])", r"\1", text)
    text = re.sub(r"[\[\]{}]", " ", text)
    text = _SLASH_RE.sub(" slash ", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _DUPLICATE_COMMA_RE.sub(",", text)
    text = re.sub(r"\.\s*\.", ".", text)
    text = re.sub(r",\s*\.", ".", text)
    text = _MULTISPACE_RE.sub(" ", text)
    return text.strip()


def _dotted(value: str, joiner: str = " point ") -> str:
    parts = re.split(r"[.-]", value)
    return joiner.join(_digit_words(part) for part in parts)


def _digit_words(value: str) -> str:
    if value.isdigit():
        return " ".join(_DIGIT_WORDS[int(digit)] for digit in value)
    return value


def _number_phrase(value: str) -> str:
    value = value.replace(",", "")
    if "." not in value:
        return _int_to_words(int(value))
    whole, fraction = value.split(".", 1)
    return (
        _int_to_words(int(whole))
        + " point "
        + " ".join(_DIGIT_WORDS[int(digit)] for digit in fraction)
    )


def _int_to_words(number: int) -> str:
    if number < 0:
        return "minus " + _int_to_words(-number)
    if number == 0:
        return "zero"

    groups: list[int] = []
    while number:
        groups.append(number % 1000)
        number //= 1000

    parts: list[str] = []
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if not group:
            continue
        words = _below_thousand(group)
        if _SCALES[index]:
            words += " " + _SCALES[index]
        parts.append(words)
    return " ".join(parts)


def _below_thousand(number: int) -> str:
    words: list[str] = []
    hundreds, rest = divmod(number, 100)
    if hundreds:
        words.append(_DIGIT_WORDS[hundreds] + " hundred")
    if rest:
        if rest < 10:
            words.append(_DIGIT_WORDS[rest])
        elif rest < 20:
            words.append(_TEENS[rest - 10])
        else:
            tens, ones = divmod(rest, 10)
            if ones:
                words.append(_TENS[tens] + "-" + _DIGIT_WORDS[ones])
            else:
                words.append(_TENS[tens])
    return " ".join(words)


def _decade_words(value: str) -> str:
    words = _year_words(int(value))
    if words.endswith("y"):
        return words[:-1] + "ies"
    return words + "s"


def _year_words(year: int) -> str:
    if year == 2000:
        return "two thousand"
    if 2001 <= year <= 2009:
        return "two thousand " + _DIGIT_WORDS[year - 2000]
    if 2010 <= year <= 2099:
        return "twenty " + _below_thousand(year - 2000)
    if year == 1900:
        return "nineteen hundred"
    if 1901 <= year <= 1999:
        rest = year - 1900
        if rest < 10:
            return "nineteen oh " + _DIGIT_WORDS[rest]
        return "nineteen " + _below_thousand(rest)
    return _int_to_words(year)


_ORDINAL_WORDS = {
    "one": "first",
    "two": "second",
    "three": "third",
    "four": "fourth",
    "five": "fifth",
    "six": "sixth",
    "seven": "seventh",
    "eight": "eighth",
    "nine": "ninth",
    "ten": "tenth",
    "eleven": "eleventh",
    "twelve": "twelfth",
    "twenty": "twentieth",
    "thirty": "thirtieth",
    "forty": "fortieth",
    "fifty": "fiftieth",
    "sixty": "sixtieth",
    "seventy": "seventieth",
    "eighty": "eightieth",
    "ninety": "ninetieth",
    "hundred": "hundredth",
    "thousand": "thousandth",
    "million": "millionth",
    "billion": "billionth",
}


def _ordinal(number: int) -> str:
    words = _int_to_words(number)
    head, separator, last = words.rpartition(" ")
    if "-" in last:  # twenty-two -> twenty-second
        tens, _, ones = last.partition("-")
        last = f"{tens}-{_ordinal_word(ones)}"
        return f"{head}{separator}{last}" if head else last
    ordinal = _ordinal_word(last)
    return f"{head}{separator}{ordinal}" if head else ordinal


def _ordinal_word(word: str) -> str:
    return _ORDINAL_WORDS.get(word, word + "th")
