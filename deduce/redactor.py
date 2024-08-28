import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from operator import attrgetter
from random import Random
from typing import Optional, Union, Tuple

from rapidfuzz.distance import DamerauLevenshtein

import docdeid as dd
from docdeid.process import SimpleRedactor


_DATE_SPLITTER_RX = re.compile('[-/. ]+')
_DIGITS = re.compile('[0-9]+')
_MONTHS_SHORT = ['jan', 'feb', 'mrt', 'apr', 'mei', 'jun',
                 'jul', 'aug', 'sep', 'okt', 'nov', 'dec']
_MONTHS_SHORT2 = ['jan', 'feb', 'mrt', 'apr', 'mei', 'jun',
                  'jul', 'aug', 'sept', 'okt', 'nov', 'dec']
_MONTHS_LONG = ['januari', 'februari', 'maart', 'april', 'mei', 'juni',
                'juli', 'augustus', 'september', 'oktober', 'november', 'december']
_MONTH_NAMES = {
        name: (i, 'short')
        for i, name in enumerate(_MONTHS_SHORT, start=1)
    } | {
        name: (i, 'short2')
        for i, name in enumerate(_MONTHS_SHORT2, start=1)
    } | {
        name: (i, 'long')
        for i, name in enumerate(_MONTHS_LONG, start=1)
    }


def try_parse_day(text: str) -> Optional[Tuple[int, bool]]:
    if _DIGITS.fullmatch(text):
        day_num = int(text)
        if 1 <= day_num <= 31:
            return day_num, text.startswith('0')
    return None


def format_day(day: int, parse: (int, bool), orig_date: str) -> str:
    if day < 10 and (parse[1] or ' ' not in orig_date):
        return f'0{day}'
    return str(day)


def try_parse_month(text: str) -> Optional[Tuple[int, Union[bool, str]]]:
    if _DIGITS.fullmatch(text):
        month_num = int(text)
        if 1 <= month_num <= 12:
            return month_num, text.startswith('0')
    return _MONTH_NAMES.get(text.lower())


def format_month(month: int, parse: (int, Union[bool, str]), orig_date: str) -> str:
    if isinstance(parse[1], bool):
        if month < 9 and (parse[1] or ' ' not in orig_date):
            return f'0{month}'
        return str(month)
    return {'short': _MONTHS_SHORT,
            'short2': _MONTHS_SHORT2,
            'long': _MONTHS_LONG}[parse[1]][month - 1]


def try_parse_year(text: str) -> Optional[Tuple[int, str, int]]:
    wo_apostrophe = text.lstrip("'`")
    apos = text[:len(text) - len(wo_apostrophe)]
    if len(wo_apostrophe) == 4:
        return int(wo_apostrophe), apos, 4
    if len(wo_apostrophe) == 2:
        years = int(wo_apostrophe)
        this_year = date.today().year
        century_start = 100 * (this_year // 100)
        if years > this_year - century_start:
            return century_start - 100 + years, apos, 2
        return century_start + years, apos, 2
    return None


def format_year(year: int, parse: (int, str, int)) -> str:
    y_display = year % 100 if parse[2] == 2 else year
    return '{0}{1:0>{2}}'.format(parse[1], y_display, parse[2])


@dataclass
class DateStrategy:
    # XXX Modelling this as a small class hierarchy with one class per strategy
    #   would perhaps be more appropriate, but also too heavy for what it's worth.
    strategy: str
    init_shift: Optional[int]
    _random: Optional[Random] = field(init=False, repr=False, default=None)
    _shift: int = field(init=False, repr=False, default=0)
    _last_stay_id: Optional[str] = field(init=False, repr=False, default=None)

    def on_document(self, metadata: Optional[dd.MetaData]):
        if self.strategy == "shift":
            # Keep the same shift if still processing the same stay.
            stay_id = None if metadata is None else metadata['stay_id']
            if stay_id is not None and stay_id == self._last_stay_id:
                return
            self._last_stay_id = stay_id
            # Initialize randomizer the first time.
            if self._random is None:
                if self.init_shift is None:
                    self._random = Random()
                else:
                    # When specified, make sure to use init_shift, not a random number.
                    self._random = Random(424242 - self.init_shift)  # whatever
                    self._shift = self.init_shift
                    logging.debug("Updated date shift amount to %d.", self._shift)
                    return
            # Compute a random shift amount.
            rand_shift = self._random.randint(-61, 60)
            # Avoid the zero shift.
            self._shift = rand_shift + 1 if rand_shift >= 0 else rand_shift
            logging.debug("Updated date shift amount to %d.", self._shift)

    def redact(self, anno: dd.Annotation) -> str:
        if self.strategy != "shift":
            raise ValueError("DateStrategy.redact is only implemented for the 'shift' "
                             "strategy.")

        # Parse.
        date_parts = _DATE_SPLITTER_RX.split(anno.text)
        if len(date_parts) <= 1:
            # Too vague (just the year), keep the original.
            return anno.text
        if len(date_parts) > 3:
            # Unusual date format or not a date, don't know how to handle.
            logging.error('Failed to parse "%s" as a valid date: too many parts',
                          anno.text)
            return anno.text
        if len(date_parts) <= 2:
            logging.error('Failed to parse "%s" as a valid date: just 2 parts',
                          anno.text)
            return anno.text
        m_parse = try_parse_month(date_parts[1])
        if m_parse is None:
            logging.error('Failed to parse "%s" as a valid date: cannot parse month',
                          anno.text)
            return anno.text
        d_parse = try_parse_day(date_parts[0])
        y_parse = try_parse_year(date_parts[2])
        if d_parse is not None and y_parse is not None:
            year_idx = 2
        else:
            d_parse = try_parse_day(date_parts[2])
            y_parse = try_parse_year(date_parts[0])
            if d_parse is not None and y_parse is not None:
                year_idx = 0
            else:
                if d_parse is None:
                    logging.error(
                        'Failed to parse "%s" as a valid date: cannot parse day',
                        anno.text)
                if y_parse is None:
                    logging.error(
                        'Failed to parse "%s" as a valid date: cannot parse year',
                        anno.text)
                return anno.text

        # Shift.
        try:
            shifted = date(y_parse[0], m_parse[0], d_parse[0]) + timedelta(days=self._shift)
        except ValueError as e:
            logging.error('Failed to parse "%s" as a valid date: %s',
                          anno.text, e)
            return anno.text

        # Format.
        joiners = _DATE_SPLITTER_RX.findall(anno.text)
        shifted_parts = [format_day(shifted.day, d_parse, anno.text),
                         format_month(shifted.month, m_parse, anno.text),
                         format_year(shifted.year, y_parse)]
        ordered_parts = (shifted_parts if year_idx == 2 else
                         list(reversed(shifted_parts)))
        return '{d[0]}{j[0]}{d[1]}{j[1]}{d[2]}'.format(j=joiners, d=ordered_parts)


class DeduceRedactor(SimpleRedactor):
    """
    Implements the redacting logic of Deduce:

    - All annotations with "patient" tag are replaced with <PATIENT>
    - With the "shift" `date_strategy`, dates are shifted by a number of days
      which is constant within a document but randomized across documents.
    - All other annotations are replaced with <TAG-n>, with n identifying a group
        of annotations with a similar text (edit_distance <= 1).
    """

    def __init__(self,
                 date_strategy: Optional[DateStrategy] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.date_strategy = date_strategy or DateStrategy("hide", 0)

    def redact(self,
               text: str,
               annotations: dd.AnnotationSet,
               metadata: Optional[dd.MetaData] = None) -> str:
        repls = {}
        self.date_strategy.on_document(metadata)

        for tag, same_tag_annos in SimpleRedactor._group_by_tag(annotations):
            sorted_annos = sorted(same_tag_annos, key=attrgetter('end_char'))
            if tag == "patient":
                repls.update((anno, f"{self.open_char}PATIENT{self.close_char}")
                             for anno in sorted_annos)
                continue

            if tag == "datum" and self.date_strategy.strategy == "shift":
                repls.update((anno, self.date_strategy.redact(anno))
                             for anno in sorted_annos)
                continue

            same_tag_repls: dict[dd.Annotation, str] = {}
            for anno in sorted_annos:
                # Look for existing similar mentions with this tag.
                for same_tag_anno, repl in same_tag_repls.items():
                    if (
                        DamerauLevenshtein.distance(
                            anno.text, same_tag_anno.text, score_cutoff=1
                        )
                        <= 1
                    ):
                        same_tag_repls[anno] = repl
                        break
                else:
                    # If no existing mention, build a new replacement string.
                    same_tag_repls[anno] = (
                        f"{self.open_char}"
                        f"{tag.upper()}"
                        f"-"
                        f"{len(same_tag_repls) + 1}"
                        f"{self.close_char}"
                    )

            repls.update(same_tag_repls)

        return self._replace_annotations_in_text(text, annotations, repls)
