import re
from unittest.mock import patch

import docdeid as dd
import pytest

from deduce.annotator import (
    BsnAnnotator,
    ContextAnnotator,
    ContextPattern,
    DynamicNameAnnotator,
    PatientNameAnnotator,
    PhoneNumberAnnotator,
    RegexpPseudoAnnotator
)
from docdeid import Annotation
from docdeid.direction import Direction
from docdeid.process.annotator import (
    as_token_pattern,
    _PatternPositionMatcher,
    SequencePattern,
    SimpleTokenPattern,
)
from deduce.person import Person
from deduce.tokenizer import DeduceTokenizer
from tests.helpers import linked_tokens

# import pydevd_pycharm
# pydevd_pycharm.settrace()


@pytest.fixture
def ds():
    ds = dd.ds.DsCollection()

    first_names = ["Andries", "pieter", "Aziz", "Bernard"]
    surnames = ["Meijer", "Smit", "Bakker", "Heerma"]

    ds["first_names"] = dd.ds.LookupSet()
    ds["first_names"].add_items_from_iterable(items=first_names)

    ds["surnames"] = dd.ds.LookupSet()
    ds["surnames"].add_items_from_iterable(items=surnames)

    return ds


@pytest.fixture
def tokenizer():
    return DeduceTokenizer()


@pytest.fixture
def regexp_pseudo_doc(tokenizer):

    return dd.Document(
        text="De patient is Na 12 jaar gestopt met medicijnen.",
        tokenizers={"default": tokenizer},
    )


@pytest.fixture
def pattern_doc(tokenizer):
    return dd.Document(
        text="De man heet Andries Meijer-Heerma, voornaam Andries.",
        tokenizers={"default": tokenizer},
    )


@pytest.fixture
def bsn_doc():
    d = dd.DocDeid()

    return d.deidentify(
        text="Geldige voorbeelden zijn: 111222333 en 123456782. "
        "Patientnummer is 01234, en ander id 01234567890."
    )


@pytest.fixture
def phone_number_doc():
    d = dd.DocDeid()

    return d.deidentify(
        text="Telefoonnummers zijn 0314-555555, (088 755 55 55) of (06)55555555, "
        "maar 065555 is te kort en 065555555555 is te lang. "
        "Verwijsnummer is 0800-9003."
    )


@pytest.fixture
def surname_pattern():
    return linked_tokens(["Van der", "Heide", "-", "Ginkel"])


def token(text: str):
    return dd.Token(text=text, start_char=0, end_char=len(text))

@pytest.fixture
def ip_pavlov():
    tokens = linked_tokens(["Uw", "patient", "I", ".", "P", ".", "Pavlov", "is",
                            "overleden"])
    return {
        'text': "Uw patient I . P . Pavlov is overleden",
        'tokens': tokens,
        'annos': [
            Annotation(tag='initiaal', text='I .',
                       start_char=11, start_token=tokens[2],
                       end_char=14, end_token=tokens[3]),
            Annotation(tag='initiaal', text='P .',
                       start_char=15, start_token=tokens[4],
                       end_char=18, end_token=tokens[5]),
            Annotation(tag='persoon', text='I . P . Pavlov',
                       start_char=11, start_token=tokens[2],
                       end_char=25, end_token=tokens[6]),
            Annotation(tag='persoon', text='Pavlov',
                       start_char=19, start_token=tokens[6],
                       end_char=25, end_token=tokens[6]),
        ]
    }

class TestPositionMatcher:
    def test_equal(self):
        assert _PatternPositionMatcher.match({"equal": "test"}, token=token("test"))[0]
        assert not _PatternPositionMatcher.match({"equal": "_"}, token=token("test"))[0]

    def test_equal_with_dataclass(self):
        assert _PatternPositionMatcher.match(SimpleTokenPattern("equal", "test"),
                                             token=token("test"))[0]

    def test_re_match(self):
        assert _PatternPositionMatcher.match({"re_match": "[a-z]"},
                                             token=token("abc"))[0]
        assert _PatternPositionMatcher.match(
            {"re_match": "[a-z]"}, token=token("abc123")
        )[0]
        assert not _PatternPositionMatcher.match({"re_match": "[a-z]"},
                                                 token=token(""))[0]
        assert not _PatternPositionMatcher.match(
            {"re_match": "[a-z]"}, token=token("123")
        )[0]
        assert not _PatternPositionMatcher.match(
            {"re_match": "[a-z]"}, token=token("123abc")
        )[0]

    def test_tag(self, ip_pavlov):
        assert _PatternPositionMatcher.match({"tag": "initiaal"},
                                             annos=ip_pavlov['annos'])[0]
        assert not _PatternPositionMatcher.match({"tag": "id"},
                                                 annos=ip_pavlov['annos'])[0]

    def test_tagged_mention(self, ip_pavlov):
        person_match = _PatternPositionMatcher.match(
            {"tagged_mention": "persoon"},
            annos=ip_pavlov['annos'],
            token=ip_pavlov['tokens'][6],
            dir=Direction.LEFT)
        # The longest mention spanning to the left from the 7th token having the target
        # tag should be returned.
        assert person_match[0]
        assert person_match[1].text == 'I . P . Pavlov'

        assert not _PatternPositionMatcher.match(
            {"tagged_mention": "persoon"},
            annos=ip_pavlov['annos'],
            token=ip_pavlov['tokens'][5],
            dir=Direction.LEFT)[0]
        assert not _PatternPositionMatcher.match(
            {"tagged_mention": "persoon"},
            annos=ip_pavlov['annos'],
            token=ip_pavlov['tokens'][7],
            dir=Direction.LEFT)[0]

        initials_match = _PatternPositionMatcher.match(
            {"tagged_mention": "initiaal"},
            annos=ip_pavlov['annos'],
            token=ip_pavlov['tokens'][4],
            dir=Direction.RIGHT)
        assert initials_match[0]
        assert initials_match[1].text == 'P .'

        assert not _PatternPositionMatcher.match(
            {"tagged_mention": "initiaal"},
            annos=ip_pavlov['annos'],
            token=ip_pavlov['tokens'][3],
            dir=Direction.RIGHT)[0]

    def test_is_initials(self):

        assert _PatternPositionMatcher.match({"is_initials": True},
                                             token=token("A"))[0]
        assert _PatternPositionMatcher.match({"is_initials": True},
                                             token=token("AB"))[0]
        assert _PatternPositionMatcher.match({"is_initials": True},
                                             token=token("ABC"))[0]
        assert _PatternPositionMatcher.match({"is_initials": True},
                                             token=token("ABCD"))[0]
        assert not _PatternPositionMatcher.match(
            {"is_initials": True}, token=token("ABCDE")
        )[0]
        assert not _PatternPositionMatcher.match({"is_initials": True},
                                                 token=token(""))[0]
        assert not _PatternPositionMatcher.match(
            {"is_initials": True}, token=token("abcd")
        )[0]
        assert not _PatternPositionMatcher.match(
            {"is_initials": True}, token=token("abcde")
        )[0]

    def test_match_like_name(self):
        pattern_position = {"like_name": True}

        assert _PatternPositionMatcher.match(pattern_position,
                                             token=token("Diederik"))[0]
        assert not _PatternPositionMatcher.match(pattern_position, token=token("Le"))[0]
        assert not _PatternPositionMatcher.match(
            pattern_position, token=token("diederik")
        )[0]
        assert not _PatternPositionMatcher.match(
            pattern_position, token=token("Diederik3")
        )[0]

    def test_match_lookup(self, ds):
        assert _PatternPositionMatcher.match(
            {"lookup": "first_names"}, token=token("Andries"), ds=ds
        )[0]
        assert not _PatternPositionMatcher.match(
            {"lookup": "first_names"}, token=token("andries"), ds=ds
        )[0]
        assert not _PatternPositionMatcher.match(
            {"lookup": "surnames"}, token=token("Andries"), ds=ds
        )[0]
        assert not _PatternPositionMatcher.match(
            {"lookup": "first_names"}, token=token("Smit"), ds=ds
        )[0]
        assert _PatternPositionMatcher.match(
            {"lookup": "surnames"}, token=token("Smit"), ds=ds
        )[0]
        assert not _PatternPositionMatcher.match(
            {"lookup": "surnames"}, token=token("smit"), ds=ds
        )[0]

    def test_match_neg_lookup(self, ds):
        assert not _PatternPositionMatcher.match(
            {"neg_lookup": "first_names"}, token=token("Andries"), ds=ds
        )[0]
        assert _PatternPositionMatcher.match(
            {"neg_lookup": "first_names"}, token=token("andries"), ds=ds
        )[0]
        assert _PatternPositionMatcher.match(
            {"neg_lookup": "surnames"}, token=token("Andries"), ds=ds
        )[0]
        assert _PatternPositionMatcher.match(
            {"neg_lookup": "first_names"}, token=token("Smit"), ds=ds
        )[0]
        assert not _PatternPositionMatcher.match(
            {"neg_lookup": "surnames"}, token=token("Smit"), ds=ds
        )[0]
        assert _PatternPositionMatcher.match(
            {"neg_lookup": "surnames"}, token=token("smit"), ds=ds
        )[0]

    def test_match_and(self):
        assert _PatternPositionMatcher.match(
            {"and": [{"equal": "Abcd"}, {"like_name": True}]},
            token=token("Abcd"),
            ds=ds,
        )[0]
        assert not _PatternPositionMatcher.match(
            {"and": [{"equal": "dcef"}, {"like_name": True}]},
            token=token("Abcd"),
            ds=ds,
        )[0]
        assert not _PatternPositionMatcher.match(
            {"and": [{"equal": "A"}, {"like_name": True}]}, token=token("A"), ds=ds
        )[0]
        assert not _PatternPositionMatcher.match(
            {"and": [{"equal": "b"}, {"like_name": True}]}, token=token("a"), ds=ds
        )[0]

    def test_match_or(self):
        assert _PatternPositionMatcher.match(
            {"or": [{"equal": "Abcd"}, {"like_name": True}]}, token=token("Abcd"), ds=ds
        )[0]
        assert _PatternPositionMatcher.match(
            {"or": [{"equal": "dcef"}, {"like_name": True}]}, token=token("Abcd"), ds=ds
        )[0]
        assert _PatternPositionMatcher.match(
            {"or": [{"equal": "A"}, {"like_name": True}]}, token=token("A"), ds=ds
        )[0]
        assert not _PatternPositionMatcher.match(
            {"or": [{"equal": "b"}, {"like_name": True}]}, token=token("a"), ds=ds
        )[0]


class TestContextAnnotator:
    def test_apply_context_pattern(self, pattern_doc):
        annotator = ContextAnnotator(pattern=[])

        annotations = dd.AnnotationSet(
            [
                dd.Annotation(
                    text="Andries",
                    start_char=12,
                    end_char=19,
                    tag="voornaam",
                    start_token=pattern_doc.get_tokens()[3],
                    end_token=pattern_doc.get_tokens()[3],
                )
            ]
        )

        assert annotator._apply_context_pattern(
            pattern_doc,
            ContextPattern({"voornaam"},
                           "{tag}+naam",
                           SequencePattern(Direction.RIGHT,
                                           set(),
                                           [as_token_pattern({"like_name": True})])),
            annotations,
        ) == dd.AnnotationSet(
            [
                dd.Annotation(
                    text="Andries Meijer",
                    start_char=12,
                    end_char=26,
                    tag="voornaam+naam",
                )
            ]
        )

    def test_apply_context_pattern_left(self, pattern_doc):
        annotator = ContextAnnotator(pattern=[])

        annotations = dd.AnnotationSet(
            [
                dd.Annotation(
                    text="Meijer",
                    start_char=20,
                    end_char=26,
                    tag="achternaam",
                    start_token=pattern_doc.get_tokens()[4],
                    end_token=pattern_doc.get_tokens()[4],
                )
            ]
        )

        assert annotator._apply_context_pattern(
            pattern_doc,
            ContextPattern({"achternaam"},
                           "naam+{tag}",
                           SequencePattern(Direction.LEFT,
                                           set(),
                                           [as_token_pattern({"like_name": True})])),
            annotations,
        ) == dd.AnnotationSet(
            [
                dd.Annotation(
                    text="Andries Meijer",
                    start_char=12,
                    end_char=26,
                    tag="naam+achternaam",
                )
            ]
        )

    def test_apply_context_pattern_skip(self, pattern_doc):
        annotator = ContextAnnotator(pattern=[])

        annotations = dd.AnnotationSet(
            [
                dd.Annotation(
                    text="Meijer",
                    start_char=20,
                    end_char=26,
                    tag="achternaam",
                    start_token=pattern_doc.get_tokens()[4],
                    end_token=pattern_doc.get_tokens()[4],
                )
            ]
        )

        assert annotator._apply_context_pattern(
            pattern_doc,
            ContextPattern({"achternaam"},
                           "{tag}+naam",
                           SequencePattern(Direction.RIGHT,
                                           {"-"},
                                           [as_token_pattern({"like_name": True})])),
            annotations,
        ) == dd.AnnotationSet(
            [
                dd.Annotation(
                    text="Meijer-Heerma",
                    start_char=20,
                    end_char=33,
                    tag="achternaam+naam",
                )
            ]
        )

    def test_annotate_multiple(self, pattern_doc):
        pattern = [
            {
                "pattern": [{"like_name": True}],
                "direction": "right",
                "pre_tag": "voornaam",
                "tag": "{tag}+naam",
            },
            {
                "pattern": [{"like_name": True}],
                "direction": "right",
                "skip": ["-"],
                "pre_tag": "achternaam",
                "tag": "{tag}+naam",
            },
        ]

        annotator = ContextAnnotator(pattern=pattern, iterative=False)

        pattern_doc.annotations = dd.AnnotationSet(
            [
                dd.Annotation(
                    text="Andries",
                    start_char=12,
                    end_char=19,
                    tag="voornaam",
                    start_token=pattern_doc.get_tokens()[3],
                    end_token=pattern_doc.get_tokens()[3],
                )
            ]
        )

        assert annotator._get_annotations(pattern_doc) == dd.AnnotationSet(
            {
                dd.Annotation(
                    text="Andries Meijer-Heerma",
                    start_char=12,
                    end_char=33,
                    tag="voornaam+naam+naam",
                )
            }
        )

    def test_annotate_iterative(self, pattern_doc):
        pattern = [
            {
                "pattern": [{"like_name": True}],
                "direction": "right",
                "skip": ["-"],
                "pre_tag": ["naam", "voornaam"],
                "tag": "{tag}+naam",
            }
        ]

        annotator = ContextAnnotator(pattern=pattern, iterative=True)

        pattern_doc.annotations = dd.AnnotationSet(
            [
                dd.Annotation(
                    text="Andries",
                    start_char=12,
                    end_char=19,
                    tag="voornaam",
                    start_token=pattern_doc.get_tokens()[3],
                    end_token=pattern_doc.get_tokens()[3],
                )
            ]
        )

        assert annotator._get_annotations(pattern_doc) == dd.AnnotationSet(
            {
                dd.Annotation(
                    text="Andries Meijer-Heerma",
                    start_char=12,
                    end_char=33,
                    tag="voornaam+naam+naam",
                )
            }
        )


class TestPatientNameAnnotator:
    def test_match_first_name_multiple(self, tokenizer):
        words = ["Jan", "Adriaan"]
        metadata = {"patient": Person(first_names=["Jan", "Adriaan"])}
        tokens = linked_tokens(words)
        doc = dd.Document(text=" ".join(words), metadata=metadata)

        ann = PatientNameAnnotator(tokenizer=tokenizer, tag="_")
        meta_matchers = ann._build_matchers_for_doc(doc)

        matched_0 = list(ann._annotate_token(doc, tokens[0], meta_matchers))
        assert [anno.start_token for anno in matched_0] == [tokens[0]]
        assert [anno.end_token for anno in matched_0] == [tokens[0]]

        matched_1 = list(ann._annotate_token(doc, tokens[1], meta_matchers))
        assert [anno.start_token for anno in matched_1] == [tokens[1]]
        assert [anno.end_token for anno in matched_1] == [tokens[1]]

    def test_match_first_name_fuzzy(self, tokenizer):
        words = ["Adriana"]
        tokens = linked_tokens(words)
        metadata = {"patient": Person(first_names=["Adriaan"])}
        doc = dd.Document(text=" ".join(words), metadata=metadata)

        ann = PatientNameAnnotator(tokenizer=tokenizer, tag="_")
        meta_matchers = ann._build_matchers_for_doc(doc)

        matched_0 = list(ann._annotate_token(doc, tokens[0], meta_matchers))
        assert [anno.start_token for anno in matched_0] == [tokens[0]]
        assert [anno.end_token for anno in matched_0] == [tokens[0]]

    def test_match_first_name_fuzzy_short(self, tokenizer):
        words = ["Dan"]
        tokens = linked_tokens(words)
        metadata = {"patient": Person(first_names=["Jan"])}
        doc = dd.Document(text=" ".join(words), metadata=metadata)

        ann = PatientNameAnnotator(tokenizer=tokenizer, tag="_")
        meta_matchers = ann._build_matchers_for_doc(doc)

        matched_0 = list(ann._annotate_token(doc, tokens[0], meta_matchers))
        assert matched_0 == []

    def test_match_initial_from_name(self, tokenizer):
        words = ["A", "J"]
        tokens = linked_tokens(words)
        metadata = {"patient": Person(first_names=["Jan", "Adriaan"])}
        doc = dd.Document(text=" ".join(words), metadata=metadata)

        ann = PatientNameAnnotator(tokenizer=tokenizer, tag="_")
        meta_matchers = ann._build_matchers_for_doc(doc)

        matched_1 = list(ann._annotate_token(doc, tokens[0], meta_matchers))
        assert [anno.start_token for anno in matched_1] == [tokens[0]]
        assert [anno.end_token for anno in matched_1] == [tokens[0]]

        matched_1 = list(ann._annotate_token(doc, tokens[1], meta_matchers))
        assert [anno.start_token for anno in matched_1] == [tokens[1]]
        assert [anno.end_token for anno in matched_1] == [tokens[1]]

    def test_match_initial_from_name_with_period(self, tokenizer):
        words = ["J", ".", "A", "."]
        metadata = {"patient": Person(first_names=["Jan", "Adriaan"])}
        tokens = linked_tokens(words)
        doc = dd.Document(text=" ".join(words), metadata=metadata)

        ann = PatientNameAnnotator(tokenizer=tokenizer, tag="_")
        meta_matchers = ann._build_matchers_for_doc(doc)

        matched_0 = list(ann._annotate_token(doc, tokens[0], meta_matchers))
        assert [anno.start_token for anno in matched_0] == [tokens[0]]
        assert [anno.end_token for anno in matched_0] == [tokens[1]]

        matched_2 = list(ann._annotate_token(doc, tokens[2], meta_matchers))
        assert [anno.start_token for anno in matched_2] == [tokens[2]]
        assert [anno.end_token for anno in matched_2] == [tokens[3]]

    def test_match_initial_from_name_no_match(self, tokenizer):
        words = ["F", "T"]
        metadata = {"patient": Person(first_names=["Jan", "Adriaan"])}
        tokens = linked_tokens(words)
        doc = dd.Document(text=" ".join(words), metadata=metadata)

        ann = PatientNameAnnotator(tokenizer=tokenizer, tag="_")
        meta_matchers = ann._build_matchers_for_doc(doc)

        matched_0 = list(ann._annotate_token(doc, tokens[0], meta_matchers))
        assert matched_0 == []

        matched_1 = list(ann._annotate_token(doc, tokens[1], meta_matchers))
        assert matched_1 == []

    def test_match_initials(self, tokenizer):
        initials = "AFTH"
        tokens = linked_tokens(["AFTH", "THFA"])

        matched_0 = PatientNameAnnotator._match_initials(tokens[0], initials)
        matched_1 = PatientNameAnnotator._match_initials(tokens[1], initials)
        assert matched_0 == (tokens[0], tokens[0])
        assert matched_1 is None

    def test_match_surname_equal(self, tokenizer, surname_pattern):
        words = ["Van der", "Heide", "-", "Ginkel", "is", "de", "naam"]
        metadata = {"surname_pattern": surname_pattern}
        tokens = linked_tokens(words)
        doc = dd.Document(text=" ".join(words), metadata=metadata)

        ann = PatientNameAnnotator(tokenizer=tokenizer, tag="_")
        meta_matchers = ann._build_matchers_for_doc(doc)

        with patch.object(tokenizer, "tokenize", return_value=surname_pattern):
            matched_0 = list(ann._annotate_token(doc, tokens[0], meta_matchers))
            assert [anno.start_token for anno in matched_0] == [tokens[0]]
            assert [anno.end_token for anno in matched_0] == [tokens[3]]

    def test_match_surname_longer_than_tokens(self, tokenizer, surname_pattern):
        words = ["Van der", "Heide"]
        metadata = {"surname_pattern": surname_pattern}
        tokens = linked_tokens(words)
        doc = dd.Document(text=" ".join(words), metadata=metadata)

        ann = PatientNameAnnotator(tokenizer=tokenizer, tag="_")
        meta_matchers = ann._build_matchers_for_doc(doc)

        with patch.object(tokenizer, "tokenize", return_value=surname_pattern):
            matched_0 = list(ann._annotate_token(doc, tokens[0], meta_matchers))
            assert matched_0 == []

    def test_match_surname_fuzzy(self, tokenizer, surname_pattern):
        words = ["Van der", "Heijde", "-", "Ginkle", "is", "de", "naam"]
        metadata = {"surname_pattern": surname_pattern}
        tokens = linked_tokens(words)
        doc = dd.Document(text=" ".join(words), metadata=metadata)

        ann = PatientNameAnnotator(tokenizer=tokenizer, tag="_")
        meta_matchers = ann._build_matchers_for_doc(doc)

        with patch.object(tokenizer, "tokenize", return_value=surname_pattern):
            matched_0 = list(ann._annotate_token(doc, tokens[0], meta_matchers))
            assert [anno.start_token for anno in matched_0] == [tokens[0]]
            assert [anno.end_token for anno in matched_0] == [tokens[3]]

    def test_match_surname_unequal_first(self, tokenizer, surname_pattern):
        words = ["v/der", "Heide", "-", "Ginkel", "is", "de", "naam"]
        metadata = {"surname_pattern": surname_pattern}
        tokens = linked_tokens(words)
        doc = dd.Document(text=" ".join(words), metadata=metadata)

        ann = PatientNameAnnotator(tokenizer=tokenizer, tag="_")
        meta_matchers = ann._build_matchers_for_doc(doc)

        with patch.object(tokenizer, "tokenize", return_value=surname_pattern):
            matched_0 = list(ann._annotate_token(doc, tokens[0], meta_matchers))
            assert matched_0 == []

    def test_match_surname_unequal_first_fuzzy(self, tokenizer, surname_pattern):
        words = ["Van den", "Heide", "-", "Ginkel", "is", "de", "naam"]
        metadata = {"surname_pattern": surname_pattern}
        tokens = linked_tokens(words)
        doc = dd.Document(text=" ".join(words), metadata=metadata)

        ann = PatientNameAnnotator(tokenizer=tokenizer, tag="_")
        meta_matchers = ann._build_matchers_for_doc(doc)

        with patch.object(tokenizer, "tokenize", return_value=surname_pattern):
            matched_0 = list(ann._annotate_token(doc, tokens[0], meta_matchers))
            assert [anno.start_token for anno in matched_0] == [tokens[0]]
            assert [anno.end_token for anno in matched_0] == [tokens[3]]

    def test_annotate_first_name(self, tokenizer):
        metadata = {
            "patient": Person(
                first_names=["Jan", "Johan"], initials="JJ", surname="Jansen"
            )
        }
        text = "De patient heet Jan"
        tokens = tokenizer.tokenize(text)
        doc = dd.Document(text=text, metadata=metadata)

        ann = PatientNameAnnotator(tokenizer=tokenizer, tag="_")

        with patch.object(doc, "get_tokens", return_value=tokens):
            with patch.object(
                tokenizer, "tokenize", return_value=linked_tokens(["Jansen"])
            ):
                annotations = ann.annotate(doc)

        assert annotations == [
            dd.Annotation(
                text="Jan",
                start_char=16,
                end_char=19,
                tag="voornaam_patient",
            )
        ]

    def test_annotate_initials_from_name(self, tokenizer):
        metadata = {
            "patient": Person(
                first_names=["Jan", "Johan"], initials="JJ", surname="Jansen"
            )
        }
        text = "De patient heet JJ"
        tokens = tokenizer.tokenize(text)
        doc = dd.Document(text=text, metadata=metadata)

        ann = PatientNameAnnotator(tokenizer=tokenizer, tag="_")

        with patch.object(doc, "get_tokens", return_value=tokens):
            with patch.object(
                tokenizer, "tokenize", return_value=linked_tokens(["Jansen"])
            ):
                annotations = ann.annotate(doc)

        assert annotations == [
            dd.Annotation(
                text="JJ",
                start_char=16,
                end_char=18,
                tag="initiaal_patient",
            )
        ]

    def test_annotate_initial(self, tokenizer):
        metadata = {
            "patient": Person(
                first_names=["Jan", "Johan"], initials="JJ", surname="Jansen"
            )
        }
        text = "De patient heet J."
        tokens = tokenizer.tokenize(text)
        doc = dd.Document(text=text, metadata=metadata)

        ann = PatientNameAnnotator(tokenizer=tokenizer, tag="_")

        with patch.object(doc, "get_tokens", return_value=tokens):
            with patch.object(
                tokenizer, "tokenize", return_value=linked_tokens(["Jansen"])
            ):
                annotations = ann.annotate(doc)

        assert annotations == [
            dd.Annotation(
                text="J.",  # Jan
                start_char=16,
                end_char=18,
                tag="initiaal_patient",
            ),
            dd.Annotation(
                text="J.",  # Johan
                start_char=16,
                end_char=18,
                tag="initiaal_patient",
            )
        ]

    def test_annotate_surname(self, tokenizer):
        metadata = {
            "patient": Person(
                first_names=["Jan", "Johan"], initials="JJ", surname="Jansen"
            )
        }
        text = "De patient heet Jansen"
        tokens = tokenizer.tokenize(text)
        doc = dd.Document(text=text, metadata=metadata)

        ann = PatientNameAnnotator(tokenizer=tokenizer, tag="_")

        with patch.object(doc, "get_tokens", return_value=tokens):
            with patch.object(
                tokenizer, "tokenize", return_value=linked_tokens(["Jansen"])
            ):
                annotations = ann.annotate(doc)

        assert annotations == [
            dd.Annotation(
                text="Jansen",
                start_char=16,
                end_char=22,
                tag="achternaam_patient",
            )
        ]


class TestDynamicNameAnnotator:
    def test_annotate_one(self, tokenizer):
        metadata = {
            "arts": [Person(
                first_names=["Jan", "Johan"],
                initials="JJ",
                surname="Jansen",
            )]
        }
        text = "De doctor heet Jansen"
        tokens = tokenizer.tokenize(text)

        ann = DynamicNameAnnotator(meta_key='arts', tokenizer=tokenizer, tag="_")
        doc = dd.Document(text=text, metadata=metadata)

        with patch.object(doc, "get_tokens", return_value=tokens):
            annotations = ann.annotate(doc)

        assert annotations == [
            dd.Annotation(
                text="Jansen",
                start_char=15,
                end_char=21,
                tag="achternaam_arts",
            )
        ]

    def test_annotate_many(self, tokenizer):
        metadata = {
            "arts": [
                Person(
                    first_names=["Ron"],
                    surname="Rivest",
                ),
                Person(
                    first_names=["Adi"],
                    surname="Shamir",
                ),
                Person(
                    first_names=["Leonard"],
                    surname="Adleman",
                ),
            ]
        }
        text = "Met vriendelijke groeten, Rivest, Shamir, Adleman"
        tokens = tokenizer.tokenize(text)

        ann = DynamicNameAnnotator(meta_key='arts', tokenizer=tokenizer, tag="_")
        doc = dd.Document(text=text, metadata=metadata)

        with patch.object(doc, "get_tokens", return_value=tokens):
            annotations = ann.annotate(doc)

        assert annotations == [
            dd.Annotation(
                text="Rivest",
                start_char=26,
                end_char=32,
                tag="achternaam_arts",
            ),
            dd.Annotation(
                text="Shamir",
                start_char=34,
                end_char=40,
                tag="achternaam_arts",
            ),
            dd.Annotation(
                text="Adleman",
                start_char=42,
                end_char=49,
                tag="achternaam_arts",
            ),
        ]


class TestRegexpPseudoAnnotator:
    def test_is_word_char(self):

        assert RegexpPseudoAnnotator._is_word_char("a")
        assert RegexpPseudoAnnotator._is_word_char("abc")
        assert not RegexpPseudoAnnotator._is_word_char("123")
        assert not RegexpPseudoAnnotator._is_word_char(" ")
        assert not RegexpPseudoAnnotator._is_word_char("\n")
        assert not RegexpPseudoAnnotator._is_word_char(".")

    def test_get_previous_word(self):

        r = RegexpPseudoAnnotator(regexp_pattern="_", tag="_")

        assert r._get_previous_word(0, "12 jaar") == ""
        assert r._get_previous_word(1, "<12 jaar") == ""
        assert r._get_previous_word(8, "patient 12 jaar") == "patient"
        assert r._get_previous_word(7, "(sinds 12 jaar)") == "sinds"
        assert r._get_previous_word(11, "patient is 12 jaar)") == "is"

    def test_get_next(self):

        r = RegexpPseudoAnnotator(regexp_pattern="_", tag="_")

        assert r._get_next_word(7, "12 jaar") == ""
        assert r._get_next_word(7, "12 jaar, geleden") == ""
        assert r._get_next_word(7, "12 jaar geleden") == "geleden"
        assert r._get_next_word(7, "12 jaar geleden geopereerd") == "geleden"

    def test_validate_match(self, regexp_pseudo_doc):

        r = RegexpPseudoAnnotator(regexp_pattern="_", tag="_")
        pattern = re.compile(r"\d+ jaar")

        match = list(pattern.finditer(regexp_pseudo_doc.text))[0]

        assert r._validate_match(match, regexp_pseudo_doc)

    def test_validate_match_pre(self, regexp_pseudo_doc):

        r = RegexpPseudoAnnotator(
            regexp_pattern="_", tag="_", pre_pseudo=["sinds", "al", "vanaf"]
        )
        pattern = re.compile(r"\d+ jaar")

        match = list(pattern.finditer(regexp_pseudo_doc.text))[0]

        assert r._validate_match(match, regexp_pseudo_doc)

    def test_validate_match_post(self, regexp_pseudo_doc):

        r = RegexpPseudoAnnotator(
            regexp_pattern="_", tag="_", post_pseudo=["geleden", "getrouwd", "gestopt"]
        )
        pattern = re.compile(r"\d+ jaar")

        match = list(pattern.finditer(regexp_pseudo_doc.text))[0]

        assert not r._validate_match(match, regexp_pseudo_doc)

    def test_validate_match_lower(self, regexp_pseudo_doc):

        r = RegexpPseudoAnnotator(
            regexp_pattern="_", tag="_", pre_pseudo=["na"], lowercase=True
        )
        pattern = re.compile(r"\d+ jaar")

        match = list(pattern.finditer(regexp_pseudo_doc.text))[0]

        assert not r._validate_match(match, regexp_pseudo_doc)


class TestBsnAnnotator:
    def test_elfproef(self):
        an = BsnAnnotator(bsn_regexp="(\\D|^)(\\d{9})(\\D|$)", capture_group=2, tag="_")

        assert an._elfproef("111222333")
        assert not an._elfproef("111222334")
        assert an._elfproef("123456782")
        assert not an._elfproef("123456783")

    def test_elfproef_wrong_length(self):
        an = BsnAnnotator(bsn_regexp="(\\D|^)(\\d{9})(\\D|$)", capture_group=2, tag="_")

        with pytest.raises(ValueError):
            an._elfproef("12345678")

    def test_elfproef_non_numeric(self):
        an = BsnAnnotator(bsn_regexp="(\\D|^)(\\d{9})(\\D|$)", capture_group=2, tag="_")

        with pytest.raises(ValueError):
            an._elfproef("test")

    def test_annotate(self, bsn_doc):
        an = BsnAnnotator(bsn_regexp="(\\D|^)(\\d{9})(\\D|$)", capture_group=2, tag="_")
        annotations = an.annotate(bsn_doc)

        expected_annotations = [
            dd.Annotation(text="111222333", start_char=26, end_char=35, tag="_"),
            dd.Annotation(text="123456782", start_char=39, end_char=48, tag="_"),
        ]

        assert annotations == expected_annotations

    def test_annotate_with_nondigits(self, bsn_doc):
        an = BsnAnnotator(bsn_regexp=r"\d{4}\.\d{2}\.\d{3}", tag="_")
        doc = dd.Document("1234.56.782")
        annotations = an.annotate(doc)

        expected_annotations = [
            dd.Annotation(text="1234.56.782", start_char=0, end_char=11, tag="_"),
        ]

        assert annotations == expected_annotations


class TestPhoneNumberAnnotator:
    def test_annotate_defaults(self, phone_number_doc):
        an = PhoneNumberAnnotator(
            phone_regexp=r"(?<!\d)"
            r"(\(?(0031|\+31|0)"
            r"(1[035]|2[0347]|3[03568]|4[03456]|5[0358]|6|7|88|800|91|90[069]|"
            r"[1-5]\d{2})\)?)"
            r" ?-? ?"
            r"((\d{2,4}[ -]?)+\d{2,4})",
            tag="_",
        )
        annotations = an.annotate(phone_number_doc)

        expected_annotations = [
            dd.Annotation(text="0314-555555", start_char=21, end_char=32, tag="_"),
            dd.Annotation(text="088 755 55 55", start_char=35, end_char=48, tag="_"),
            dd.Annotation(text="(06)55555555", start_char=53, end_char=65, tag="_"),
            dd.Annotation(text="0800-9003", start_char=135, end_char=144, tag="_"),
        ]

        assert annotations == expected_annotations

    def test_annotate_short(self, phone_number_doc):
        an = PhoneNumberAnnotator(
            phone_regexp=r"(?<!\d)"
            r"(\(?(0031|\+31|0)"
            r"(1[035]|2[0347]|3[03568]|4[03456]|5[0358]|6|7|88|800|91|90[069]|"
            r"[1-5]\d{2})\)?)"
            r" ?-? ?"
            r"((\d{2,4}[ -]?)+\d{2,4})",
            min_digits=4,
            max_digits=8,
            tag="_",
        )
        annotations = an.annotate(phone_number_doc)

        expected_annotations = [
            dd.Annotation(text="065555", start_char=72, end_char=78, tag="_")
        ]

        assert annotations == expected_annotations

    def test_annotate_long(self, phone_number_doc):
        an = PhoneNumberAnnotator(
            phone_regexp=r"(?<!\d)"
            r"(\(?(0031|\+31|0)"
            r"(1[035]|2[0347]|3[03568]|4[03456]|5[0358]|6|7|88|800|91|90[069]|"
            r"[1-5]\d{2})\)?)"
            r" ?-? ?"
            r"((\d{2,4}[ -]?)+\d{2,4})",
            min_digits=11,
            max_digits=12,
            tag="_",
        )
        annotations = an.annotate(phone_number_doc)

        expected_annotations = [
            dd.Annotation(text="065555555555", start_char=93, end_char=105, tag="_")
        ]

        assert annotations == expected_annotations
