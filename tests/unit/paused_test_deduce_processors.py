""" TODO: only organization tests below, needs to move to regression tests. """

from typing import Optional

import docdeid as dd

from deduce.deduce import Deduce
from deduce.lookup_sets import get_lookup_sets
from deduce.tokenizer import DeduceTokenizer

config = Deduce()._initialize_config()
lookup_sets = get_lookup_sets()
tokenizer = DeduceTokenizer()

deduce_processors = Deduce._initialize_annotators(
    config["annotators"].copy(), lookup_sets, tokenizer
)


def get_annotator(name: str, group: Optional[str] = None) -> dd.process.Annotator:
    if group is not None:
        return deduce_processors[group][name]

    return deduce_processors[name]


def annotate_text(
    text: str, annotators: list[dd.process.Annotator]
) -> dd.AnnotationSet:
    doc = dd.Document(text, tokenizers={"default": tokenizer})

    for annotator in annotators:
        annotator.process(doc)

    return doc.annotations


class TestLookupAnnotators:
    def test_annotate_institution(self):
        print("config=", config)

        text = "Reinaerde, Universitair Medisch Centrum Utrecht, UMCU, Diakonessenhuis"
        annotator = get_annotator("institution", group="institutions")

        expected_annotations = {
            dd.Annotation(
                text="Universitair Medisch Centrum Utrecht",
                start_char=11,
                end_char=47,
                tag=annotator.tag,
            ),
            dd.Annotation(
                text="Diakonessenhuis", start_char=55, end_char=70, tag=annotator.tag
            ),
            dd.Annotation(
                text="Centrum", start_char=32, end_char=39, tag=annotator.tag
            ),
            dd.Annotation(text="UMCU", start_char=49, end_char=53, tag=annotator.tag),
            dd.Annotation(
                text="Reinaerde", start_char=0, end_char=9, tag=annotator.tag
            ),
        }

        annotations = annotate_text(text, [annotator])

        assert annotations == expected_annotations


class TestRegexpAnnotators:
    def test_annotate_altrecht_regexp(self):
        text = "Altrecht Bipolair, altrecht Jong, Altrecht psychose"
        annotator = get_annotator("altrecht", group="institutions")
        expected_annotations = {
            dd.Annotation(
                text="Altrecht Bipolair", start_char=0, end_char=17, tag=annotator.tag
            ),
            dd.Annotation(
                text="altrecht Jong", start_char=19, end_char=32, tag=annotator.tag
            ),
            dd.Annotation(
                text="Altrecht", start_char=34, end_char=42, tag=annotator.tag
            ),
        }

        annotations = annotate_text(text, [annotator])

        assert annotations == expected_annotations