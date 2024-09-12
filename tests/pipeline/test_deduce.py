from datetime import date

import pytest

import docdeid as dd
from docdeid.utils import annotate_intext
from deduce import Deduce
from deduce.person import Person

text = (
    "betreft: Jan Jansen, bsn 111222333, patnr 000334433. De patient J. Jansen is 64 "
    "jaar oud en woonachtig in Utrecht, IJSWEG 10r. Hij werd op 10 oktober 2018 door "
    "arts Peter de Visser ontslagen van de kliniek van het UMCU. Voor nazorg kan hij "
    "worden bereikt via j.JNSEN.123@gmail.com of (06)12345678. "
    "Thuismedicatie: X-Cure 25000 IE/ml; "
    "Vader, PETER Jansen, 104 jr, woont ook in Utrecht. Met collegiale groeten, "
    "Jan de Visser."
    # FIXME "aan de" is joined to one token (due to "lst_interfix/items.txt"),
    #   preventing "de Quervain ziekte" from matching. Furthermore, when I
    #   managed to get this term censored, the "aan" word was censored, too.
    #   Use a simple whitespace/punctuation-based tokenizer for that annotator
    #   to fix this issue.
    # " De patient lijdt aan de Quervain ziekte."
)


@pytest.fixture
def model(shared_datadir):
    return Deduce(
        save_lookup_structs=False,
        build_lookup_structs=True,
        lookup_data_path=shared_datadir / "lookup",
    )


@pytest.fixture
def strict_model(shared_datadir):
    return Deduce(
        save_lookup_structs=False,
        build_lookup_structs=True,
        lookup_data_path=shared_datadir / "lookup",
        config={
            'annotators': {
                'patient_name': {
                    'args': {
                        'tolerance': 0
                    }
                }
            }
        }
    )


@pytest.fixture
def model_with_doctors(shared_datadir):
    return Deduce(
        save_lookup_structs=False,
        build_lookup_structs=True,
        lookup_data_path=shared_datadir / "lookup",
        config={
            "annotators": {
                "doctor_names": {
                    "annotator_type": "deduce.annotator.DynamicNameAnnotator",
                    "group": "names",
                    "args": {
                        "meta_key": "doctors",
                        "tag": "_",
                        "tolerance": 0
                    }
                },
            }
        }
    )


@pytest.fixture
def model_birth_date(shared_datadir):
    return Deduce(
        save_lookup_structs=False,
        build_lookup_structs=True,
        lookup_data_path=shared_datadir / "lookup",
        config={
            'redactor_date_strategy': 'shift',
            'redactor_date_strategy_init_shift': 1,
            'redactor_date_strategy_include_key': 'birth_date',
        }
    )


# import pydevd_pycharm
# pydevd_pycharm.settrace()


class TestDeduce:
    def test_annotate(self, model):
        metadata = {"patient": Person(first_names=["Jan"], surname="Jansen")}

        doc = model.deidentify(text, metadata=metadata)

        expected_annotations = {
            dd.Annotation(
                text="(06)12345678",
                start_char=284,
                end_char=296,
                tag="telefoonnummer",
            ),
            dd.Annotation(text="111222333", start_char=25, end_char=34, tag="bsn"),
            dd.Annotation(
                text="Peter de Visser", start_char=165, end_char=180, tag="persoon"
            ),
            dd.Annotation(
                text="j.JNSEN.123@gmail.com",
                start_char=259,
                end_char=280,
                tag="emailadres",
            ),
            dd.Annotation(text="J. Jansen", start_char=64, end_char=73, tag="patient"),
            dd.Annotation(text="Jan Jansen", start_char=9, end_char=19, tag="patient"),
            dd.Annotation(
                text="10 oktober 2018", start_char=139, end_char=154, tag="datum"
            ),
            dd.Annotation(text="64", start_char=77, end_char=79, tag="leeftijd"),
            dd.Annotation(text="000334433", start_char=42, end_char=51, tag="id"),
            dd.Annotation(text="Utrecht", start_char=106, end_char=113, tag="locatie"),
            dd.Annotation(
                text="IJSWEG 10r", start_char=115, end_char=125, tag="locatie"
            ),
            dd.Annotation(text="UMCU", start_char=214, end_char=218, tag="ziekenhuis"),
            dd.Annotation(
                text="PETER Jansen", start_char=341, end_char=353, tag="persoon"
            ),
            dd.Annotation(text="104", start_char=355, end_char=358, tag="leeftijd"),
            dd.Annotation(text="Utrecht", start_char=376, end_char=383, tag="locatie"),
            dd.Annotation(
                text="Jan de Visser", start_char=409, end_char=422, tag="persoon"
            ),
        }

        assert set(doc.annotations) == expected_annotations

    def test_deidentify(self, model):
        metadata = {"patient": Person(first_names=["Jan"], surname="Jansen")}
        doc = model.deidentify(text, metadata=metadata)

        expected_deidentified = (
            "betreft: [PATIENT], bsn [BSN-1], patnr [ID-1]. De patient [PATIENT] is "
            "[LEEFTIJD-1] jaar oud en woonachtig in [LOCATIE-1], [LOCATIE-2]. Hij werd "
            "op [DATUM-1] door arts [PERSOON-1] ontslagen van de kliniek van het "
            "[ZIEKENHUIS-1]. Voor nazorg kan hij worden bereikt via [EMAILADRES-1] "
            "of [TELEFOONNUMMER-1]. "
            "Thuismedicatie: X-Cure 25000 IE/ml; "
            "Vader, [PERSOON-2], [LEEFTIJD-2] jr, woont "
            # XXX Btw, if we wanted more perfect security, we should
            #   not give away whether two mentions of age (or street or
            #   anything) were equal before deidentification or not.
            #   Concretely, it shouldn't matter whether LEEFTIJD-1 is the same
            #   as LEEFTIJD-2.
            "ook in [LOCATIE-1]. Met collegiale groeten, [PERSOON-3]."
        )

        assert doc.deidentified_text == expected_deidentified

    def test_mention_idxs(self, model):
        # Given the same inputs as in `test_deidentify`.
        metadata = {"patient": Person(first_names=["Jan"], surname="Jansen")}
        doc = model.deidentify(text, metadata=metadata)

        # The document metadata should now contain information about indices assigned
        # to every mention.
        assert doc.metadata['tagged_mentions']['locatie'] == ['Utrecht', 'IJSWEG 10r']

        # When we set the metadata to reflect annotations identified in (hypothetical)
        # preceding related text,
        modified_metadata = dict(
            metadata,
            tagged_mentions={'locatie': ['Frankrijk', 'Xtrecht', 'Wielingen']})

        # Then, mention indices should be shifted accordingly.
        doc2 = model.deidentify(text, metadata=modified_metadata)
        assert doc2.metadata['tagged_mentions']['locatie'] == [
            'Frankrijk', 'Xtrecht', 'Wielingen', 'IJSWEG 10r'
        ]

        # And, they should also be used in the deidentified text.
        assert 'woonachtig in [LOCATIE-2], [LOCATIE-4]' in doc2.deidentified_text
        assert 'ook in [LOCATIE-2]' in doc2.deidentified_text

        # When we provide custom annotations to redact,
        tagged_mentions = {'kw': ['patient', 'betreft']}

        metadata_with_annos = dict(
            metadata,
            annotations=dd.AnnotationSet([
                dd.Annotation(text='betreft', start_char=0, end_char=7, tag='kw'),
                dd.Annotation(text='bsn', start_char=21, end_char=24, tag='kw')
            ]),
            tagged_mentions=tagged_mentions
        )

        # Then, it should be used by the redactor.
        doc3 = model.deidentify(text,
                                metadata=metadata_with_annos,
                                enabled={'post_processing', 'redactor'})
        assert doc3.deidentified_text.startswith('[KW-2]: Jan Jansen, [KW-3] 111')

        # When we provide empty custom annotations to redact,
        tagged_mentions = {}

        metadata_with_annos_2 = dict(
            metadata,
            annotations=dd.AnnotationSet([
                dd.Annotation(text='betreft', start_char=0, end_char=7, tag='kw'),
                dd.Annotation(text='bsn', start_char=21, end_char=24, tag='kw')
            ]),
            tagged_mentions=tagged_mentions
        )

        # Then, it should still be used by the redactor.
        doc4 = model.deidentify(text,
                                metadata=metadata_with_annos_2,
                                enabled={'post_processing', 'redactor'})
        assert tagged_mentions['kw'] == ['betreft', 'bsn']

        # And, annotating the text without supplying annotations again should work.
        assert annotate_intext(doc4).startswith('<KW>betreft</KW>: Jan')

    def test_annotate_intext(self, model):
        metadata = {"patient": Person(first_names=["Jan"], surname="Jansen")}
        doc = model.deidentify(text, metadata=metadata)

        expected_intext_annotated = (
            "betreft: <PATIENT>Jan Jansen</PATIENT>, bsn <BSN>111222333</BSN>, "
            "patnr <ID>000334433</ID>. De patient <PATIENT>J. Jansen</PATIENT> is "
            "<LEEFTIJD>64</LEEFTIJD> jaar oud en woonachtig in <LOCATIE>Utrecht"
            "</LOCATIE>, <LOCATIE>IJSWEG 10r</LOCATIE>. Hij werd op <DATUM>10 "
            "oktober 2018</DATUM> door arts <PERSOON>Peter de "
            "Visser</PERSOON> ontslagen van de kliniek van het "
            "<ZIEKENHUIS>UMCU</ZIEKENHUIS>. Voor nazorg kan hij worden "
            "bereikt via <EMAILADRES>j.JNSEN.123@gmail.com</EMAILADRES> of "
            "<TELEFOONNUMMER>(06)12345678</TELEFOONNUMMER>. "
            # " De patient lijdt aan de Quervain ziekte."
            "Thuismedicatie: X-Cure 25000 IE/ml; "
            "Vader, <PERSOON>PETER Jansen</PERSOON>, "
            "<LEEFTIJD>104</LEEFTIJD> jr, woont ook in "
            "<LOCATIE>Utrecht</LOCATIE>. Met collegiale groeten, "
            "<PERSOON>Jan de Visser</PERSOON>."
        )

        assert dd.utils.annotate_intext(doc) == expected_intext_annotated

    def test_patient_2(self, model):
        metadata = {"patient": Person(first_names=["Jan"], surname="Jansen")}
        doc = (
            "Lorem ipsum JANSEN sit amet, Peter Jansen adipiscing elit. "
            "Curabitur J. Jansen sapien, J. P. Jansen a vestibulum quis, "
            "facilisis vel J Jansen. Jan de Visser iaculis gravida nulla. "
            "Etiam quis Jan van den Jansen. Integer rutrum, Killaars P."
        )
        want = (
            "Lorem ipsum [PATIENT] sit amet, [PERSOON-1] adipiscing elit. "
            "Curabitur [PATIENT] sapien, [PERSOON-2] a vestibulum quis, "
            "facilisis vel [PATIENT]. [PERSOON-3] iaculis gravida nulla. "
            "Etiam quis [PERSOON-4]. Integer rutrum, [PERSOON-5]."
        )

        deid = model.deidentify(doc, metadata=metadata)
        assert deid.deidentified_text == want

    def test_patient_all_caps(self, model):
        metadata = {"patient": Person(first_names=["PETER", "ARTJOM"],
                                      surname="KATER")}
        doc = "Betreft: Kater Peter Artjom. PETER ARTJOM KATER heeft hoest"
        want = "Betreft: [PATIENT]. [PATIENT] heeft hoest"

        deid = model.deidentify(doc, metadata=metadata)
        assert deid.deidentified_text == want

    def test_patient_accents(self, strict_model):
        metadata = {"patient": Person(first_names=["FRANCOIS",
                                                   "ŠTĚPANIČ",
                                                   "CHLOÉ"],
                                      surname="CROSS")}
        # This works, too, but I found no good way to configure
        # the context annotator to handle a combination of first
        # names and initials, to test both cases at once.
        # doc = "Betreft: François Chloe Cross."
        doc = "Betreft: S. Cross."
        want = "Betreft: [PATIENT]."
        deid = strict_model.deidentify(doc, metadata=metadata)
        assert deid.deidentified_text == want

        # OK, let's just add a second case to the test method.
        doc = "Betreft: François Chloe Cross."
        want = "Betreft: [PATIENT]."
        deid = strict_model.deidentify(doc, metadata=metadata)
        assert deid.deidentified_text == want

    def test_single_initial(self, model):
        metadata = {"patient": Person(first_names=["Jan"], surname="Jansen")}
        doc = "J-katheter plaatsing. Plaatsen J Ch 3/15 links. MCG, XYZ"
        want = doc

        deid = model.deidentify(doc, metadata=metadata)
        assert deid.deidentified_text == want

    def test_single_initial_unicode(self, model, model_with_doctors):
        metadata = {"patient": Person(first_names=["Raf", "Ňaf", "Ỗlaf"])}
        doc = "Patient heet R.Ň.Ỗ. Type ECG: Ň 2000 m/R."
        want = "Patient heet [PATIENT] Type ECG: Ň 2000 m/R."

        deid = model.deidentify(doc, metadata=metadata)
        assert deid.deidentified_text == want

        doc_upper2 = "Patient heet r.Ň.Ỗ. Type ECG: Ň 2000 m/R."
        want_upper2 = "Patient heet r.[PATIENT] Type ECG: Ň 2000 m/R."
        deid2 = model.deidentify(doc_upper2, metadata=metadata)
        assert deid2.deidentified_text == want_upper2

        with_doctor = dict(metadata,
                           doctors=[Person(first_names=["Lisa"],
                                           surname="Mona")])
        # Note the "μL" token -- previously, it would confuse the pipeline so it
        # would consider it an initial of a person, which would get eventually
        # converted to the "PERSOON" tag.
        doc_upper3 = "Patient heet r.ň.Ỗ. Type ECG: L 2000 m/μL."
        want_upper3 = doc_upper3
        deid3 = model_with_doctors.deidentify(doc_upper3, metadata=with_doctor)
        assert deid3.deidentified_text == want_upper3

    def test_street_pattern_1(self, model):
        doc = "Evelien Terlien, woonachtig Veentien 15, 3017 IN Holtien, aangezien."
        want = "[PERSOON-1], woonachtig [LOCATIE-1], [LOCATIE-2], aangezien."

        deid = model.deidentify(doc)
        assert deid.deidentified_text == want

    def test_all_caps_names(self, model):
        metadata = {"patient": Person(first_names=["Peter", "Artjom"],
                                      surname="Kater")}
        doc = "Betreft: PETER ARTJOM KATER. mr. P.A. Kater bereikt KATZ ADL 2"
        want = "Betreft: [PATIENT]. [PATIENT] bereikt KATZ ADL 2"

        deid = model.deidentify(doc, metadata=metadata)
        assert deid.deidentified_text == want

    def test_all_caps_multiple_names(self, model):
        doc = "Betreft: PETER KATZ. mr. Katz bereikt KATZ ADL 2"
        want = "Betreft: [PERSOON-1]. [PERSOON-2] bereikt KATZ ADL 2"

        deid = model.deidentify(doc)
        assert deid.deidentified_text == want

    def test_hospital_name(self, model):
        doc = "therapie Sint Jacob 9/11/2001"
        want = "therapie [ZIEKENHUIS-1] [DATUM-1]"

        deid = model.deidentify(doc)
        assert deid.deidentified_text == want

    def test_hospital_ambig(self, model):
        metadata = {"patient": Person(first_names=["Jan"],
                                      surname="Jacob")}
        doc = "uw pt Jacob werd overgebracht naar st Jacob"
        want = "uw pt [PATIENT] werd overgebracht naar [ZIEKENHUIS-1]"

        deid = model.deidentify(doc, metadata=metadata)
        assert deid.deidentified_text == want

    def test_adjacent_dates(self, model):
        doc = "opname 8/11/2001 10:05:00; therapie 9/11/2001-7/12/2001"
        want = "opname [DATUM-1] 10:05:00; therapie [DATUM-2]-[DATUM-3]"

        deid = model.deidentify(doc)
        assert deid.deidentified_text == want

    def test_birth_date_shifting(self, model_birth_date):
        metadata = {"patient": Person(first_names=["Jan"],
                                      surname="Jacob"),
                    "birth_date": date(year=2022, month=2, day=24)}
        # Yes, the last date present in the text does not play the role of a birth
        # date, but that's not a bug because
        #   1. this program does basically only keyword matching, it doesn't
        #      interpret the context to know this mention does not express the birth
        #      date;
        #   2. the "birth_date" meta key is not interpreted, either, it could as well
        #      be "special_operation_day" and the behaviour should stay the same.
        doc = ("patient Jan Jacob °24/02/2022, opname 25/02/2022 o.w.v. letsels van "
               "2022-02-24")
        want = ("patient [PATIENT] °25/02/2022, opname 25/02/2022 o.w.v. letsels van "
                "2022-02-25")

        deid = model_birth_date.deidentify(doc, metadata=metadata)
        assert deid.deidentified_text == want

    def test_unrecognized_patient(self, model):
        metadata = {"patient": Person(first_names=["Jan", "Jacob"],
                                      surname="Zeppelin")}
        doc = ("Deze patiënt(e)\n\nZeppelin Jan Jacob\n\nNiet in staat is van te "
               "werken.")
        want = ("Deze patiënt(e)\n\n[PATIENT]\n\nNiet in staat is van te werken.")

        deid = model.deidentify(doc, metadata=metadata)
        assert deid.deidentified_text == want