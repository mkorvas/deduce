import docdeid as dd

from deduce.redactor import DeduceRedactor, DateStrategy


class TestDeduceRedactor:
    def test_redact_patient(self):
        proc = DeduceRedactor()
        text = "Jan Jansen"

        annotations = dd.AnnotationSet(
            [
                dd.Annotation(text="Jan", start_char=0, end_char=3, tag="patient"),
                dd.Annotation(text="Jansen", start_char=4, end_char=10, tag="patient"),
            ]
        )

        expected_text = "[PATIENT] [PATIENT]"

        assert proc.redact(text, annotations) == expected_text

    def test_redact_mixed(self):
        proc = DeduceRedactor()
        text = "Jan Jansen, wonende in Rotterdam"

        annotations = dd.AnnotationSet(
            [
                dd.Annotation(
                    text="Jan Jansen", start_char=0, end_char=10, tag="patient"
                ),
                dd.Annotation(
                    text="Rotterdam", start_char=23, end_char=32, tag="woonplaats"
                ),
            ]
        )

        expected_text = "[PATIENT], wonende in [WOONPLAATS-1]"

        assert proc.redact(text, annotations) == expected_text

    def test_redact_count_multiple(self):
        proc = DeduceRedactor()
        text = "Jan Jansen, wonende in Rotterdam, verhuisd vanuit Groningen"

        annotations = dd.AnnotationSet(
            [
                dd.Annotation(
                    text="Rotterdam", start_char=23, end_char=32, tag="woonplaats"
                ),
                dd.Annotation(
                    text="Groningen", start_char=50, end_char=59, tag="woonplaats"
                ),
            ]
        )

        expected_text = (
            "Jan Jansen, wonende in [WOONPLAATS-1], verhuisd vanuit [WOONPLAATS-2]"
        )

        assert proc.redact(text, annotations) == expected_text

    def test_redact_count_multiple_fuzzy(self):
        proc = DeduceRedactor()
        text = "Jan Jansen, wonende in Ommen, verhuisd vanuit Emmen"

        annotations = dd.AnnotationSet(
            [
                dd.Annotation(
                    text="Ommen", start_char=23, end_char=28, tag="woonplaats"
                ),
                dd.Annotation(
                    text="Emmen", start_char=46, end_char=51, tag="woonplaats"
                ),
            ]
        )

        expected_text = (
            "Jan Jansen, wonende in [WOONPLAATS-1], verhuisd vanuit [WOONPLAATS-1]"
        )

        assert proc.redact(text, annotations) == expected_text

    def test_shift_date(self):
        proc = DeduceRedactor(date_strategy=DateStrategy("shift", 7, None))
        text = "2024-07-29 t/m 28. dec 24"

        annotations = dd.AnnotationSet(
            [
                dd.Annotation(text="2024-07-29", start_char=0, end_char=10,
                              tag="datum"),
                dd.Annotation(text="28. dec 24", start_char=15, end_char=25,
                              tag="datum"),
            ]
        )

        expected_text = "2024-08-05 t/m 4. jan 25"

        assert proc.redact(text, annotations) == expected_text
