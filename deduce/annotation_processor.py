"""Contains components for processing AnnotationSet."""
from abc import abstractmethod

import docdeid as dd
from docdeid import AnnotationSet, Annotation
from frozendict import frozendict


class DeduceMergeAdjacentAnnotations(dd.process.MergeAdjacentAnnotations):
    """
    Merges adjacent tags, according to Deduce logic:

    - adjacent date or ID annotations are not merged;
    - adjacent annotations with mixed patient/person tags are replaced
      with the "persoon" annotation;
    - adjacent annotations with patient tags are replaced with the "patient" annotation.
    """

    def _tags_match(self, left_tag: str, right_tag: str) -> bool:
        """
        Define whether two tags match. This is the case when they are equal strings, and
        additionally patient and person tags are also regarded as equal.

        Args:
            left_tag: The left tag.
            right_tag: The right tag.

        Returns:
            ``True`` if tags match, ``False`` otherwise.
        """

        patient_part = [tag.endswith("_patient") for tag in (left_tag, right_tag)]
        return left_tag not in ("datum", "id") and (
            left_tag == right_tag
            or all(patient_part)
            # XXX Why only this way, why not enable also "persoon" as the left tag
            #  and patient_part as the right one? Note that this would affect also
            #  the implementation of the next method.
            or (patient_part[0] and right_tag == "persoon")
        )

    def _adjacent_annotations_replacement(
        self,
        left_annotation: dd.Annotation,
        right_annotation: dd.Annotation,
        text: str,
    ) -> dd.Annotation:
        """
        Replace two annotations that have equal tags with a new annotation.

        If one of the two annotations has the "patient" tag (and the other is either
        "patient" or "persoon"), the other annotation will be used. In other cases, the
        tags are always equal.
        """

        ltag = left_annotation.tag
        rtag = right_annotation.tag
        replacement_tag = (
            ltag if ltag == rtag else
            "persoon" if rtag == "persoon" else
            "patient"
        )

        return dd.Annotation(
            text=text[left_annotation.start_char : right_annotation.end_char],
            start_char=left_annotation.start_char,
            end_char=right_annotation.end_char,
            tag=replacement_tag,
        )


class PersonAnnotationConverter(dd.process.AnnotationProcessor):
    """
    Responsible for processing the annotations produced by all name annotators (regular
    and context-based).

    Any overlap with annotations that contain "pseudo" in their tag is removed, as are
    those annotations. Then resolves overlap between remaining annotations, and maps the
    tags to either "patient" or "persoon", based on whether "patient" is in all
    constituent tags (e.g. voornaam_patient+achternaam_patient => patient,
    achternaam_onbekend => persoon).
    """

    def __init__(self) -> None:
        def map_tag_to_prio(tag: str) -> (int, int, int):
            """
            Maps from the tag of a mention to its priority. The lower, the higher
            priority.

            The return value is a tuple of:
              1. Is this a pseudo tag? If it is, it's a priority.
              2. How many subtags does the tag have? The more, the higher priority.
              3. Is this a patient tag? If it is, it's a priority.
            """
            is_pseudo = "pseudo" in tag
            num_subtags = tag.count("+") + 1
            is_patient = (tag.count("patient") == num_subtags and
                          'achternaam_patient' in tag)
            return (-int(is_pseudo), -num_subtags, -int(is_patient))

        self._overlap_resolver = dd.process.OverlapResolver(
            sort_by=("tag", "length"),
            sort_by_callbacks=frozendict(
                tag=map_tag_to_prio,
                length=lambda x: -x,
            ),
        )

    def process_annotations(
        self, annotations: dd.AnnotationSet, text: str
    ) -> dd.AnnotationSet:
        new_annotations = self._overlap_resolver.process_annotations(
            annotations, text=text
        )

        real_annos = (
            anno
            for anno in new_annotations
            if "pseudo" not in anno.tag and anno.text.strip()
        )
        with_patient = (
            dd.Annotation(
                text=anno.text,
                start_char=anno.start_char,
                end_char=anno.end_char,
                tag=PersonAnnotationConverter._resolve_tag(anno.tag),
                priority=anno.priority,
            )
            for anno in real_annos
        )
        return dd.AnnotationSet(with_patient)

    @classmethod
    def _resolve_tag(cls, tag: str) -> str:
        if "+" not in tag:
            return tag if "patient" in tag else "persoon"
        return (
            "patient"
            if all("patient" in part for part in tag.split("+"))
            else "persoon"
        )


class FilterAnnotations(dd.process.AnnotationProcessor):
    """Filters annotation by a predicate."""

    @abstractmethod
    def should_keep(self, anno: Annotation) -> bool:
        """Determines whether the `anno` annotation should be retained."""

    def process_annotations(
            self, annotations: AnnotationSet, text: str
    ) -> AnnotationSet:
        return AnnotationSet(filter(self.should_keep, annotations))


class RemoveAnnotations(FilterAnnotations):
    """Removes all annotations with corresponding tags."""

    def __init__(self, tags: list[str]) -> None:
        self.tags = tags

    def should_keep(self, anno: Annotation) -> bool:
        return anno.tag not in self.tags


class RemoveSingleInitial(FilterAnnotations):
    """\
    Removes all annotations of initials that contain only a single capital letter.

    This annotator should be downstream of `DeduceMergeAdjacentAnnotations`,
    otherwise it might remove *all* annotations of people's initials.
    """

    def should_keep(self, anno: Annotation) -> bool:
        if 'initiaal' in anno.tag:
            # Try to take 2 uppercase letters from the mention.
            uppers = filter(str.isupper, anno.text)
            try:
                next(uppers)
                next(uppers)
            except StopIteration:
                return False
        return True


class RemoveAllCapsPersons(FilterAnnotations):
    """\
    Removes single-word annotations of persons that contain only capital letter.
    """

    def should_keep(self, anno: Annotation) -> bool:
        return anno.tag != 'persoon' or ' ' in anno.text or not anno.text.isupper()


class CleanAnnotationTag(dd.process.AnnotationProcessor):
    """Renames tags using a mapping."""

    def __init__(self, tag_map: dict[str, str]) -> None:
        self.tag_map = tag_map

    def process_annotations(
        self, annotations: AnnotationSet, text: str
    ) -> AnnotationSet:
        new_annotations = AnnotationSet()

        for annotation in annotations:
            if annotation.tag in self.tag_map:
                new_annotations.add(
                    dd.Annotation(
                        start_char=annotation.start_char,
                        end_char=annotation.end_char,
                        text=annotation.text,
                        start_token=annotation.start_token,
                        end_token=annotation.end_token,
                        tag=self.tag_map[annotation.tag],
                        priority=annotation.priority,
                    )
                )
            else:
                new_annotations.add(annotation)

        return new_annotations
