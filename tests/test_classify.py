from __future__ import annotations

from audiobook.classify import classify_chapter
from audiobook.types import SegmentType
from tests.helpers import make_block, make_chapter, make_page


def _classify(blocks):
    return classify_chapter(make_chapter([make_page(1, blocks)]))


def test_prose_heading_and_code_are_distinguished():
    items = _classify(
        [
            make_block("The Sysadmin Approach", font="MyriadPro-SemiboldCond", size=18.9),
            make_block("Body text lives here.", y0=150),
            make_block(
                "def restart(service):\n    return service.start()",
                y0=200,
                font="UbuntuMono-Regular",
                size=8.5,
            ),
            make_block("zone", y0=260, font="UbuntuMono-Regular", size=8.5),
        ]
    )
    types = [(item.type, item.reason) for item in items]
    assert types[0] == (SegmentType.PROSE, "heading")
    assert types[1] == (SegmentType.PROSE, "")
    assert types[2] == (SegmentType.CODE, "")
    assert types[3] == (SegmentType.PROSE, "")


def test_furniture_is_skipped_with_reasons():
    items = _classify(
        [
            make_block("CHAPTER 1\nIntroduction", y0=76, size=25.2, font="MyriadPro-SemiboldCond"),
            make_block("4 | Chapter 1: Introduction", y0=612, size=8.5),
            make_block("www.it-ebooks.info", y0=640, size=8.0),
            make_block("1 A footnote about things.", y0=560, size=8.0),
            make_block("Real body text.", y0=300),
        ]
    )
    reasons = [item.reason for item in items if item.skip]
    assert reasons == ["chapter-opener", "footer", "watermark", "footnote"]
    assert not items[-1].skip


def test_figure_keeps_its_caption_and_gets_the_label():
    items = _classify(
        [
            make_block("", y0=200, height=100, image=True),
            make_block(
                "Figure 1-1. How a load balancer routes requests.",
                y0=310,
                size=9.5,
                font="MinionPro-It",
            ),
        ]
    )
    assert [item.type for item in items] == [
        SegmentType.FIGURE,
        SegmentType.CAPTION,
    ]
    assert items[0].label == "Figure 1-1"
    assert items[1].caption_body == "How a load balancer routes requests."
    assert items[1].label == "Figure 1-1"


def test_vector_figure_without_image_is_synthesized():
    items = _classify(
        [
            make_block(
                "Figure 2-3. A diagram drawn with vector paths.",
                y0=200,
                size=9.5,
                font="MinionPro-It",
            )
        ]
    )
    assert [item.type for item in items] == [
        SegmentType.FIGURE,
        SegmentType.CAPTION,
    ]
    assert items[0].label == "Figure 2-3"
    assert not items[0].blocks


def test_figure_parts_are_not_announced_separately():
    items = _classify(
        [
            make_block("", y0=200, height=60, image=True),
            make_block("", y0=265, height=60, image=True),
            make_block(
                "Figure 3-1. Two panels of one figure.",
                y0=340,
                size=9.5,
                font="MinionPro-It",
            ),
        ]
    )
    figures = [item for item in items if item.type is SegmentType.FIGURE]
    assert len(figures) == 2
    assert figures[0].skip and figures[0].reason == "figure-part"
    assert not figures[1].skip
    assert figures[1].label == "Figure 3-1"


def test_table_caption_collects_rows_until_prose():
    items = _classify(
        [
            make_block("Table 4-2. Machines and roles.", y0=200, size=9),
            make_block("Machine | Role", y0=220, size=9, font="MyriadPro-Cond"),
            make_block("web-1 | frontend", y0=235, size=9, font="MyriadPro-Cond"),
            make_block("Back to prose.", y0=280, size=10.5),
        ]
    )
    table = next(item for item in items if item.type is SegmentType.TABLE)
    caption = next(item for item in items if item.type is SegmentType.CAPTION)
    assert table.label == "Table 4-2"
    assert len(table.blocks) == 2
    assert caption.caption_body == "Machines and roles."
    assert items[-1].type is SegmentType.PROSE
