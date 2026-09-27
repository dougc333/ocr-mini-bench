from block_pipeline import (
    build_parent_geometry,
    normalized_box,
    padded_box,
    validate_layout,
    validate_parent_layout,
)


def geometry():
    return {
        "horizontal_boundaries": [
            {"boundary_id": "Y000", "position_normalized": 0, "source": "page_edge"},
            {"boundary_id": "Y001", "position_normalized": 100, "source": "visible_rule"},
        ],
        "vertical_boundaries": [
            {"boundary_id": "X000", "position_normalized": 0, "source": "page_edge"},
            {
                "boundary_id": "X001",
                "position_normalized": 500,
                "source": "inferred_alignment",
            },
            {"boundary_id": "X002", "position_normalized": 600, "source": "visible_rule"},
            {"boundary_id": "X003", "position_normalized": 1000, "source": "page_edge"},
        ],
    }


def test_validate_layout_canonicalizes_ids_and_order():
    layout = {
        "horizontal_bands": [
            {
                "row_start_boundary_id": "Y000",
                "row_end_boundary_id": "Y001",
                "label": "Header",
                "blocks": [
                    {
                        "col_start_boundary_id": "X001",
                        "col_end_boundary_id": "X003",
                        "label": "B",
                        "kind": "header",
                    },
                    {
                        "col_start_boundary_id": "X000",
                        "col_end_boundary_id": "X001",
                        "label": "A",
                        "kind": "header",
                    },
                ],
            }
        ]
    }
    bands = validate_layout(layout, geometry())
    assert bands[0]["band_id"] == "h01"
    assert [block["block_id"] for block in bands[0]["blocks"]] == ["h01-v01", "h01-v02"]


def test_validate_layout_rejects_overlapping_vertical_blocks():
    layout = {
        "horizontal_bands": [
            {
                "row_start_boundary_id": "Y000",
                "row_end_boundary_id": "Y001",
                "label": "x",
                "blocks": [
                    {
                        "col_start_boundary_id": "X000",
                        "col_end_boundary_id": "X002",
                        "label": "a",
                        "kind": "other",
                    },
                    {
                        "col_start_boundary_id": "X001",
                        "col_end_boundary_id": "X003",
                        "label": "b",
                        "kind": "other",
                    },
                ],
            }
        ]
    }
    try:
        validate_layout(layout, geometry())
    except ValueError as exc:
        assert "Overlapping vertical" in str(exc)
    else:
        raise AssertionError("expected overlap rejection")


def test_validate_layout_rejects_invented_boundary_id():
    layout = {
        "horizontal_bands": [
            {
                "row_start_boundary_id": "Y999",
                "row_end_boundary_id": "Y001",
                "label": "x",
                "blocks": [
                    {
                        "col_start_boundary_id": "X000",
                        "col_end_boundary_id": "X003",
                        "label": "x",
                        "kind": "other",
                    }
                ],
            }
        ]
    }
    try:
        validate_layout(layout, geometry())
    except ValueError as exc:
        assert "Unknown boundary ID" in str(exc)
    else:
        raise AssertionError("expected invented boundary rejection")


def test_coordinate_conversion_and_padding():
    assert normalized_box(2000, 1000, 250, 100, 750, 900) == (500, 100, 1500, 900)
    assert padded_box((2, 3, 1998, 997), 2000, 1000, 6) == (0, 0, 2000, 1000)


def test_parent_scoped_boundaries_cannot_cross_regions():
    page_geometry = {
        "page": 1,
        "horizontal_boundaries": [
            {"boundary_id": "Y000", "position_normalized": 0, "source": "page_edge"},
            {"boundary_id": "Y001", "position_normalized": 500, "source": "visible_rule"},
            {"boundary_id": "Y002", "position_normalized": 1000, "source": "page_edge"},
        ],
        "vertical_boundaries": [
            {"boundary_id": "X000", "position_normalized": 0, "source": "page_edge"},
            {"boundary_id": "X001", "position_normalized": 500, "source": "inferred_alignment"},
            {"boundary_id": "X002", "position_normalized": 1000, "source": "page_edge"},
        ],
    }
    parents = build_parent_geometry(page_geometry)
    first, second = parents["parent_regions"]
    invalid = {
        "parent_regions": [
            {
                "parent_region_id": first["parent_region_id"],
                "cells": [
                    {
                        "row_start_boundary_id": first["horizontal_boundaries"][0]["boundary_id"],
                        "row_end_boundary_id": second["horizontal_boundaries"][-1]["boundary_id"],
                        "col_start_boundary_id": first["vertical_boundaries"][0]["boundary_id"],
                        "col_end_boundary_id": first["vertical_boundaries"][-1]["boundary_id"],
                        "merged": True,
                        "label": "invalid",
                        "kind": "other",
                    }
                ],
            },
            {
                "parent_region_id": second["parent_region_id"],
                "cells": [
                    {
                        "row_start_boundary_id": second["horizontal_boundaries"][0]["boundary_id"],
                        "row_end_boundary_id": second["horizontal_boundaries"][-1]["boundary_id"],
                        "col_start_boundary_id": second["vertical_boundaries"][0]["boundary_id"],
                        "col_end_boundary_id": second["vertical_boundaries"][-1]["boundary_id"],
                        "merged": True,
                        "label": "valid",
                        "kind": "other",
                    }
                ],
            },
        ]
    }
    try:
        validate_parent_layout(invalid, parents)
    except ValueError as exc:
        assert "does not belong to parent" in str(exc)
    else:
        raise AssertionError("expected cross-parent boundary rejection")
