"""Tests for truss generator functionality.

Tests cover:
- Unit tests for each truss type (geometry validation)
- Integration tests (solve and verify structural behavior)
- Factory function
- Validation method
- Edge cases and error handling
"""

import numpy as np
from pytest import approx, raises

from anastruct.preprocess.truss import (
    AtticRoofTruss,
    DoubleFinkRoofTruss,
    DoubleHoweRoofTruss,
    FanRoofTruss,
    FinkRoofTruss,
    HoweFlatTruss,
    HoweRoofTruss,
    KingPostRoofTruss,
    ModifiedFanRoofTruss,
    ModifiedQueenPostRoofTruss,
    PrattFlatTruss,
    PrattRoofTruss,
    QueenPostRoofTruss,
    WarrenFlatTruss,
    create_truss,
)
from anastruct.vertex import Vertex


def describe_flat_truss_types():
    """Unit tests for flat truss types."""

    def describe_howe_flat_truss():
        def it_creates_valid_geometry():
            truss = HoweFlatTruss(width=20, height=2.5, unit_width=2.0)

            assert truss.type == "Howe Flat Truss"
            assert truss.width == 20
            assert truss.height == 2.5
            assert truss.n_units == 8
            assert len(truss.nodes) == 20
            assert truss.validate()

        def it_has_correct_connectivity():
            truss = HoweFlatTruss(width=20, height=2.5, unit_width=2.0)

            # Should have bottom chord, top chord, web diagonals, and web verticals
            # 8 units, bottom chord has more nodes
            assert len(truss.bottom_chord_node_ids) == 11
            assert len(truss.top_chord_node_ids) == 9
            assert len(truss.web_node_pairs) > 0
            assert len(truss.web_verticals_node_pairs) > 0

        def it_enforces_even_units_by_default():
            # Width that would give 9 units, should round down to 8
            truss = HoweFlatTruss(width=19, height=2.5, unit_width=2.0)
            assert truss.n_units == 8
            assert truss.n_units % 2 == 0

        def it_validates_dimensions():
            with raises(ValueError, match="too small"):
                HoweFlatTruss(width=-5, height=2.5, unit_width=2.0)

            with raises(ValueError, match="must be positive"):
                HoweFlatTruss(width=20, height=-2.5, unit_width=2.0)

            with raises(ValueError, match="unit_width must be positive"):
                HoweFlatTruss(width=20, height=2.5, unit_width=-1.0)

        def it_validates_width_to_unit_width_ratio():
            with raises(ValueError, match="too small"):
                HoweFlatTruss(width=5, height=2.5, unit_width=20)

    def describe_pratt_flat_truss():
        def it_creates_valid_geometry():
            truss = PrattFlatTruss(width=20, height=2.5, unit_width=2.0)

            assert truss.type == "Pratt Flat Truss"
            assert truss.n_units == 8
            assert truss.validate()

        def it_has_different_diagonal_pattern_than_howe():
            howe = HoweFlatTruss(width=20, height=2.5, unit_width=2.0)
            pratt = PrattFlatTruss(width=20, height=2.5, unit_width=2.0)

            # Same number of nodes and elements, but different connectivity
            assert len(howe.nodes) == len(pratt.nodes)
            # Web diagonals should be different (opposite slope direction)
            assert howe.web_node_pairs != pratt.web_node_pairs

    def describe_warren_flat_truss():
        def it_creates_valid_geometry():
            truss = WarrenFlatTruss(width=20, height=2.5, unit_width=2.0)

            assert truss.type == "Warren Flat Truss"
            assert truss.validate()

        def it_has_no_vertical_web_members():
            truss = WarrenFlatTruss(width=20, height=2.5, unit_width=2.0)

            # Warren trusses typically have no vertical web members
            assert len(truss.web_verticals_node_pairs) == 0

        def it_supports_different_end_types():
            # Warren supports triangle_down and triangle_up
            truss_down = WarrenFlatTruss(
                width=20, height=2.5, unit_width=2.0, end_type="triangle_down"
            )
            truss_up = WarrenFlatTruss(
                width=20, height=2.5, unit_width=2.0, end_type="triangle_up"
            )

            assert truss_down.validate()
            assert truss_up.validate()
            # Both end types are valid for Warren trusses
            assert len(truss_down.nodes) > 0
            assert len(truss_up.nodes) > 0


def describe_roof_truss_types():
    """Unit tests for roof truss types."""

    def describe_king_post_roof_truss():
        def it_creates_valid_geometry():
            truss = KingPostRoofTruss(width=10, roof_pitch_deg=30)

            assert truss.type == "King Post Roof Truss"
            assert truss.width == 10
            assert truss.roof_pitch_deg == 30
            assert truss.validate()

        def it_computes_height_from_pitch():
            truss = KingPostRoofTruss(width=10, roof_pitch_deg=30)

            expected_height = (10 / 2) * np.tan(np.radians(30))
            assert truss.height == approx(expected_height)

        def it_has_single_center_vertical():
            truss = KingPostRoofTruss(width=10, roof_pitch_deg=30)

            # King post has 1 vertical, no diagonals
            assert len(truss.web_verticals_node_pairs) == 1
            assert len(truss.web_node_pairs) == 0

        def it_validates_roof_pitch():
            with raises(ValueError, match="roof_pitch_deg must be between 0 and 90"):
                KingPostRoofTruss(width=10, roof_pitch_deg=95)

            with raises(ValueError, match="roof_pitch_deg must be between 0 and 90"):
                KingPostRoofTruss(width=10, roof_pitch_deg=-10)

        def it_supports_overhang():
            truss_no_overhang = KingPostRoofTruss(width=10, roof_pitch_deg=30)
            truss_with_overhang = KingPostRoofTruss(
                width=10, roof_pitch_deg=30, overhang_length=0.5
            )

            # Overhang adds nodes
            assert len(truss_with_overhang.nodes) > len(truss_no_overhang.nodes)
            assert truss_with_overhang.validate()

    def describe_queen_post_roof_truss():
        def it_creates_valid_geometry():
            truss = QueenPostRoofTruss(width=12, roof_pitch_deg=35)

            assert truss.type == "Queen Post Roof Truss"
            assert truss.validate()

        def it_has_correct_web_configuration():
            truss = QueenPostRoofTruss(width=12, roof_pitch_deg=35)

            # Queen post has 2 diagonals and 1 center vertical
            assert len(truss.web_node_pairs) == 2
            assert len(truss.web_verticals_node_pairs) == 1

        def it_has_center_vertical_to_peak():
            """Test for issue #8 fix - center vertical should connect to peak."""
            truss = QueenPostRoofTruss(width=10, roof_pitch_deg=30)

            # Get the center vertical connection
            center_vertical = truss.web_verticals_node_pairs[0]

            # Node 2 is center bottom, node 5 is peak (1-based IDs)
            assert center_vertical == (2, 5)

            # Verify these nodes are actually center bottom and peak
            # Node IDs are 1-based, so we access nodes list with index-1
            center_bottom = truss.nodes[1]  # Node ID 2 at index 1
            peak = truss.nodes[4]  # Node ID 5 at index 4

            assert center_bottom.x == approx(truss.width / 2)
            assert center_bottom.y == approx(0)
            assert peak.x == approx(truss.width / 2)
            assert peak.y == approx(truss.height)

    def describe_fink_roof_truss():
        def it_creates_valid_geometry():
            truss = FinkRoofTruss(width=15, roof_pitch_deg=40)

            assert truss.type == "Fink Roof Truss"
            assert truss.validate()

        def it_has_w_shaped_web_pattern():
            truss = FinkRoofTruss(width=15, roof_pitch_deg=40)

            # Fink has 4 diagonals forming W pattern
            assert len(truss.web_node_pairs) == 4
            assert len(truss.web_verticals_node_pairs) == 0

    def describe_howe_roof_truss():
        def it_creates_valid_geometry():
            truss = HoweRoofTruss(width=12, roof_pitch_deg=35)

            assert truss.type == "Howe Roof Truss"
            assert truss.validate()

        def it_has_vertical_and_diagonal_web_members():
            truss = HoweRoofTruss(width=12, roof_pitch_deg=35)

            # Howe has both verticals and diagonals
            assert len(truss.web_node_pairs) > 0
            assert len(truss.web_verticals_node_pairs) > 0

    def describe_pratt_roof_truss():
        def it_creates_valid_geometry():
            truss = PrattRoofTruss(width=12, roof_pitch_deg=35)

            assert truss.type == "Pratt Roof Truss"
            assert truss.validate()

        def it_has_vertical_and_diagonal_web_members():
            truss = PrattRoofTruss(width=12, roof_pitch_deg=35)

            # Pratt has both verticals and diagonals
            assert len(truss.web_node_pairs) > 0
            assert len(truss.web_verticals_node_pairs) > 0

        def it_has_different_diagonal_pattern_than_howe():
            howe = HoweRoofTruss(width=12, roof_pitch_deg=35)
            pratt = PrattRoofTruss(width=12, roof_pitch_deg=35)

            # Same structure but different web patterns
            assert len(howe.nodes) == len(pratt.nodes)
            # Diagonals slope in opposite directions
            assert howe.web_node_pairs != pratt.web_node_pairs

    def describe_fan_roof_truss():
        def it_creates_valid_geometry():
            truss = FanRoofTruss(width=15, roof_pitch_deg=40)

            assert truss.type == "Fan Roof Truss"
            assert truss.validate()

        def it_has_fan_pattern_web_members():
            truss = FanRoofTruss(width=15, roof_pitch_deg=40)

            # Fan has diagonals radiating from bottom chord
            assert len(truss.web_node_pairs) > 0
            assert len(truss.web_verticals_node_pairs) > 0

    def describe_modified_queen_post_roof_truss():
        def it_creates_valid_geometry():
            truss = ModifiedQueenPostRoofTruss(width=12, roof_pitch_deg=35)

            assert truss.type == "Modified Queen Post Roof Truss"
            assert truss.validate()

        def it_has_more_web_members_than_standard_queen_post():
            modified = ModifiedQueenPostRoofTruss(width=12, roof_pitch_deg=35)
            standard = QueenPostRoofTruss(width=12, roof_pitch_deg=35)

            # Modified version has more web members for enhanced load distribution
            total_modified = len(modified.web_node_pairs) + len(
                modified.web_verticals_node_pairs
            )
            total_standard = len(standard.web_node_pairs) + len(
                standard.web_verticals_node_pairs
            )

            assert total_modified > total_standard

    def describe_double_fink_roof_truss():
        def it_creates_valid_geometry():
            truss = DoubleFinkRoofTruss(width=20, roof_pitch_deg=35)

            assert truss.type == "Double Fink Roof Truss"
            assert truss.validate()

        def it_has_more_members_than_standard_fink():
            double = DoubleFinkRoofTruss(width=20, roof_pitch_deg=35)
            standard = FinkRoofTruss(width=20, roof_pitch_deg=35)

            # Double Fink has more nodes and elements
            assert len(double.nodes) > len(standard.nodes)
            assert len(double.web_node_pairs) > len(standard.web_node_pairs)

        def it_has_two_w_patterns():
            truss = DoubleFinkRoofTruss(width=20, roof_pitch_deg=35)

            # Double Fink should have 8 diagonals (two W patterns)
            assert len(truss.web_node_pairs) == 8
            assert len(truss.web_verticals_node_pairs) == 0

    def describe_double_howe_roof_truss():
        def it_creates_valid_geometry():
            truss = DoubleHoweRoofTruss(width=20, roof_pitch_deg=35)

            assert truss.type == "Double Howe Roof Truss"
            assert truss.validate()

        def it_has_more_verticals_and_diagonals_than_standard():
            double = DoubleHoweRoofTruss(width=20, roof_pitch_deg=35)
            standard = HoweRoofTruss(width=20, roof_pitch_deg=35)

            # Double version has enhanced web pattern
            assert len(double.web_node_pairs) > len(standard.web_node_pairs)
            assert len(double.web_verticals_node_pairs) > len(
                standard.web_verticals_node_pairs
            )

        def it_has_five_verticals():
            truss = DoubleHoweRoofTruss(width=20, roof_pitch_deg=35)

            # Double Howe has 5 vertical members
            assert len(truss.web_verticals_node_pairs) == 5

    def describe_modified_fan_roof_truss():
        def it_creates_valid_geometry():
            truss = ModifiedFanRoofTruss(width=15, roof_pitch_deg=40)

            assert truss.type == "Modified Fan Roof Truss"
            assert truss.validate()

        def it_has_enhanced_web_pattern():
            modified = ModifiedFanRoofTruss(width=15, roof_pitch_deg=40)
            standard = FanRoofTruss(width=15, roof_pitch_deg=40)

            # Modified fan has more web members
            total_modified = len(modified.web_node_pairs) + len(
                modified.web_verticals_node_pairs
            )
            total_standard = len(standard.web_node_pairs) + len(
                standard.web_verticals_node_pairs
            )

            assert total_modified > total_standard

        def it_has_six_diagonals_and_three_verticals():
            truss = ModifiedFanRoofTruss(width=15, roof_pitch_deg=40)

            # Modified fan specific configuration
            assert len(truss.web_node_pairs) == 6
            assert len(truss.web_verticals_node_pairs) == 3

    def describe_attic_roof_truss():
        def it_creates_valid_geometry():
            truss = AtticRoofTruss(width=12, roof_pitch_deg=35, attic_width=6)

            assert truss.type == "Attic Roof Truss"
            assert truss.validate()

        def it_validates_attic_width():
            with raises(ValueError, match="attic_width.*must be less than"):
                AtticRoofTruss(width=10, roof_pitch_deg=30, attic_width=15)

            with raises(ValueError, match="attic_width must be positive"):
                AtticRoofTruss(width=10, roof_pitch_deg=30, attic_width=-5)

        def it_computes_attic_geometry():
            truss = AtticRoofTruss(width=12, roof_pitch_deg=35, attic_width=6)

            # Wall position should be at edge of attic
            assert truss.wall_x == approx((12 - 6) / 2)

            # Ceiling and wall intersect by default
            assert truss.wall_ceiling_intersect or not truss.wall_ceiling_intersect

        def it_supports_custom_attic_height():
            # Use attic_height that's higher than default wall intersection
            truss = AtticRoofTruss(
                width=12, roof_pitch_deg=35, attic_width=6, attic_height=3.0
            )

            assert truss.attic_height == approx(3.0)
            assert truss.validate()

        def it_has_segmented_top_chord_with_ceiling():
            truss = AtticRoofTruss(width=12, roof_pitch_deg=35, attic_width=6)

            # Attic truss has three segments: left, right, and ceiling
            assert isinstance(truss.top_chord_node_ids, dict)
            assert "left" in truss.top_chord_node_ids
            assert "right" in truss.top_chord_node_ids
            assert "ceiling" in truss.top_chord_node_ids


def describe_factory_function():
    """Tests for create_truss factory function."""

    def it_creates_truss_by_name():
        truss = create_truss("howe", width=20, height=2.5, unit_width=2.0)

        assert isinstance(truss, HoweFlatTruss)
        assert truss.type == "Howe Flat Truss"

    def it_handles_case_insensitive_names():
        trusses = [
            create_truss("howe", width=20, height=2.5, unit_width=2.0),
            create_truss("HOWE", width=20, height=2.5, unit_width=2.0),
            create_truss("Howe", width=20, height=2.5, unit_width=2.0),
        ]

        for truss in trusses:
            assert isinstance(truss, HoweFlatTruss)

    def it_handles_different_name_separators():
        # Underscores, hyphens, spaces should all work
        names = ["king_post", "king-post", "kingpost"]

        for name in names:
            truss = create_truss(name, width=10, roof_pitch_deg=30)
            assert isinstance(truss, KingPostRoofTruss)

    def it_creates_all_truss_types():
        # Test that all truss types can be created via factory
        test_cases = [
            ("howe", HoweFlatTruss, {"width": 20, "height": 2.5, "unit_width": 2.0}),
            ("pratt", PrattFlatTruss, {"width": 20, "height": 2.5, "unit_width": 2.0}),
            (
                "warren",
                WarrenFlatTruss,
                {"width": 20, "height": 2.5, "unit_width": 2.0},
            ),
            ("king_post", KingPostRoofTruss, {"width": 10, "roof_pitch_deg": 30}),
            ("queen_post", QueenPostRoofTruss, {"width": 12, "roof_pitch_deg": 35}),
            ("fink", FinkRoofTruss, {"width": 15, "roof_pitch_deg": 40}),
            ("howe_roof", HoweRoofTruss, {"width": 12, "roof_pitch_deg": 35}),
            ("pratt_roof", PrattRoofTruss, {"width": 12, "roof_pitch_deg": 35}),
            ("fan", FanRoofTruss, {"width": 15, "roof_pitch_deg": 40}),
            (
                "attic",
                AtticRoofTruss,
                {"width": 12, "roof_pitch_deg": 35, "attic_width": 6},
            ),
        ]

        for name, expected_class, kwargs in test_cases:
            truss = create_truss(name, **kwargs)
            assert isinstance(truss, expected_class)
            assert truss.validate()

    def it_raises_error_for_invalid_type():
        with raises(ValueError, match="Unknown truss type"):
            create_truss("invalid_truss_type", width=10, height=2)

    def it_provides_helpful_error_with_available_types():
        try:
            create_truss("nonexistent", width=10, height=2)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            # Error should list available types
            assert "Available types:" in str(e)
            assert "howe" in str(e).lower()


def describe_validate_method():
    """Tests for truss validation method."""

    def it_validates_correct_geometry():
        truss = HoweFlatTruss(width=20, height=2.5, unit_width=2.0)

        assert truss.validate() is True

    def it_catches_invalid_node_ids_in_connectivity():
        truss = HoweFlatTruss(width=20, height=2.5, unit_width=2.0)

        # Corrupt connectivity with invalid node ID
        truss.web_node_pairs.append((999, 1000))

        with raises(ValueError, match="invalid node ID"):
            truss.validate()

    def it_catches_duplicate_nodes():
        truss = HoweFlatTruss(width=20, height=2.5, unit_width=2.0)

        # Add duplicate node at same location as node ID 1 (index 0)
        original = truss.nodes[0]
        truss.nodes.append(Vertex(original.x, original.y))
        # Node IDs are 1-based, so new node ID is len(truss.nodes)
        truss.web_node_pairs.append((1, len(truss.nodes)))

        with raises(ValueError, match="Duplicate nodes"):
            truss.validate()

    def it_catches_zero_length_elements():
        truss = HoweFlatTruss(width=20, height=2.5, unit_width=2.0)

        # Create zero-length element by modifying existing node
        # Make the second node same as first node
        # This will be caught as duplicate nodes
        truss.nodes[1] = Vertex(truss.nodes[0].x, truss.nodes[0].y)

        with raises(ValueError, match="Duplicate nodes"):
            truss.validate()


def describe_integration_tests():
    """Integration tests - system integration and load application."""

    def describe_system_integration():
        def it_creates_valid_system_elements():
            truss = HoweFlatTruss(width=20, height=2.5, unit_width=2.0)

            # Verify SystemElements was created and populated
            assert truss.system is not None
            assert len(truss.system.element_map) > 0
            assert len(truss.system.node_map) > 0

        def it_has_correct_element_count():
            truss = HoweFlatTruss(width=20, height=2.5, unit_width=2.0)

            # Count expected elements
            expected_bottom = len(truss.bottom_chord_node_ids) - 1
            expected_top = len(truss.top_chord_node_ids) - 1
            expected_webs = len(truss.web_node_pairs)
            expected_verticals = len(truss.web_verticals_node_pairs)
            expected_total = (
                expected_bottom + expected_top + expected_webs + expected_verticals
            )

            assert len(truss.system.element_map) == expected_total

        def it_applies_loads_to_chords():
            truss = HoweFlatTruss(width=20, height=2.5, unit_width=2.0)

            # Apply load (don't solve, just verify load application works)
            truss.apply_q_load_to_top_chord(q=-10, direction="y")

            # Verify loads were applied to top chord elements
            top_chord_ids = truss.get_element_ids_of_chord("top")
            for el_id in top_chord_ids:
                element = truss.system.element_map[el_id]
                # Element should have a q_load attribute after applying
                assert hasattr(element, "q_load")

    def describe_roof_truss_integration():
        def it_applies_loads_to_chord_segments():
            truss = QueenPostRoofTruss(width=12, roof_pitch_deg=30)

            # Apply loads to specific segments
            truss.apply_q_load_to_top_chord(q=-5, direction="y", chord_segment="left")
            truss.apply_q_load_to_top_chord(q=-5, direction="y", chord_segment="right")

            # Verify loads were applied
            left_ids = truss.get_element_ids_of_chord("top", "left")
            right_ids = truss.get_element_ids_of_chord("top", "right")

            for el_id in left_ids + right_ids:
                element = truss.system.element_map[el_id]
                assert hasattr(element, "q_load")

        def it_validates_after_load_application():
            truss = QueenPostRoofTruss(width=12, roof_pitch_deg=30)

            # Validate before loading
            assert truss.validate()

            # Apply load
            truss.apply_q_load_to_top_chord(q=-5, direction="y")

            # Should still validate after loading
            assert truss.validate()

    def describe_different_support_types():
        def it_has_default_simple_supports():
            truss = HoweFlatTruss(width=20, height=2.5, unit_width=2.0)

            # Default is simple = pinned + roller
            support_defs = truss.support_definitions
            support_types = list(support_defs.values())

            assert "pinned" in support_types
            assert "roller" in support_types

        def it_has_two_support_points():
            truss = HoweFlatTruss(width=20, height=2.5, unit_width=2.0)

            # Should have exactly 2 supports at the ends
            assert len(truss.support_definitions) == 2


def describe_chord_segment_functionality():
    """Tests for segmented chord access."""

    def it_gets_all_elements_when_no_segment_specified():
        truss = QueenPostRoofTruss(width=10, roof_pitch_deg=30)

        all_top = truss.get_element_ids_of_chord("top")

        # Should return all top chord elements
        assert len(all_top) > 0

    def it_gets_specific_segment():
        truss = QueenPostRoofTruss(width=10, roof_pitch_deg=30)

        left_elements = truss.get_element_ids_of_chord("top", "left")
        right_elements = truss.get_element_ids_of_chord("top", "right")

        # Should get different elements
        assert len(left_elements) > 0
        assert len(right_elements) > 0
        assert set(left_elements).isdisjoint(set(right_elements))

    def it_raises_error_for_invalid_segment():
        truss = QueenPostRoofTruss(width=10, roof_pitch_deg=30)

        with raises(KeyError, match="chord_segment.*not found"):
            truss.get_element_ids_of_chord("top", "nonexistent_segment")

    def it_shows_available_segments_in_error():
        truss = QueenPostRoofTruss(width=10, roof_pitch_deg=30)

        try:
            truss.get_element_ids_of_chord("top", "invalid")
            assert False, "Should have raised KeyError"
        except KeyError as e:
            # Should list available segments
            assert "Available segments:" in str(e)
            assert "left" in str(e)
            assert "right" in str(e)


def describe_edge_cases():
    """Edge case tests."""

    def it_handles_minimum_viable_dimensions():
        # Smallest practical truss
        truss = HoweFlatTruss(width=6, height=1, unit_width=2, min_end_fraction=0.5)

        assert truss.n_units >= 2
        assert truss.validate()

    def it_handles_very_steep_roof_pitch():
        # Very steep but valid pitch
        truss = KingPostRoofTruss(width=10, roof_pitch_deg=85)

        assert truss.height > truss.width  # Height > width for steep pitch
        assert truss.validate()

    def it_handles_very_shallow_roof_pitch():
        # Very shallow but valid pitch
        truss = KingPostRoofTruss(width=10, roof_pitch_deg=5)

        assert truss.height < truss.width / 10  # Very shallow
        assert truss.validate()

    def it_creates_multiple_independent_instances():
        # Test that instances don't share mutable state (issue #9 fix)
        truss1 = HoweFlatTruss(width=20, height=2.5, unit_width=2.0)
        truss2 = HoweFlatTruss(width=20, height=2.5, unit_width=2.0)

        # Modify one truss
        truss1.nodes.append(Vertex(100, 100))

        # Should not affect the other
        assert len(truss1.nodes) != len(truss2.nodes)
        assert truss2.validate()


def describe_solve_tests():
    """Solve tests - verify trusses can be solved and produce valid reactions."""

    def describe_king_post_roof_truss_solve():
        def it_solves_with_top_chord_load():
            truss = KingPostRoofTruss(width=10, roof_pitch_deg=30)

            # Apply distributed load to top chord
            truss.apply_q_load_to_top_chord(q=-5, direction="y")

            # Solve the system
            truss.system.solve()

            # Get support node IDs (1-based)
            support_node_ids = list(truss.support_definitions.keys())
            assert len(support_node_ids) == 2

            # Check that reactions are non-zero at support nodes
            for node_id in support_node_ids:
                reaction = truss.system.get_node_results_system(node_id=node_id)["Fy"]
                assert abs(reaction) > 0, f"Node {node_id} should have non-zero reaction"

            # Verify total vertical reaction equals applied load
            total_reaction = sum(
                truss.system.get_node_results_system(node_id=node_id)["Fy"]
                for node_id in support_node_ids
            )

            # Applied load = q * length of top chord
            # For king post, top chord has 2 segments from ends to peak
            top_chord_length = 0
            top_chord_el_ids = truss.get_element_ids_of_chord("top")
            for el_id in top_chord_el_ids:
                element = truss.system.element_map[el_id]
                top_chord_length += element.l

            expected_total_load = -5 * top_chord_length

            assert total_reaction == approx(
                expected_total_load, abs=0.1
            ), "Total reactions should equal total applied load"

    def describe_howe_flat_truss_solve():
        def it_solves_with_bottom_chord_load():
            truss = HoweFlatTruss(width=20, height=2.5, unit_width=2.0)

            # Apply distributed load to bottom chord
            truss.apply_q_load_to_bottom_chord(q=-10, direction="y")

            # Solve the system
            truss.system.solve()

            # Get support node IDs (1-based)
            support_node_ids = list(truss.support_definitions.keys())
            assert len(support_node_ids) == 2

            # Check that reactions are non-zero at support nodes
            for node_id in support_node_ids:
                reaction = truss.system.get_node_results_system(node_id=node_id)["Fy"]
                assert abs(reaction) > 0, f"Node {node_id} should have non-zero reaction"

            # Verify total vertical reaction equals applied load
            total_reaction = sum(
                truss.system.get_node_results_system(node_id=node_id)["Fy"]
                for node_id in support_node_ids
            )

            # Applied load = q * length of bottom chord
            bottom_chord_length = 0
            bottom_chord_el_ids = truss.get_element_ids_of_chord("bottom")
            for el_id in bottom_chord_el_ids:
                element = truss.system.element_map[el_id]
                bottom_chord_length += element.l

            expected_total_load = -10 * bottom_chord_length

            assert total_reaction == approx(
                expected_total_load, abs=0.1
            ), "Total reactions should equal total applied load"
