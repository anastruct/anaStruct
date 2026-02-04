"""Tests for beam generator functionality.

Tests cover:
- Unit tests for each beam type (geometry, supports, connectivity)
- Integration tests (load application and solve)
- Factory function
- Validation method
- Edge cases and error handling
"""

import warnings

import numpy as np
from pytest import approx, raises

from anastruct.preprocess.beam import (
    CantileverBeam,
    FourSpanBeam,
    LeftCantileverBeam,
    LeftProppedBeam,
    MultiSpanBeam,
    ProppedBeam,
    RightCantileverBeam,
    RightProppedBeam,
    SimpleBeam,
    ThreeSpanBeam,
    TwoSpanBeam,
    create_beam,
)
from anastruct.vertex import Vertex


def describe_simple_beam():
    def it_creates_valid_geometry():
        beam = SimpleBeam(length=10)

        assert beam.type == "Simple Beam"
        assert beam.length == 10
        assert len(beam.nodes) == 2
        assert beam.validate()

    def it_has_correct_supports():
        beam = SimpleBeam(length=10)

        assert beam.support_definitions[1] == "pinned"
        assert beam.support_definitions[2] == "roller"
        assert len(beam.support_definitions) == 2

    def it_has_one_span():
        beam = SimpleBeam(length=10)

        assert len(beam.node_ids) == 1
        assert beam.node_ids[0] == [1, 2]
        assert len(beam.element_ids) == 1
        assert len(beam.element_ids[0]) == 1

    def it_has_correct_node_positions():
        beam = SimpleBeam(length=10)

        assert beam.nodes[0].x == approx(0.0)
        assert beam.nodes[0].y == approx(0.0)
        assert beam.nodes[1].x == approx(10.0)
        assert beam.nodes[1].y == approx(0.0)

    def it_creates_system_elements():
        beam = SimpleBeam(length=10)

        assert len(beam.system.element_map) == 1

    def it_validates_positive_length():
        with raises(ValueError, match="must be positive"):
            SimpleBeam(length=-5)

        with raises(ValueError, match="must be positive"):
            SimpleBeam(length=0)


def describe_cantilever_beam():
    def it_creates_valid_geometry():
        beam = CantileverBeam(length=5)

        assert beam.type == "Cantilever Beam"
        assert beam.length == 5
        assert len(beam.nodes) == 2
        assert beam.validate()

    def it_defaults_to_right_cantilever():
        beam = CantileverBeam(length=5)

        # Default cantilever_side="right" means free end on the right, fixed on left
        assert beam.cantilever_side == "right"
        assert beam.support_definitions[1] == "fixed"
        assert len(beam.support_definitions) == 1

    def it_supports_left_cantilever():
        beam = CantileverBeam(length=5, cantilever_side="left")

        # Free end on the left, fixed on right
        assert beam.cantilever_side == "left"
        assert beam.support_definitions[2] == "fixed"
        assert len(beam.support_definitions) == 1

    def it_validates_cantilever_side():
        with raises(ValueError, match="cantilever_side"):
            CantileverBeam(length=5, cantilever_side="middle")


def describe_right_cantilever_beam():
    def it_creates_valid_geometry():
        beam = RightCantileverBeam(length=5)

        assert beam.type == "Right Cantilever Beam"
        assert beam.cantilever_side == "right"
        assert beam.validate()

    def it_has_fixed_support_on_left():
        beam = RightCantileverBeam(length=5)

        assert beam.support_definitions[1] == "fixed"
        assert 2 not in beam.support_definitions


def describe_left_cantilever_beam():
    def it_creates_valid_geometry():
        beam = LeftCantileverBeam(length=5)

        assert beam.type == "Left Cantilever Beam"
        assert beam.cantilever_side == "left"
        assert beam.validate()

    def it_has_fixed_support_on_right():
        beam = LeftCantileverBeam(length=5)

        assert beam.support_definitions[2] == "fixed"
        assert 1 not in beam.support_definitions


def describe_multi_span_beam():
    def it_creates_with_num_spans():
        beam = MultiSpanBeam(length=30, num_spans=3)

        assert beam.type == "Multi-Span Beam"
        assert beam.length == approx(30)
        assert len(beam.nodes) == 4
        assert len(beam.node_ids) == 3
        assert beam.validate()

    def it_creates_with_span_lengths():
        beam = MultiSpanBeam(span_lengths=[4, 6, 5])

        assert beam.length == approx(15)
        assert len(beam.nodes) == 4
        assert len(beam.node_ids) == 3

    def it_has_supports_at_all_interior_and_end_nodes():
        beam = MultiSpanBeam(length=30, num_spans=3)

        # Node 1 = pinned, nodes 2, 3, 4 = roller
        assert beam.support_definitions[1] == "pinned"
        assert beam.support_definitions[2] == "roller"
        assert beam.support_definitions[3] == "roller"
        assert beam.support_definitions[4] == "roller"
        assert len(beam.support_definitions) == 4

    def it_supports_left_cantilever():
        beam = MultiSpanBeam(span_lengths=[3, 5, 5], cantilevers="left")

        # Left cantilever: first span is unsupported on the left
        assert 1 not in beam.support_definitions
        assert beam.support_definitions[2] == "pinned"
        assert beam.support_definitions[3] == "roller"
        assert beam.support_definitions[4] == "roller"

    def it_supports_right_cantilever():
        beam = MultiSpanBeam(span_lengths=[5, 5, 3], cantilevers="right")

        # Right cantilever: last span is unsupported on the right
        assert beam.support_definitions[1] == "pinned"
        assert beam.support_definitions[2] == "roller"
        assert 4 not in beam.support_definitions

    def it_supports_both_cantilevers():
        beam = MultiSpanBeam(span_lengths=[3, 5, 5, 3], cantilevers="both")

        # Both cantilevers: first and last spans unsupported at ends
        assert 1 not in beam.support_definitions
        assert beam.support_definitions[2] == "pinned"
        assert beam.support_definitions[3] == "roller"
        assert 5 not in beam.support_definitions

    def it_has_one_element_per_span():
        beam = MultiSpanBeam(length=30, num_spans=3)

        for span in range(3):
            assert len(beam.element_ids[span]) == 1

    def it_validates_input_combinations():
        with raises(ValueError, match="Either num_spans or span_lengths"):
            MultiSpanBeam()

        with raises(ValueError, match="Only one of"):
            MultiSpanBeam(span_lengths=[5, 5], num_spans=2)

        with raises(ValueError, match="length must also be provided"):
            MultiSpanBeam(num_spans=3)

    def it_validates_cantilevers_parameter():
        with raises(ValueError, match="cantilevers must be"):
            MultiSpanBeam(length=20, num_spans=2, cantilevers="top")

    def it_creates_equal_spans_from_num_spans():
        beam = MultiSpanBeam(length=30, num_spans=3)

        assert beam.span_lengths == [approx(10), approx(10), approx(10)]


def describe_two_span_beam():
    def it_creates_valid_geometry():
        beam = TwoSpanBeam(length=20)

        assert beam.type == "Two-Span Beam"
        assert beam.length == approx(20)
        assert len(beam.nodes) == 3
        assert len(beam.node_ids) == 2
        assert beam.validate()

    def it_has_correct_supports():
        beam = TwoSpanBeam(length=20)

        assert beam.support_definitions[1] == "pinned"
        assert beam.support_definitions[2] == "roller"
        assert beam.support_definitions[3] == "roller"


def describe_three_span_beam():
    def it_creates_valid_geometry():
        beam = ThreeSpanBeam(length=30)

        assert beam.type == "Three-Span Beam"
        assert beam.length == approx(30)
        assert len(beam.nodes) == 4
        assert len(beam.node_ids) == 3
        assert beam.validate()

    def it_has_support_at_every_node():
        beam = ThreeSpanBeam(length=30)

        assert len(beam.support_definitions) == 4
        assert beam.support_definitions[1] == "pinned"
        for i in range(2, 5):
            assert beam.support_definitions[i] == "roller"


def describe_four_span_beam():
    def it_creates_valid_geometry():
        beam = FourSpanBeam(length=40)

        assert beam.type == "Four-Span Beam"
        assert beam.length == approx(40)
        assert len(beam.nodes) == 5
        assert len(beam.node_ids) == 4
        assert beam.validate()

    def it_has_support_at_every_node():
        beam = FourSpanBeam(length=40)

        assert len(beam.support_definitions) == 5


def describe_propped_beam():
    def it_creates_right_propped():
        beam = ProppedBeam(
            interior_length=8, cantilever_length=3, cantilever_side="right"
        )

        assert beam.type == "Propped Beam"
        assert beam.length == approx(11)
        assert beam.cantilever_side == "right"
        assert beam.validate()

    def it_creates_left_propped():
        beam = ProppedBeam(
            interior_length=8, cantilever_length=3, cantilever_side="left"
        )

        assert beam.length == approx(11)
        assert beam.cantilever_side == "left"
        assert beam.validate()

    def it_has_supports_on_interior_span_only():
        beam = ProppedBeam(
            interior_length=8, cantilever_length=3, cantilever_side="right"
        )

        # Right cantilever: support at nodes 1 and 2, not at node 3
        assert beam.support_definitions[1] == "pinned"
        assert beam.support_definitions[2] == "roller"
        assert 3 not in beam.support_definitions

    def it_validates_cantilever_side():
        with raises(ValueError, match="cantilever_side"):
            ProppedBeam(interior_length=8, cantilever_length=3, cantilever_side="up")


def describe_right_propped_beam():
    def it_creates_valid_geometry():
        beam = RightProppedBeam(interior_length=8, cantilever_length=3)

        assert beam.type == "Right Propped Beam"
        assert beam.cantilever_side == "right"
        assert beam.validate()

    def it_has_no_support_at_right_end():
        beam = RightProppedBeam(interior_length=8, cantilever_length=3)

        last_node = len(beam.nodes)
        assert last_node not in beam.support_definitions


def describe_left_propped_beam():
    def it_creates_valid_geometry():
        beam = LeftProppedBeam(interior_length=8, cantilever_length=3)

        assert beam.type == "Left Propped Beam"
        assert beam.cantilever_side == "left"
        assert beam.validate()

    def it_has_no_support_at_left_end():
        beam = LeftProppedBeam(interior_length=8, cantilever_length=3)

        assert 1 not in beam.support_definitions


def describe_factory_function():
    """Tests for create_beam factory function."""

    def it_creates_beam_by_name():
        beam = create_beam("simple", length=10)

        assert isinstance(beam, SimpleBeam)
        assert beam.type == "Simple Beam"

    def it_handles_case_insensitive_names():
        beams = [
            create_beam("simple", length=10),
            create_beam("SIMPLE", length=10),
            create_beam("Simple", length=10),
        ]

        for beam in beams:
            assert isinstance(beam, SimpleBeam)

    def it_handles_different_name_separators():
        beam_underscore = create_beam("two_span", length=20)
        beam_hyphen = create_beam("two-span", length=20)
        beam_space = create_beam("two span", length=20)

        assert isinstance(beam_underscore, TwoSpanBeam)
        assert isinstance(beam_hyphen, TwoSpanBeam)
        assert isinstance(beam_space, TwoSpanBeam)

    def it_creates_all_beam_types():
        beams = {
            "simple": {"length": 10},
            "cantilever": {"length": 5},
            "right_cantilever": {"length": 5},
            "left_cantilever": {"length": 5},
            "multi_span": {"length": 20, "num_spans": 2},
            "two_span": {"length": 20},
            "three_span": {"length": 30},
            "four_span": {"length": 40},
            "propped": {"interior_length": 8, "cantilever_length": 3},
            "right_propped": {"interior_length": 8, "cantilever_length": 3},
            "left_propped": {"interior_length": 8, "cantilever_length": 3},
        }

        for name, kwargs in beams.items():
            beam = create_beam(name, **kwargs)
            assert beam.validate()

    def it_raises_error_for_invalid_type():
        with raises(ValueError, match="Unknown beam type"):
            create_beam("nonexistent", length=10)

    def it_provides_helpful_error_with_available_types():
        with raises(ValueError, match="Available types"):
            create_beam("invalid_type", length=10)


def describe_validate_method():
    def it_validates_correct_geometry():
        beam = SimpleBeam(length=10)
        assert beam.validate()

    def it_catches_invalid_node_ids():
        beam = SimpleBeam(length=10)

        # Inject an invalid node ID
        beam.node_ids[0] = [1, 99]

        with raises(ValueError, match="invalid node ID"):
            beam.validate()

    def it_catches_duplicate_nodes():
        beam = SimpleBeam(length=10)

        # Make both nodes at the same position
        beam.nodes[1] = Vertex(0.0, 0.0)

        with raises(ValueError, match="Duplicate nodes"):
            beam.validate()


def describe_integration_tests():
    """Integration tests with load application."""

    def describe_distributed_loads():
        def it_applies_q_load_to_simple_beam():
            beam = SimpleBeam(length=10)
            beam.apply_q_load_to_spans(q=-5, direction="y")

            # Verify load was applied to the element
            assert beam.system.loads_q is not None
            assert len(beam.system.loads_q) > 0

        def it_applies_q_load_to_specific_span():
            beam = TwoSpanBeam(length=20)
            # Only load span 0
            beam.apply_q_load_to_spans(q=-10, direction="y", spans=0)

            # Only the element in span 0 should have a load
            loaded_element_id = beam.element_ids[0][0]
            assert loaded_element_id in beam.system.loads_q

        def it_applies_q_load_to_multiple_spans():
            beam = ThreeSpanBeam(length=30)
            beam.apply_q_load_to_spans(q=-5, direction="y", spans=[0, 2])

            # Elements in spans 0 and 2 should be loaded
            for span in [0, 2]:
                el_id = beam.element_ids[span][0]
                assert el_id in beam.system.loads_q

        def it_applies_q_load_to_all_spans_by_default():
            beam = ThreeSpanBeam(length=30)
            beam.apply_q_load_to_spans(q=-5, direction="y")

            # All elements should be loaded
            for span in range(3):
                el_id = beam.element_ids[span][0]
                assert el_id in beam.system.loads_q

    def describe_point_loads():
        def it_applies_point_load_at_existing_node():
            beam = SimpleBeam(length=10)
            # Load at start node (absolute_location=0 is at node 1)
            beam.apply_point_load_to_spans(Fy=-100, absolute_location=0.0)

            assert len(beam.system.loads_point) > 0

        def it_inserts_node_for_point_load():
            beam = SimpleBeam(length=10)
            original_elements = len(beam.system.element_map)

            beam.apply_point_load_to_spans(Fy=-100, absolute_location=3.0)

            # Should have split the element, creating 2 elements from 1
            assert len(beam.system.element_map) == original_elements + 1

        def it_updates_internal_ids_after_node_insertion():
            beam = SimpleBeam(length=10)

            beam.apply_point_load_to_spans(Fy=-100, absolute_location=3.0)

            # node_ids for span 0 should now have 3 nodes
            assert len(beam.node_ids[0]) == 3
            # element_ids for span 0 should now have 2 elements
            assert len(beam.element_ids[0]) == 2

        def it_validates_location_arguments():
            beam = SimpleBeam(length=10)

            with raises(ValueError, match="Either absolute_location or relative_location"):
                beam.apply_point_load_to_spans(Fy=-100)

            with raises(ValueError, match="Only one of"):
                beam.apply_point_load_to_spans(
                    Fy=-100, absolute_location=5.0, relative_location=0.5
                )

        def it_applies_load_at_relative_location():
            beam = SimpleBeam(length=10)

            beam.apply_point_load_to_spans(Fy=-100, relative_location=0.5)

            # Should have inserted a node at midpoint and applied load
            assert len(beam.node_ids[0]) == 3
            assert len(beam.system.loads_point) > 0

    def describe_cantilever_loads():
        def it_applies_tip_load_to_cantilever():
            beam = RightCantileverBeam(length=5)
            beam.apply_point_load_to_spans(Fy=-100, relative_location=1.0)

            # Load should be applied at the tip node
            assert len(beam.system.loads_point) > 0

        def it_applies_distributed_load_to_cantilever():
            beam = LeftCantileverBeam(length=8)
            beam.apply_q_load_to_spans(q=-10, direction="y")

            assert len(beam.system.loads_q) > 0

    def describe_solve_integration_tests():
        """Solve-based integration tests to verify reactions."""

        def it_solves_simple_beam_with_udl():
            # Simple beam with UDL: q=-5, direction="y" on length=10
            # Reactions at each end = |q|*L/2 = 5*10/2 = 25 (upward)
            beam = SimpleBeam(length=10)
            beam.apply_q_load_to_spans(q=-5, direction="y")
            beam.system.solve()

            # Node IDs are 1-based: left node=1, right node=2
            left_reaction = beam.system.get_node_results_system(node_id=1)["Fy"]
            right_reaction = beam.system.get_node_results_system(node_id=2)["Fy"]

            # get_node_results_system returns reaction in system sign convention
            assert abs(left_reaction) == approx(25, abs=1e-6)
            assert abs(right_reaction) == approx(25, abs=1e-6)

        def it_solves_two_span_beam_with_udl():
            # Two-span beam with UDL: verify solve works for indeterminate structure
            beam = TwoSpanBeam(length=20)
            beam.apply_q_load_to_spans(q=-10, direction="y")
            beam.system.solve()

            # Total load = 10 * 20 = 200, sum of reactions must equal 200
            total_reaction = sum(
                abs(beam.system.get_node_results_system(node_id=i)["Fy"])
                for i in range(1, 4)
            )
            assert total_reaction == approx(200, abs=1e-3)

        def it_solves_right_cantilever_with_tip_load():
            # RightCantileverBeam length=5, Fy=-100 at tip
            # Fixed end reaction magnitude = 100
            beam = RightCantileverBeam(length=5)
            beam.apply_point_load_to_spans(Fy=-100, relative_location=1.0)
            beam.system.solve()

            # Fixed support is at left node (node_id=1)
            fixed_reaction = beam.system.get_node_results_system(node_id=1)["Fy"]

            assert abs(fixed_reaction) == approx(100, abs=1e-6)

        def it_solves_left_cantilever_with_udl():
            # LeftCantileverBeam length=8, q=-10
            # Fixed end reaction magnitude = 80
            beam = LeftCantileverBeam(length=8)
            beam.apply_q_load_to_spans(q=-10, direction="y")
            beam.system.solve()

            # Fixed support is at right node (node_id=2)
            fixed_reaction = beam.system.get_node_results_system(node_id=2)["Fy"]

            assert abs(fixed_reaction) == approx(80, abs=1e-6)


def describe_span_element_ids():
    """Tests for get_element_ids_of_spans."""

    def it_gets_all_elements_when_none():
        beam = ThreeSpanBeam(length=30)

        all_ids = beam.get_element_ids_of_spans(spans=None)
        assert len(all_ids) == 3

    def it_gets_single_span():
        beam = ThreeSpanBeam(length=30)

        span_ids = beam.get_element_ids_of_spans(spans=1)
        assert len(span_ids) == 1

    def it_gets_multiple_spans():
        beam = ThreeSpanBeam(length=30)

        span_ids = beam.get_element_ids_of_spans(spans=[0, 2])
        assert len(span_ids) == 2

    def it_raises_for_invalid_span():
        beam = ThreeSpanBeam(length=30)

        with raises(KeyError, match="span number"):
            beam.get_element_ids_of_spans(spans=99)


def describe_angled_beams():
    """Tests for beams at non-zero angles."""

    def it_creates_angled_simple_beam():
        beam = SimpleBeam(length=10, angle=45)

        expected_x = 10 * np.cos(np.radians(45))
        expected_y = 10 * np.sin(np.radians(45))
        assert beam.nodes[1].x == approx(expected_x)
        assert beam.nodes[1].y == approx(expected_y)
        assert beam.validate()

    def it_creates_vertical_beam():
        beam = SimpleBeam(length=10, angle=90)

        assert beam.nodes[1].x == approx(0.0, abs=1e-10)
        assert beam.nodes[1].y == approx(10.0)

    def it_warns_for_radian_like_angle():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SimpleBeam(length=10, angle=1.5)

            assert len(w) == 1
            assert "degrees, not radians" in str(w[0].message)

    def it_does_not_warn_for_zero_angle():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SimpleBeam(length=10, angle=0)

            assert len(w) == 0

    def it_does_not_warn_for_normal_angle():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SimpleBeam(length=10, angle=45)

            assert len(w) == 0

    def it_normalizes_negative_angle():
        beam = SimpleBeam(length=10, angle=-90)
        assert beam.angle == approx(270)


def describe_edge_cases():
    def it_creates_multiple_independent_instances():
        beam1 = SimpleBeam(length=10)
        beam2 = SimpleBeam(length=20)

        # Instances should be independent
        assert beam1.length != beam2.length
        assert len(beam1.nodes) == len(beam2.nodes)
        assert beam1.nodes[1].x != beam2.nodes[1].x

    def it_handles_very_short_beam():
        beam = SimpleBeam(length=0.001)
        assert beam.validate()

    def it_handles_very_long_beam():
        beam = SimpleBeam(length=1e6)
        assert beam.validate()

    def it_handles_unequal_span_lengths():
        beam = MultiSpanBeam(span_lengths=[3, 7, 5, 2])

        assert beam.length == approx(17)
        assert len(beam.nodes) == 5
        assert beam.validate()

    def it_handles_custom_section():
        section = {"EI": 2e6, "EA": 3e8, "g": 5.0}
        beam = SimpleBeam(length=10, section=section)

        assert beam.section == section
