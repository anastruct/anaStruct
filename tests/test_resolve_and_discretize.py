import pytest

from anastruct import SystemElements


def simply_supported_beam(n_elements: int = 4) -> SystemElements:
    ss = SystemElements(EI=1810, EA=460_000)
    ss.add_multiple_elements([[0, 0], [6, 0]], n_elements)
    ss.add_support_hinged(1)
    ss.add_support_roll(ss.id_last_node)
    return ss


def test_resolve_after_changing_loads_matches_a_fresh_model():
    # The first solve gives zero transverse displacements; they must stay free.
    reused = simply_supported_beam()
    reused.point_load(reused.id_last_node, Fx=-10)
    reused.solve()
    reused.remove_loads()
    reused.point_load(3, Fy=-10)
    reused.solve()

    fresh = simply_supported_beam()
    fresh.point_load(3, Fy=-10)
    fresh.solve()

    assert reused.get_node_displacements(3)["uy"] == pytest.approx(
        fresh.get_node_displacements(3)["uy"]
    )


@pytest.mark.parametrize(
    "location, q, direction",
    [
        ([[0, 0], [6, 0]], -5, "element"),
        ([[0, 0], [4, 3]], [-2, -8], "y"),
        ([[0, 0], [4, 3]], [3, 1], "x"),
    ],
)
def test_discretize_keeps_the_distributed_load(location, q, direction):
    def build() -> SystemElements:
        ss = SystemElements()
        ss.add_element(location)
        ss.add_support_hinged(1)
        ss.add_support_hinged(2)
        ss.q_load(q=q, element_id=1, direction=direction)
        return ss

    original = build()
    original.solve()
    n = 10
    discretized = build()
    discretized.discretize(n)
    discretized.solve()

    for node, node_discretized in ((1, 1), (2, n + 1)):
        for force in ("Fx", "Fy"):
            assert discretized.get_node_results_system(node_discretized)[
                force
            ] == pytest.approx(original.get_node_results_system(node)[force], abs=1e-6)
