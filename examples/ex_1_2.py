from anastruct.fem import system as se


def run() -> None:
    """Run example 1.2"""
    system = se.SystemElements()
    system.add_element(location=[[3, 4], [0, 0]], EA=5e9, EI=8000)
    system.add_element(location=[[8, 4], [3, 4]], EA=5e9, EI=4000)
    system.show_structure()

    system.q_load(element_id=2, q=-10)

    system.add_support_hinged(node_id=2)
    system.add_support_fixed(node_id=3)

    system.solve()
    system.show_structure()
    system.show_reaction_force()
    system.show_axial_force()
    system.show_shear_force()
    system.show_bending_moment()
    system.show_displacement()


if __name__ == "__main__":
    run()
