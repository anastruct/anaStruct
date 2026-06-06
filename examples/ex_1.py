from anastruct.fem.system import SystemElements

ss = SystemElements()
ss.add_element(location=[[0, 0], [3, 4]], EA=5e9, EI=8000)
ss.add_element(location=[[3, 4], [8, 4]], EA=5e9, EI=4000)

ss.q_load(element_id=2, q=-10)

ss.add_support_hinged(node_id=1)
ss.add_support_fixed(node_id=3)

ss.solve()
ss.show_structure()
ss.show_reaction_force()
ss.show_axial_force()
ss.show_shear_force()
ss.show_bending_moment()
ss.show_displacement()
