from anastruct.fem.system import SystemElements

ss = SystemElements()
ss.add_element(location=[[0, 0], [5, 0]], EA=5e9, EI=8000)
ss.add_element(location=[[5, 0], [5, 5]], EA=5e9, EI=4000)
ss.moment_load(Tz=10, node_id=3)
ss.add_support_hinged(node_id=1)
ss.add_support_hinged(node_id=3)

ss.solve()
ss.show_structure()
ss.show_reaction_force()
ss.show_axial_force()
ss.show_shear_force()
ss.show_bending_moment()
ss.show_displacement()
