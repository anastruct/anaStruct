from anastruct.fem.system import SystemElements

ss = SystemElements(EA=15000, EI=5000)

# Add beams to the system.
ss.add_element(location=[[0, 0], [0, 5]])
ss.add_element(location=[[0, 5], [5, 5]])
ss.add_element(location=[[5, 5], [5, 0]])

# Add a fixed support at node 1.
ss.add_support_fixed(node_id=1)

# Add a rotational spring support at node 4.
ss.add_support_spring(node_id=4, translation=3, k=4000)

# Add loads.
ss.point_load(Fx=30, node_id=2)
ss.q_load(q=-10, element_id=2)

# Solve
ss.solve()

# Get visual results.
ss.show_structure()
ss.show_reaction_force()
ss.show_axial_force()
ss.show_shear_force()
ss.show_bending_moment()
ss.show_displacement()
