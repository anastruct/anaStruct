import math
from anastruct.fem.system import SystemElements

ss = SystemElements(EA=5000)
ss.add_truss_element(location=[[0, 0], [0, 5]])
ss.add_truss_element(location=[[0, 5], [5, 5]])
ss.add_truss_element(location=[[5, 5], [5, 0]])
ss.add_truss_element(location=[[0, 0], [5, 5]], EA=5000 * math.sqrt(2))

ss.add_support_hinged(node_id=1)
ss.add_support_hinged(node_id=4)

ss.point_load(Fx=10, node_id=2)

ss.solve()
ss.show_structure()
ss.show_reaction_force()
ss.show_axial_force()
ss.show_shear_force()
ss.show_displacement(factor=10)
