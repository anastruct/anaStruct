from anastruct import SystemElements

ss = SystemElements(EA=356000, EI=1330)
ss.add_element(location=[[0, 0], [10, 0]])
ss.add_element(location=[[10, 0], [20, 0]])
ss.add_support_hinged(node_id=1)
ss.add_support_hinged(node_id=3)
ss.point_load(Fy=-500, node_id=2)
ss.solve()
ss.show_displacement()
