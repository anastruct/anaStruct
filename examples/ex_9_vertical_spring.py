from anastruct.fem.system import SystemElements

ss = SystemElements()
ss.add_element([[0, 0], [1, 0]])
ss.add_element([2, 0])

ss.add_support_hinged(1)
ss.add_support_spring(3, 2, 3000)
ss.point_load(2, Fy=-1000)
ss.solve()
ss.show_structure()
ss.show_bending_moment()
ss.show_displacement()
