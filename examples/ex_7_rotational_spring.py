from anastruct.fem.system import SystemElements

# Test the primary force vector when applying a q_load at a hinged element.

ss = SystemElements()
ss.add_element([[0.0, 0.0], [3.5, 0.0]])
ss.add_element([7, 0], spring={1: 100})
ss.add_support_fixed([1])
ss.add_support_hinged(3)
ss.point_load(2, Fy=-100)

if __name__ == "__main__":
    ss.solve()
    ss.show_displacement()
    ss.show_reaction_force()
    ss.show_bending_moment()
