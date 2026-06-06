from anastruct.fem.system import SystemElements

# Test dead loads on the structure. TO DO! Distributed axial force

ss = SystemElements()
ss.add_element([[0, 0], [10, 5]], g=1)
ss.add_support_hinged(1)
ss.add_support_roll(2, 2)

if __name__ == "__main__":
    ss.solve()
    ss.show_reaction_force()
    ss.show_axial_force()
    ss.show_bending_moment()
