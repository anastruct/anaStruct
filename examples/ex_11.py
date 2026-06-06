from anastruct.fem.system import SystemElements

# Test dead loads on the structure. TO DO! Distributed axial force


ss = SystemElements()
ss.add_element([[0, 6], [5, 4]], g=5)
ss.add_element([7, 3.2], g=5)
ss.add_element([[0, 0], [5, 4]], spring={2: 0})
ss.add_support_hinged([1, 4])

if __name__ == "__main__":
    ss.solve()
    ss.show_axial_force()
