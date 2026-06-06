from anastruct.fem.system import SystemElements

ss = SystemElements()
ss.add_element([[0, 0], [2, 0]])
ss.add_element([4, 0])
ss.add_element([6, 0])
ss.add_support_fixed([1, 4])

ss.moment_load([2, 3], [20, -20])

if __name__ == "__main__":
    ss.solve()
    ss.show_bending_moment()
