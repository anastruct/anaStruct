from anastruct.fem.system import SystemElements

ss = SystemElements(EI=5e3, EA=1e5)

ss.add_multiple_elements([[0, 0], [0, 10]], 10)

top_node = ss.nearest_node("y", 10) + 1
ss.add_support_roll(top_node, 1)
ss.add_support_hinged(1)

ss.point_load(top_node, Fy=-1)

if __name__ == "__main__":
    print(ss.solve(geometrical_non_linear=True))
    ss.show_displacement()
