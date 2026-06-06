from anastruct.fem.system import SystemElements

ss = SystemElements()

ss.add_element([10, 0])
ss.add_support_hinged(1)
ss.add_support_hinged(2)
ss.q_load([-1, -3], 1)
ss.insert_node(1, factor=0.3)
ss.point_load(3, Fy=-10)
ss.insert_node(3, [5, 5])

if __name__ == "__main__":
    ss.show_structure(values_only=True)  # type: ignore
