from anastruct.fem.system import SystemElements

ss = SystemElements(mesh=250)
ss.add_element(location=[(0, 0), (1.33, 0)], EA=356, EI=1332)
ss.add_element(location=[(1.33, 0), (2.0, 0)], EA=356, EI=1332)
ss.add_support_hinged(node_id=ss.find_node_id((0, 0)))
ss.add_support_hinged(node_id=ss.find_node_id((1.33, 0)))
ss.q_load(q=-1000.0, element_id=1, direction="y")
ss.q_load(q=-1000.0, element_id=2, direction="y")

if __name__ == "__main__":
    ss.solve()
    ss.show_displacement()
