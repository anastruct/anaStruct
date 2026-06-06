from anastruct.fem.system import SystemElements

# tests parallel direction of the q-load and dead load.

ss = SystemElements()

ss.add_element([-2, 2])
ss.add_element([0, 4])
ss.add_element([-2, 6])
ss.add_element([0, 8])

ss.add_support_hinged(1)
ss.add_support_hinged(5)

ss.q_load(1, (1, 2, 3, 4), "x")

if __name__ == "__main__":
    ss.solve()
    print([a["Fx"] for a in ss.get_node_results_system() if isinstance(a, dict)])
    ss.show_results()
