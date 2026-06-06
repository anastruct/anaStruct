from anastruct.fem.system import SystemElements

# tests parallel direction of the q-load and dead load.

ss = SystemElements()
a = 1
ss.add_element([2, 2], g=a)
ss.add_element([4, 0], g=a)
ss.add_element([6, 2], g=a)
ss.add_element([8, 0], g=a)

ss.add_support_hinged(1)
ss.add_support_hinged(5)

if __name__ == "__main__":
    ss.solve()
    print([a["Fx"] for a in ss.get_node_results_system() if isinstance(a, dict)])
    ss.show_results()
