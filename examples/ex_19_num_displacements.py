from anastruct.fem.system import SystemElements

ss = SystemElements(mesh=10)
ss.add_element([10, 0])
ss.add_element([20, 0])
ss.q_load(-10, [1, 2])
ss.add_support_hinged([1, 2, 3])

if __name__ == "__main__":
    ss.solve()
    for el in ss.element_map.values():
        print(el.deflection.max())
    ss.show_results()
