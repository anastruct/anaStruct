from anastruct.fem.system import SystemElements

# Geting section form sectionbase

ss = SystemElements()

ss.add_element([[0, 0], [10, 0]], steelsection="IPE 300", E=210e9, sw=True, orient="z")
ss.add_element([[0, 1], [10, 1]], b=1.0, h=1.0, E=210e9, gamma=20)
ss.add_element([[0, 2], [10, 2]], d=0.6, E=210e9)

if __name__ == "__main__":
    ss.show_structure(verbosity=0, annotations=True)
