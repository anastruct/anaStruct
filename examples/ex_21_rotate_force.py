from anastruct.fem.system import SystemElements

ss = SystemElements()

ss.add_element([0, 10])
ss.add_element([5, 10])
ss.add_element([5, 0])

ss.add_support_hinged(1)
ss.add_support_hinged(4)

ss.point_load(2, Fy=-10, rotation=30)

if __name__ == "__main__":
    ss.show_structure()
