import matplotlib.pyplot as plt
import numpy as np

from anastruct import LoadCase, LoadCombination
from anastruct.fem.system import SystemElements

ss = SystemElements()

height = 10

x = np.cumsum([0, 4, 7, 7, 4])
y = np.zeros(x.shape)
x = np.append(x, x[::-1])
y = np.append(y, y + height)

ss.add_element_grid(x, y)
ss.add_element([[0, 0], [0, height]])
ss.add_element([[4, 0], [4, height]])
ss.add_element([[11, 0], [11, height]])
ss.add_element([[18, 0], [18, height]])
ss.add_support_hinged([1, 5])
ss.show_structure()

lc_wind = LoadCase("wind")
lc_wind.q_load(q=-1, element_id=[10, 11, 12, 13, 5])

print(lc_wind)

ss.apply_load_case(lc_wind)
ss.show_structure()

# reset the structure
ss.remove_loads()

# create another load case
lc_cables = LoadCase("cables")
lc_cables.point_load(node_id=[2, 3, 4], Fy=-100)

combination = LoadCombination("ULS")
combination.add_load_case(lc_wind, 1.5)
combination.add_load_case(lc_cables, factor=1.2)

results = combination.solve(ss)

for k, ss in results.items():
    ss.show_structure()
    ss.show_displacement(show=False)
    plt.title(k)
    plt.show()  # type: ignore
