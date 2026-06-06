import unittest

import numpy as np

from anastruct.fem import system as se


class SimpleTest(unittest.TestCase):
    show = False

    def test_example_1(self):
        system = se.SystemElements()
        system.add_element(location=[[0, 0], [3, 4]], EA=5e9, EI=8000)
        system.add_element(location=[[3, 4], [8, 4]], EA=5e9, EI=4000)
        system.q_load(element_id=2, q=-10)
        system.add_support_hinged(node_id=1)
        system.add_support_fixed(node_id=3)
        system.solve()
        system.show_structure(show=self.show)
        system.show_bending_moment(show=self.show)
        system.show_axial_force(show=self.show)
        system.show_reaction_force(show=self.show)
        system.show_shear_force(show=self.show)

    def test_example_2(self):
        system = se.SystemElements()
        system.add_truss_element(location=[[0, 0], [0, 5]], EA=5000)
        system.add_truss_element(location=[[0, 5], [5, 5]], EA=5000)
        system.add_truss_element(location=[[5, 5], [5, 0]], EA=5000)
        system.add_truss_element(location=[[0, 0], [5, 5]], EA=5000 * np.sqrt(2))
        system.add_support_hinged(node_id=1)
        system.add_support_hinged(node_id=4)
        system.point_load(Fx=10, node_id=2)
        system.solve()
        system.show_structure(show=self.show)
        system.show_bending_moment(show=self.show)
        system.show_axial_force(show=self.show)
        system.show_reaction_force(show=self.show)
        system.show_shear_force(show=self.show)

    def test_example_3(self):
        system = se.SystemElements()
        system.add_element(location=[[0, 0], [0, 5]], EA=15000, EI=5000)
        system.add_element(location=[[0, 5], [5, 5]], EA=15000, EI=5000)
        system.add_element(location=[[5, 5], [5, 0]], EA=15000, EI=5000)
        system.add_support_fixed(node_id=1)
        system.add_support_spring(node_id=4, translation=3, k=4000)
        system.point_load(Fx=30, node_id=2)
        system.q_load(q=-10, element_id=2)
        system.solve()
        system.show_structure(show=self.show)
        system.show_bending_moment(show=self.show)
        system.show_axial_force(show=self.show)
        system.show_reaction_force(show=self.show)
        system.show_shear_force(show=self.show)

    def test_example_4(self):
        system = se.SystemElements()
        system.add_element(location=[[0, 0], [5, 0]], EA=5e9, EI=8000, spring={2: 0})
        system.add_element(location=[[5, 0], [5, 5]], EA=5e9, EI=4000)
        system.moment_load(Tz=10, node_id=3)
        system.add_support_hinged(node_id=1)
        system.add_support_hinged(node_id=3)
        system.show_structure(show=self.show)
        system.solve()
        system.show_bending_moment(show=self.show)
        system.show_axial_force(show=self.show)
        system.show_reaction_force(show=self.show)
        system.show_shear_force(show=self.show)

    def test_example_5(self):
        system = se.SystemElements()
        system.add_element(location=[[0, 0], [5, 0]], EA=5e9, EI=8000)
        system.add_element(location=[[5, 0], [5, 5]], EA=5e9, EI=4000)
        system.moment_load(Tz=10, node_id=3)
        system.add_support_hinged(node_id=1)
        system.add_support_hinged(node_id=3)
        system.solve()
        system.show_structure(show=self.show)
        system.show_bending_moment(show=self.show)
        system.show_axial_force(show=self.show)
        system.show_reaction_force(show=self.show)
        system.show_shear_force(show=self.show)

    def test_example_6(self):
        system = se.SystemElements()
        system.add_element(location=[[0, 0], [5, 0]], steelsection="IPE 300")
        system.add_element(location=[[5, 0], [5, 5]], steelsection="IPE 300")
        system.moment_load(Tz=10, node_id=3)
        system.add_support_hinged(node_id=1)
        system.add_support_hinged(node_id=3)
        # Structure plot colors
        system.change_plot_colors({"support": "b"})
        system.change_plot_colors({"element": "b"})
        system.change_plot_colors({"element_text": "b"})
        system.change_plot_colors({"annotation": "b"})
        system.change_plot_colors({"node_text": "b"})
        system.change_plot_colors({"point_load_fill": "k"})
        system.change_plot_colors({"point_load_edge": "b"})
        system.change_plot_colors({"point_load_text": "b"})
        system.change_plot_colors({"q_load": "k"})
        system.change_plot_colors({"q_load_arrow_edge": "g"})
        system.change_plot_colors({"q_load_arrow_face": "g"})
        system.change_plot_colors({"q_load_text": "k"})
        system.change_plot_colors({"moment_load": "b"})
        system.change_plot_colors({"moment_load_text": "b"})
        system.show_structure(show=self.show, annotations=True)
        system.solve()
        # Result plot colors
        system.change_plot_colors({"reaction_force_arrow_edge": "g"})
        system.change_plot_colors({"reaction_force_arrow_fill": "g"})
        system.change_plot_colors({"displaced_elem": "g"})
        system.change_plot_colors({"axial_force_pos": "g"})
        system.change_plot_colors({"axial_force_sign": "g"})
        system.change_plot_colors({"reaction_force_arrow_edge": "g"})
        system.change_plot_colors({"reaction_force_arrow_fill": "g"})
        system.change_plot_colors({"reaction_force_text": "b"})
        system.change_plot_colors({"bending_moment": "g"})
        system.change_plot_colors({"shear_force": "g"})
        system.show_displacement(show=self.show)
        system.show_bending_moment(show=self.show)
        system.show_axial_force(show=self.show)
        system.show_reaction_force(show=self.show)
        system.show_shear_force(show=self.show)


if __name__ == "__main__":
    unittest.main()
