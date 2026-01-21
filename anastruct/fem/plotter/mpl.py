import math
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import matplotlib.colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from anastruct.basic import find_nearest, rotate_xy
from anastruct.fem.plotter.values import (
    PlottingValues,
    det_scaling_factor,
    plot_values_axial_force,
    plot_values_bending_moment,
    plot_values_deflection,
    plot_values_element,
    plot_values_shear_force,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from anastruct.fem.node import Node
    from anastruct.fem.system import SystemElements


PATCH_SIZE = 0.03


class Plotter:
    def __init__(self, system: "SystemElements", mesh: int):
        """Class for plotting the structure.

        Args:
            system (SystemElements): System of elements
            mesh (int): Number of points to plot
        """
        self.plot_values = PlottingValues(system, mesh)
        self.mesh: int = self.plot_values.mesh
        self.system: "SystemElements" = system
        self.axes: List["Axes"] = []
        self.one_fig: Optional["Axes"] = None
        self.max_q: float = 0
        self.max_qn: float = 0
        self.max_system_point_load: float = 0
        self.fig: Optional["Figure"] = None
        self.plot_colors: Dict[str, str] = {
            "support": "r",
            "hinge": "k",
            "element": "k",
            "node_number": "k",
            "element_number": "k",
            "displaced_elem": "C0",
            "point_load_fill": "orange",
            "point_load_edge": "b",
            "point_load_text": "k",
            "q_load": "g",
            "q_load_arrow_face": "k",
            "q_load_arrow_edge": "k",
            "q_load_text": "b",
            "moment_load": "orange",
            "moment_load_text": "k",
            "reaction_force_arrow_edge": "orange",
            "reaction_force_arrow_fill": "b",
            "reaction_force_text": "b",
            "axial_force_neg": "C0",
            "axial_force_pos": "C1",
            "axial_force_sign": "b",
            "bending_moment": "C0",
            "shear_force": "C0",
            "annotation": "b",
        }

    def change_plot_colors(self, colors: Dict) -> None:
        """Changes the plotting color for various components of the plot.

        Args:
            colors (Dict): A dictionary containing plot components and colors
            as key-value pairs.
        """
        for item, color in colors.items():
            if not isinstance(color, str) or not isinstance(item, str):
                print("Plot components and colors must be passed in as strings")
            elif self.plot_colors.get(item) is None:
                print(
                    str(item) + " is not a valid plot component to change the color of"
                )
            elif not matplotlib.colors.is_color_like(color):
                print(str(color + "is not a valid matplotlib color"))
            else:
                self.plot_colors[item] = color

    @property
    def max_val_structure(self) -> float:
        """Returns the maximum value of the structure.

        Returns:
            float: Maximum value of the structure
        """
        return self.plot_values.max_val_structure

    def __start_plot(
        self, figsize: Optional[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Starts the plot by initialising a matplotlib plot window of the given size.

        Args:
            figsize (Optional[Tuple[float, float]]): Figure size

        Returns:
            Tuple[float, float]: Figure size (width, height)
        """
        plt.close("all")
        self.fig = plt.figure(figsize=figsize)
        self.axes = [self.fig.add_subplot(111)]
        plt.tight_layout()
        return (self.fig.get_figwidth(), self.fig.get_figheight())

    def __fixed_support_patch(self, max_val: float, axes_i: int = 0) -> None:
        """Plots the fixed supports.

        Args:
            max_val (float): Max scale of the plot
            axes_i (int, optional): Which set of axes to plot on (for multi-plot windows). Defaults to 0.
        """
        width = height = PATCH_SIZE * max_val
        for node in self.system.supports_fixed:
            support_patch = mpatches.Rectangle(
                (node.vertex.x - width * 0.5, node.vertex.y - width * 0.5),
                width,
                height,
                color=self.plot_colors["support"],
                zorder=9,
            )
            self.axes[axes_i].add_patch(support_patch)

    def __hinged_support_patch(self, max_val: float, axes_i: int = 0) -> None:
        """Plots the hinged supports.

        Args:
            max_val (float): Max scale of the plot
            axes_i (int, optional): Which set of axes to plot on (for multi-plot windows). Defaults to 0.
        """
        radius = PATCH_SIZE * max_val
        for node in self.system.supports_hinged:
            support_patch = mpatches.RegularPolygon(
                (node.vertex.x, node.vertex.y - radius),
                numVertices=3,
                radius=radius,
                color=self.plot_colors["support"],
                zorder=9,
            )
            self.axes[axes_i].add_patch(support_patch)

    def __rotational_support_patch(self, max_val: float, axes_i: int = 0) -> None:
        """Plots the rotational supports.

        Args:
            max_val (float): Max scale of the plot
            axes_i (int, optional): Which set of axes to plot on (for multi-plot windows). Defaults to 0.
        """
        width = height = PATCH_SIZE * max_val
        for node in self.system.supports_rotational:
            support_patch = mpatches.Rectangle(
                (node.vertex.x - width * 0.5, node.vertex.y - width * 0.5),
                width,
                height,
                color=self.plot_colors["support"],
                zorder=9,
                fill=False,
            )
            self.axes[axes_i].add_patch(support_patch)

    def __roll_support_patch(self, max_val: float, axes_i: int = 0) -> None:
        """Plots the roller supports.

        Args:
            max_val (float): Max scale of the plot
            axes_i (int, optional): Which set of axes to plot on (for multi-plot windows). Defaults to 0.
        """
        radius = PATCH_SIZE * max_val
        count = 0
        for node in self.system.supports_roll:
            direction = self.system.supports_roll_direction[count]
            rotate = self.system.supports_roll_rotate[count]
            x1 = np.cos(np.pi) * radius + node.vertex.x + radius  # vertex.x
            z1 = np.sin(np.pi) * radius + node.vertex.y  # vertex.y
            x2 = (
                np.cos(np.radians(90)) * radius + node.vertex.x + radius
            )  # vertex.x + radius
            z2 = np.sin(np.radians(90)) * radius + node.vertex.y  # vertex.y + radius
            x3 = (
                np.cos(np.radians(270)) * radius + node.vertex.x + radius
            )  # vertex.x + radius
            z3 = np.sin(np.radians(270)) * radius + node.vertex.y  # vertex.y - radius

            triangle = np.array([[x1, z1], [x2, z2], [x3, z3]])

            if node.id in self.system.inclined_roll:
                angle = self.system.inclined_roll[node.id]
                triangle = rotate_xy(triangle, angle + np.pi * 0.5)
                support_patch_poly = mpatches.Polygon(
                    triangle, color=self.plot_colors["support"], zorder=9
                )
                self.axes[axes_i].add_patch(support_patch_poly)
                self.axes[axes_i].plot(
                    triangle[1:, 0] - 0.5 * radius * np.sin(angle),
                    triangle[1:, 1] - 0.5 * radius * np.cos(angle),
                    color=self.plot_colors["support"],
                )
                if not rotate:
                    rect_patch_regpoly = mpatches.RegularPolygon(
                        (node.vertex.x, radius - node.vertex.y),
                        numVertices=4,
                        radius=radius,
                        orientation=angle,
                        color=self.plot_colors["support"],
                        zorder=9,
                        fill=False,
                    )
                    self.axes[axes_i].add_patch(rect_patch_regpoly)

            elif direction == 2:  # horizontal roll
                support_patch_regpoly = mpatches.RegularPolygon(
                    (node.vertex.x, node.vertex.y - radius),
                    numVertices=3,
                    radius=radius,
                    color=self.plot_colors["support"],
                    zorder=9,
                )
                self.axes[axes_i].add_patch(support_patch_regpoly)
                y = node.vertex.y - 2 * radius
                self.axes[axes_i].plot(
                    [node.vertex.x - radius, node.vertex.x + radius],
                    [y, y],
                    color=self.plot_colors["support"],
                )
                if not rotate:
                    rect_patch_rect = mpatches.Rectangle(
                        (node.vertex.x - radius / 2, node.vertex.y - radius / 2),
                        radius,
                        radius,
                        color=self.plot_colors["support"],
                        zorder=9,
                        fill=False,
                    )
                    self.axes[axes_i].add_patch(rect_patch_rect)
            elif direction == 1:  # vertical roll
                # translate the support to the node

                support_patch_poly = mpatches.Polygon(
                    triangle, color=self.plot_colors["support"], zorder=9
                )
                self.axes[axes_i].add_patch(support_patch_poly)

                y = node.vertex.y - radius
                self.axes[axes_i].plot(
                    [node.vertex.x + radius * 1.5, node.vertex.x + radius * 1.5],
                    [y, y + 2 * radius],
                    color=self.plot_colors["support"],
                )
                if not rotate:
                    rect_patch_rect = mpatches.Rectangle(
                        (node.vertex.x - radius / 2, node.vertex.y - radius / 2),
                        radius,
                        radius,
                        color=self.plot_colors["support"],
                        zorder=9,
                        fill=False,
                    )
                    self.axes[axes_i].add_patch(rect_patch_rect)
            count += 1

    def __rotating_spring_support_patch(self, max_val: float, axes_i: int = 0) -> None:
        """Plots the rotational spring supports.

        Args:
            max_val (float): Max scale of the plot
            axes_i (int, optional): Which set of axes to plot on (for multi-plot windows). Defaults to 0.
        """
        radius = PATCH_SIZE * max_val

        for node, _ in self.system.supports_spring_z:
            r = np.arange(0, radius, 0.001)
            theta = 25 * np.pi * r / (0.2 * max_val)
            x = np.cos(theta) * r + node.vertex.x
            y = np.sin(theta) * r - radius + node.vertex.y
            self.axes[axes_i].plot(x, y, color=self.plot_colors["support"], zorder=9)

            # Triangle
            support_patch = mpatches.RegularPolygon(
                (node.vertex.x, node.vertex.y - radius * 3),
                numVertices=3,
                radius=radius * 0.9,
                color=self.plot_colors["support"],
                zorder=9,
            )
            self.axes[axes_i].add_patch(support_patch)

    def __spring_support_patch(self, max_val: float, axes_i: int = 0) -> None:
        """Plots the linear spring supports.

        Args:
            max_val (float): Max scale of the plot
            axes_i (int, optional): Which set of axes to plot on (for multi-plot windows). Defaults to 0.
        """
        h = PATCH_SIZE * max_val
        left = -0.5 * h
        right = 0.5 * h
        dh = 0.2 * h

        for node, _ in self.system.supports_spring_y:
            yval = np.arange(0, -9, -1) * dh + node.vertex.y
            xval = (
                np.array([0, 0, left, right, left, right, left, 0, 0]) + node.vertex.x
            )

            self.axes[axes_i].plot(
                xval, yval, color=self.plot_colors["support"], zorder=10
            )

            # Triangle
            support_patch = mpatches.RegularPolygon(
                (node.vertex.x, node.vertex.y - h * 2.6),
                numVertices=3,
                radius=h * 0.9,
                color=self.plot_colors["support"],
                zorder=10,
            )
            self.axes[axes_i].add_patch(support_patch)

        for node, _ in self.system.supports_spring_x:
            xval = np.arange(0, 9, 1, dtype=np.float64) * dh + node.vertex.x
            yval = (
                np.array([0, 0, left, right, left, right, left, 0, 0]) + node.vertex.y
            )

            self.axes[axes_i].plot(
                xval, yval, color=self.plot_colors["support"], zorder=10
            )

            # Triangle
            support_patch = mpatches.RegularPolygon(
                (node.vertex.x + h * 1.7, node.vertex.y - h),
                numVertices=3,
                radius=h * 0.9,
                color=self.plot_colors["support"],
                zorder=10,
            )
            self.axes[axes_i].add_patch(support_patch)

    def __internal_hinges_patch(self, max_val: float, axes_i: int = 0) -> None:
        """Plots the internal hinges.

        Args:
            max_val (float): Max scale of the plot
            axes_i (int, optional): Which set of axes to plot on (for multi-plot windows). Defaults to 0.
        """
        radius = PATCH_SIZE * max_val / 2
        for node in self.system.internal_hinges:
            support_patch = mpatches.Circle(
                (node.vertex.x, node.vertex.y),
                radius,
                edgecolor=self.plot_colors["hinge"],
                facecolor="w",
                linewidth=2,
                zorder=9,
            )
            self.axes[axes_i].add_patch(support_patch)

    def __q_load_patch(self, max_val: float, verbosity: int, axes_i: int = 0) -> None:
        """Plots the distributed loads.

        xn1;yn1  q-load   xn1;yn1
        -------------------
        |__________________|
        x1;y1  element    x2;y2

        Args:
            max_val (float): Max scale of the plot
            verbosity (int): 0: show values and arrows, 1: show load block only
            axes_i (int, optional): Which set of axes to plot on (for multi-plot windows). Defaults to 0.
        """

        def __plot_patch(
            h1: float,
            h2: float,
            x1: float,
            y1: float,
            x2: float,
            y2: float,
            ai: float,
            qi: float,
            q: float,
            direction: float,
            el_angle: float,  # pylint: disable=unused-argument
        ) -> None:
            """Plots the distributed load patch.

            Args:
                h1 (float): start height
                h2 (float): end height
                x1 (float): start x coordinate
                y1 (float): start y coordinate
                x2 (float): end x coordinate
                y2 (float): end y coordinate
                ai (float): angle of the element
                qi (float): start load magnitude
                q (float): end load magnitude
                direction (float): 1 or -1, depending on the direction of the load
                el_angle (float): angle of the element
            """
            # - value, because the positive y of the system is opposite of positive y of the plotter
            xn1 = x1 + np.sin(ai) * h1 * direction
            yn1 = y1 + np.cos(ai) * h1 * direction
            xn2 = x2 + np.sin(ai) * h2 * direction
            yn2 = y2 + np.cos(ai) * h2 * direction
            coordinates = ([x1, xn1, xn2, x2], [y1, yn1, yn2, y2])
            self.axes[axes_i].plot(*coordinates, color=self.plot_colors["q_load"])
            rec = mpatches.Polygon(
                np.vstack(coordinates).T, color=self.plot_colors["q_load"], alpha=0.3
            )
            self.axes[axes_i].add_patch(rec)

            if verbosity == 0:
                # arrow
                # pos = np.sqrt(((y1 - y2) ** 2) + ((x1 - x2) ** 2))
                # cg = ((pos / 3) * (qi + 2 * q)) / (qi + q)
                # height = math.sin(el_angle) * cg
                # base = math.cos(el_angle) * cg

                len_x1 = np.sin(ai - np.pi) * 0.6 * h1 * direction
                len_x2 = np.sin(ai - np.pi) * 0.6 * h2 * direction
                len_y1 = np.cos(ai - np.pi) * 0.6 * h1 * direction
                len_y2 = np.cos(ai - np.pi) * 0.6 * h2 * direction
                step_x = np.linspace(xn1, xn2, 11)
                step_y = np.linspace(yn1, yn2, 11)
                step_len_x = np.linspace(len_x1, len_x2, 11)
                step_len_y = np.linspace(len_y1, len_y2, 11)
                average_h = (h1 + h2) / 2
                # fc = face color, ec = edge color
                self.axes[axes_i].text(
                    xn1,
                    yn1,
                    f"q={qi}",
                    color=self.plot_colors["q_load_text"],
                    fontsize=9,
                    zorder=10,
                )
                self.axes[axes_i].text(
                    xn2,
                    yn2,
                    f"q={q}",
                    color=self.plot_colors["q_load_text"],
                    fontsize=9,
                    zorder=10,
                )

                # add multiple arrows to fill load
                for counter, step_xi in enumerate(step_x):
                    if q + qi >= 0:
                        if counter == 0:
                            shape = "right"
                        elif counter == 10:
                            shape = "left"
                        else:
                            shape = "full"
                    else:
                        if counter == 0:
                            shape = "left"
                        elif counter == 10:
                            shape = "right"
                        else:
                            shape = "full"

                    self.axes[axes_i].arrow(
                        step_xi,
                        step_y[counter],
                        step_len_x[counter],
                        step_len_y[counter],
                        head_width=average_h * 0.25,
                        head_length=0.4
                        * np.sqrt(step_len_y[counter] ** 2 + step_len_x[counter] ** 2),
                        ec=self.plot_colors["q_load_arrow_edge"],
                        fc=self.plot_colors["q_load_arrow_face"],
                        shape=shape,
                    )

        for q_id in self.system.loads_q.keys():
            el = self.system.element_map[q_id]
            qi = el.q_load[0]
            q = el.q_load[1]

            x1 = el.vertex_1.x
            y1 = el.vertex_1.y
            x2 = el.vertex_2.x
            y2 = el.vertex_2.y

            if max(qi, q) > 0:
                direction = 1
            else:
                direction = -1

            h1 = 0.05 * max_val * abs(qi) / self.max_q
            h2 = 0.05 * max_val * abs(q) / self.max_q

            assert el.q_angle is not None
            ai = np.pi / 2 - el.q_angle
            el_angle = el.angle
            __plot_patch(h1, h2, x1, y1, x2, y2, ai, qi, q, direction, el_angle)

            if el.q_perp_load[0] != 0 or el.q_perp_load[1] != 0:
                qi = el.q_perp_load[0]
                q = el.q_perp_load[1]

                x1 = el.vertex_1.x + np.sin(ai) * h1 * direction * 2
                y1 = el.vertex_1.y + np.cos(ai) * h1 * direction * 2
                x2 = el.vertex_2.x + np.sin(ai) * h2 * direction * 2
                y2 = el.vertex_2.y + np.cos(ai) * h2 * direction * 2

                if max(qi, q) > 0:
                    direction = 1
                else:
                    direction = -1

                h1 = 0.05 * max_val * abs(qi) / self.max_q
                h2 = 0.05 * max_val * abs(q) / self.max_q

                ai = -el.q_angle
                el_angle = el.angle
                __plot_patch(h1, h2, x1, y1, x2, y2, ai, qi, q, direction, el_angle)

    @staticmethod
    def __arrow_patch_values(
        Fx: float, Fy: float, node: "Node", h: float
    ) -> Tuple[float, float, float, float, float]:
        """Determines the values for the point load arrow patch.

        Args:
            Fx (float): Point load magnitude in x direction
            Fy (float): Point load magnitude in y direction
            node (Node): Node upon which load is applied
            h (float): Scale variable

        Returns:
            Tuple[float, float, float, float, float]: x, y, len_x, len_y, F (for matplotlib plotter)
        """

        F = (Fx**2 + Fy**2) ** 0.5
        len_x = Fx / F * h
        len_y = -Fy / F * h
        x = node.vertex.x - len_x * 1.2
        y = node.vertex.y - len_y * 1.2

        return x, y, len_x, len_y, F

    def __point_load_patch(
        self, max_plot_range: float, verbosity: int = 0, axes_i: int = 0
    ) -> None:
        """Plots the point loads.

        Args:
            max_plot_range (float): Max scale of the plot
            verbosity (int, optional): 0: show values, 1: show arrow only. Defaults to 0.
            axes_i (int, optional): Which set of axes to plot on (for multi-plot windows). Defaults to 0.
        """

        for k in self.system.loads_point:
            Fx, Fy = self.system.loads_point[k]
            F = (Fx**2 + Fy**2) ** 0.5
            node = self.system.node_map[k]
            h = 0.1 * max_plot_range * F / self.max_system_point_load
            x, y, len_x, len_y, F = self.__arrow_patch_values(Fx, Fy, node, h)

            self.axes[axes_i].arrow(
                x,
                y,
                len_x,
                len_y,
                head_width=h * 0.15,
                head_length=0.2 * h,
                ec=self.plot_colors["point_load_edge"],
                fc=self.plot_colors["point_load_fill"],
                zorder=11,
            )
            if verbosity == 0:
                self.axes[axes_i].text(
                    x,
                    y,
                    f"F={F}",
                    color=self.plot_colors["point_load_text"],
                    fontsize=9,
                    zorder=10,
                )

    def __moment_load_patch(self, max_val: float, axes_i: int = 0) -> None:
        """Plots the moment loads.

        Args:
            max_val (float): Max scale of the plot
            axes_i (int, optional): Which set of axes to plot on (for multi-plot windows). Defaults to 0.
        """
        h = 0.2 * max_val
        for k, v in self.system.loads_moment.items():
            node = self.system.node_map[k]
            if v > 0:
                self.axes[axes_i].plot(
                    node.vertex.x,
                    node.vertex.y,
                    marker=r"$\circlearrowleft$",
                    ms=25,
                    color=self.plot_colors["moment_load"],
                )
            else:
                self.axes[axes_i].plot(
                    node.vertex.x,
                    node.vertex.y,
                    marker=r"$\circlearrowright$",
                    ms=25,
                    color=self.plot_colors["moment_load"],
                )
            self.axes[axes_i].text(
                node.vertex.x + h * 0.2,
                node.vertex.y + h * 0.2,
                f"T={v}",
                color=self.plot_colors["moment_load_text"],
                fontsize=9,
                zorder=10,
            )

    def plot_structure(
        self,
        figsize: Optional[Tuple[float, float]],
        verbosity: int,
        show: bool = False,
        supports: bool = True,
        scale: float = 1,
        offset: Sequence[float] = (0, 0),
        gridplot: bool = False,
        annotations: bool = True,
        axes_i: int = 0,
    ) -> Optional["Figure"]:
        """Plots the structure.

        Args:
            figsize (Optional[Tuple[float, float]]): Figure size
            verbosity (int): 0: show node and element IDs, 1: show structure only
            show (bool, optional): If True, plt.figure will plot. Defaults to False.
            supports (bool, optional): If True, supports are plotted. Defaults to True.
            scale (float, optional): Scale of the plot. Defaults to 1.
            offset (Sequence[float], optional): Offset of the plot. Defaults to (0, 0).
            gridplot (bool, optional): If True, the plot will be added to a grid of plots. Defaults to False.
            annotations (bool, optional): If True, structure annotations are plotted. Defaults to True.
            axes_i (int, optional): Which set of axes to plot on (for multi-plot windows). Defaults to 0.

        Returns:
            Optional[Figure]: Returns figure object if in testing mode, else None
        """
        if not gridplot:
            figsize = self.__start_plot(figsize)
            axes_i = 0
        else:
            assert figsize is not None  # figsize is always set in gridplots

        x, y = self.plot_values.structure()

        max_x: float = np.max(x)
        min_x: float = np.min(x)
        max_y: float = np.max(y)
        min_y: float = np.min(y)

        center_x = (max_x - min_x) / 2 + min_x + -offset[0]
        center_y = (max_y - min_y) / 2 + min_y + -offset[1]

        max_plot_range = max(max_x - min_x, max_y - min_y)
        ax_range = max_plot_range * scale
        plusxrange = center_x + ax_range
        plusyrange = center_y + ax_range * figsize[1] / figsize[0]
        minxrange = center_x - ax_range
        minyrange = center_y - ax_range * figsize[1] / figsize[0]

        self.axes[axes_i].axis((minxrange, plusxrange, minyrange, plusyrange))

        for el in self.system.element_map.values():
            x_val, y_val = plot_values_element(el)
            self.axes[axes_i].plot(
                x_val, y_val, color=self.plot_colors["element"], marker="s"
            )

            if verbosity == 0:
                # add node ID to plot
                ax_range = max_plot_range * 0.015
                self.axes[axes_i].text(
                    x_val[0] + ax_range,
                    y_val[0] + ax_range,
                    f"{el.node_id1}",
                    color=self.plot_colors["node_number"],
                    fontsize=9,
                    zorder=10,
                )
                self.axes[axes_i].text(
                    x_val[-1] + ax_range,
                    y_val[-1] + ax_range,
                    f"{el.node_id2}",
                    color=self.plot_colors["node_number"],
                    fontsize=9,
                    zorder=10,
                )

                # add element ID to plot
                factor = 0.02 * self.max_val_structure
                x_scalar = (x_val[0] + x_val[-1]) / 2 - np.sin(el.angle) * factor
                y_scalar = (y_val[0] + y_val[-1]) / 2 + np.cos(el.angle) * factor

                self.axes[axes_i].text(
                    x_scalar,
                    y_scalar,
                    str(el.id),
                    color=self.plot_colors["element_number"],
                    fontsize=9,
                    zorder=10,
                )

                # add element annotation to plot
                # TO DO: check how this holds with multiple structure scales.
                if annotations:
                    x_scalar += +np.sin(el.angle) * factor * 2.3
                    y_scalar += -np.cos(el.angle) * factor * 2.3
                    self.axes[axes_i].text(
                        x_scalar,
                        y_scalar,
                        el.section_name,
                        color=self.plot_colors["annotation"],
                        fontsize=9,
                        zorder=10,
                    )

        # add supports
        if supports:
            self.__fixed_support_patch(max_plot_range * scale)
            self.__hinged_support_patch(max_plot_range * scale)
            self.__rotational_support_patch(max_plot_range * scale)
            self.__roll_support_patch(max_plot_range * scale)
            self.__rotating_spring_support_patch(max_plot_range * scale)
            self.__spring_support_patch(max_plot_range * scale)
            self.__internal_hinges_patch(max_plot_range * scale)

        if verbosity == 0:
            # add_loads
            self.__q_load_patch(max_plot_range, verbosity)
            self.__point_load_patch(max_plot_range, verbosity)
            self.__moment_load_patch(max_plot_range)

        if show:
            self.plot()
            return None
        return self.fig

    def _add_node_values(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        value_1: float,
        value_2: float,
        digits: int,
        axes_i: int = 0,
    ) -> None:
        """Adds the node values to the plot.

        Args:
            x_val (np.ndarray): X locations
            y_val (np.ndarray): Y locations
            value_1 (float): Value of first number
            value_2 (float): Value of second number
            digits (int): Number of digits to round to
            axes_i (int, optional): Which set of axes to plot on (for multi-plot windows). Defaults to 0.
        """
        offset = self.max_val_structure * 0.015

        # add value to plot
        self.axes[axes_i].text(
            x_val[1] - offset,
            y_val[1] + offset,
            f"{round(value_1, digits)}",
            fontsize=9,
            ha="center",
            va="center",
        )
        self.axes[axes_i].text(
            x_val[-2] - offset,
            y_val[-2] + offset,
            f"{round(value_2, digits)}",
            fontsize=9,
            ha="center",
            va="center",
        )

    def _add_element_values(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        value: float,
        index: int,
        digits: int = 2,
        axes_i: int = 0,
    ) -> None:
        """Adds the element values to the plot.

        Args:
            x_val (np.ndarray): X locations
            y_val (np.ndarray): Y locations
            value (float): Value of number
            index (int): Index of value
            digits (int, optional): Number of digits to round to. Defaults to 2.
            axes_i (int, optional): Which set of axes to plot on (for multi-plot windows). Defaults to 0.
        """
        self.axes[axes_i].text(
            x_val[index],
            y_val[index],
            f"{round(value, digits)}",
            fontsize=9,
            ha="center",
            va="center",
        )

    def plot_result(
        self,
        axis_values: Sequence,
        force_1: Optional[float] = None,
        force_2: Optional[float] = None,
        digits: int = 2,
        node_results: bool = True,
        fill_polygon: bool = True,
        color: str = "b",
        axes_i: int = 0,
    ) -> None:
        """Plots a single result on the structure.

        Args:
            axis_values (Sequence): X and Y values
            force_1 (Optional[float], optional): First force to plot. Defaults to None.
            force_2 (Optional[float], optional): Second force to plot. Defaults to None.
            digits (int, optional): Number of digits to round to. Defaults to 2.
            node_results (bool, optional): Whether or not to plot nodal results. Defaults to True.
            fill_polygon (bool, optional): Whether or not to fill a polygon for the result. Defaults to True.
            color (int, optional): Color index with which to draw. Defaults to 0.
            axes_i (int, optional): Which set of axes to plot on (for multi-plot windows). Defaults to 0.
        """
        if fill_polygon:
            rec = mpatches.Polygon(np.vstack(axis_values).T, color=color, alpha=0.3)
            self.axes[axes_i].add_patch(rec)
        # plot force
        x_val = axis_values[0]
        y_val = axis_values[1]

        self.axes[axes_i].plot(x_val, y_val, color=color)

        if node_results and force_1 and force_2:
            self._add_node_values(x_val, y_val, force_1, force_2, digits, axes_i=axes_i)

    def plot(self) -> None:
        """Plots the figure."""
        plt.show()

    def axial_force(
        self,
        factor: Optional[float] = None,
        figsize: Optional[Tuple[float, float]] = None,
        verbosity: int = 0,
        scale: float = 1,
        offset: Sequence[float] = (0, 0),
        show: bool = True,
        gridplot: bool = False,
        axes_i: int = 0,
    ) -> Optional["Figure"]:
        """Plots the axial force.

        Args:
            factor (Optional[float], optional): Scaling factor. Defaults to None.
            figsize (Optional[Tuple[float, float]], optional): Figure size. Defaults to None.
            verbosity (int, optional): 0: show values, 1: show axial force only. Defaults to 0.
            scale (float, optional): Scale of the plot. Defaults to 1.
            offset (Sequence[float], optional): Offset of the plot. Defaults to (0, 0).
            show (bool, optional): If True, plt.figure will plot. Defaults to False.
            gridplot (bool, optional): If True, the plot will be added to a grid of plots. Defaults to False.
            axes_i (int, optional): Which set of axes to plot on (for multi-plot windows). Defaults to 0.

        Returns:
            Optional[Figure]: Returns figure object if in testing mode, else None
        """
        self.plot_structure(
            figsize, 1, scale=scale, offset=offset, gridplot=gridplot, axes_i=axes_i
        )
        assert list(self.system.element_map.values())[0].axial_force is not None
        con = len(list(self.system.element_map.values())[0].axial_force)

        if factor is None:
            max_force = max(
                map(
                    lambda el: max(
                        abs(el.N_1 or 0.0),
                        abs(el.N_2 or 0.0),
                        abs(((el.all_qn_load[0] + el.all_qn_load[1]) / 16) * el.l**2),
                    ),
                    self.system.element_map.values(),
                )
            )
            factor = det_scaling_factor(max_force, self.max_val_structure)

        for el in self.system.element_map.values():
            assert el.N_1 is not None
            assert el.N_2 is not None
            if (
                math.isclose(el.N_1, 0, rel_tol=1e-5, abs_tol=1e-9)
                and math.isclose(el.N_2, 0, rel_tol=1e-5, abs_tol=1e-9)
                and math.isclose(el.all_qn_load[0], 0, rel_tol=1e-5, abs_tol=1e-9)
                and math.isclose(el.all_qn_load[1], 0, rel_tol=1e-5, abs_tol=1e-9)
            ):
                continue
            axis_values = plot_values_axial_force(el, factor, con)
            color = (
                self.plot_colors["axial_force_neg"]
                if el.N_1 < 0
                else self.plot_colors["axial_force_pos"]
            )
            self.plot_result(
                axis_values,
                el.N_1,
                el.N_2,
                node_results=not bool(verbosity),
                color=color,
                axes_i=axes_i,
            )

            point = (el.vertex_2 - el.vertex_1) / 2 + el.vertex_1
            if el.N_1 < 0:
                point.displace_polar(
                    alpha=el.angle + 0.5 * np.pi,
                    radius=0.5 * el.N_1 * factor,
                    inverse_y_axis=True,
                )

                if verbosity == 0:
                    self.axes[axes_i].text(
                        point.x,
                        point.y,
                        "-",
                        ha="center",
                        va="center",
                        fontsize=20,
                        color=self.plot_colors["axial_force_sign"],
                    )
            if el.N_1 > 0:
                point.displace_polar(
                    alpha=el.angle + 0.5 * np.pi,
                    radius=0.5 * el.N_1 * factor,
                    inverse_y_axis=True,
                )

                if verbosity == 0:
                    self.axes[axes_i].text(
                        point.x,
                        point.y,
                        "+",
                        ha="center",
                        va="center",
                        fontsize=14,
                        color=self.plot_colors["axial_force_sign"],
                    )

        if show:
            self.plot()
            return None
        return self.fig

    def bending_moment(
        self,
        factor: Optional[float] = None,
        figsize: Optional[Tuple[float, float]] = None,
        verbosity: int = 0,
        scale: float = 1,
        offset: Sequence[float] = (0, 0),
        show: bool = True,
        gridplot: bool = False,
        axes_i: int = 0,
    ) -> Optional["Figure"]:
        """Plots the bending moment.

        Args:
            factor (Optional[float], optional): Scaling factor. Defaults to None.
            figsize (Optional[Tuple[float, float]], optional): Figure size. Defaults to None.
            verbosity (int, optional): 0: show values, 1: show bending moment only. Defaults to 0.
            scale (float, optional): Scale of the plot. Defaults to 1.
            offset (Sequence[float], optional): Offset of the plot. Defaults to (0, 0).
            show (bool, optional): If True, plt.figure will plot. Defaults to False.
            gridplot (bool, optional): If True, the plot will be added to a grid of plots. Defaults to False.
            axes_i (int, optional): Which set of axes to plot on (for multi-plot windows). Defaults to 0.

        Returns:
            Optional[Figure]: Returns figure object if in testing mode, else None
        """
        self.plot_structure(
            figsize, 1, scale=scale, offset=offset, gridplot=gridplot, axes_i=axes_i
        )
        assert list(self.system.element_map.values())[0].bending_moment is not None
        con = len(list(self.system.element_map.values())[0].bending_moment)
        if factor is None:
            # maximum moment determined by comparing the node's moments and the sagging moments.
            max_moment = max(
                map(
                    lambda el: (
                        max(
                            abs(el.node_1.Tz),
                            abs(el.node_2.Tz),
                            abs(
                                ((el.all_qp_load[0] + el.all_qp_load[1]) / 16) * el.l**2
                            ),
                        )
                        if el.type == "general"
                        else 0
                    ),
                    self.system.element_map.values(),
                )
            )
            factor = det_scaling_factor(max_moment, self.max_val_structure)

        # determine the axis values
        for el in self.system.element_map.values():
            if (
                math.isclose(el.node_1.Tz, 0, rel_tol=1e-5, abs_tol=1e-9)
                and math.isclose(el.node_2.Tz, 0, rel_tol=1e-5, abs_tol=1e-9)
                and math.isclose(el.all_qp_load[0], 0, rel_tol=1e-5, abs_tol=1e-9)
                and math.isclose(el.all_qp_load[1], 0, rel_tol=1e-5, abs_tol=1e-9)
            ):
                # If True there is no bending moment, so no need for plotting.
                continue
            axis_values = plot_values_bending_moment(el, factor, con)
            node_results = verbosity == 0
            self.plot_result(
                axis_values,
                abs(el.node_1.Tz),
                abs(el.node_2.Tz),
                node_results=node_results,
                color=self.plot_colors["bending_moment"],
                axes_i=axes_i,
            )

            if el.all_qp_load:
                assert el.bending_moment is not None
                m_sag = min(el.bending_moment)
                index = find_nearest(el.bending_moment, m_sag)[1]
                offset1 = self.max_val_structure * -0.05

                if verbosity == 0:
                    x = axis_values[0][index] + np.sin(-el.angle) * offset1
                    y = axis_values[1][index] + np.cos(-el.angle) * offset1
                    self.axes[axes_i].text(x, y, f"{round(m_sag, 1)}", fontsize=9)
        if show:
            self.plot()
            return None
        return self.fig

    def shear_force(
        self,
        factor: Optional[float] = None,
        figsize: Optional[Tuple[float, float]] = None,
        verbosity: int = 0,
        scale: float = 1,
        offset: Sequence[float] = (0, 0),
        show: bool = True,
        gridplot: bool = False,
        include_structure: bool = True,
        axes_i: int = 0,
    ) -> Optional["Figure"]:
        """Plots the shear force.

        Args:
            factor (Optional[float], optional): Scaling factor. Defaults to None.
            figsize (Optional[Tuple[float, float]], optional): Figure size. Defaults to None.
            verbosity (int, optional): 0: show values, 1: show shear force only. Defaults to 0.
            scale (float, optional): Scale of the plot. Defaults to 1.
            offset (Sequence[float], optional): Offset of the plot. Defaults to (0, 0).
            show (bool, optional): If True, plt.figure will plot. Defaults to False.
            gridplot (bool, optional): If True, the plot will be added to a grid of plots. Defaults to False.
            include_structure (bool, optional): If True, the structure will be plotted. Defaults to True.
            axes_i (int, optional): Which set of axes to plot on (for multi-plot windows). Defaults to 0.

        Returns:
            Optional[Figure]: Returns figure object if in testing mode, else None
        """
        if include_structure:
            self.plot_structure(
                figsize, 1, scale=scale, offset=offset, gridplot=gridplot, axes_i=axes_i
            )
        if factor is None:
            max_force = max(
                map(
                    lambda el: (
                        np.max(np.abs(el.shear_force))
                        if el.shear_force is not None
                        else 0.0
                    ),
                    self.system.element_map.values(),
                )
            )
            factor = det_scaling_factor(max_force, self.max_val_structure)

        for el in self.system.element_map.values():
            if (
                math.isclose(el.node_1.Tz, 0, rel_tol=1e-5, abs_tol=1e-9)
                and math.isclose(el.node_2.Tz, 0, rel_tol=1e-5, abs_tol=1e-9)
                and math.isclose(el.all_qp_load[0], 0, rel_tol=1e-5, abs_tol=1e-9)
                and math.isclose(el.all_qp_load[1], 0, rel_tol=1e-5, abs_tol=1e-9)
            ):
                # If True there is no bending moment and no shear, thus no shear force,
                # so no need for plotting.
                continue
            axis_values = plot_values_shear_force(el, factor)
            assert el.shear_force is not None
            shear_1 = el.shear_force[0]
            shear_2 = el.shear_force[-1]

            self.plot_result(
                axis_values,
                shear_1,
                shear_2,
                node_results=not bool(verbosity),
                color=self.plot_colors["shear_force"],
                axes_i=axes_i,
            )
        if show:
            self.plot()
            return None
        return self.fig

    def reaction_force(
        self,
        figsize: Optional[Tuple[float, float]],
        verbosity: int,
        scale: float,
        offset: Sequence[float],
        show: bool,
        gridplot: bool = False,
        axes_i: int = 0,
    ) -> Optional["Figure"]:
        """Plots the reaction forces.

        Args:
            figsize (Optional[Tuple[float, float]]): Figure size
            verbosity (int): 0: show node and element IDs, 1: show structure only
            scale (float): Scale of the plot
            offset (Sequence[float]): Offset of the plot
            show (bool): If True, plt.figure will plot
            gridplot (bool, optional): If True, the plot will be added to a grid of plots. Defaults to False.
            axes_i (int, optional): Which set of axes to plot on (for multi-plot windows). Defaults to 0.

        Returns:
            Optional[Figure]: Returns figure object if in testing mode, else None
        """
        self.plot_structure(
            figsize,
            1,
            supports=False,
            scale=scale,
            offset=offset,
            gridplot=gridplot,
            axes_i=axes_i,
        )

        h = 0.2 * self.max_val_structure
        max_force = max(
            map(
                lambda node: max(abs(node.Fx), abs(node.Fy)),
                self.system.reaction_forces.values(),
            )
        )

        for node in self.system.reaction_forces.values():
            if not math.isclose(node.Fx, 0, rel_tol=1e-5, abs_tol=1e-9):
                # x direction
                scale = abs(node.Fx) / max_force * h
                sol = self.__arrow_patch_values(node.Fx, 0, node, scale)
                x = sol[0]
                y = sol[1]
                len_x = sol[2]
                len_y = sol[3]

                self.axes[axes_i].arrow(
                    x,
                    y,
                    len_x,
                    len_y,
                    head_width=h * 0.15,
                    head_length=0.2 * scale,
                    ec=self.plot_colors["reaction_force_arrow_edge"],
                    fc=self.plot_colors["reaction_force_arrow_fill"],
                    zorder=11,
                )

                if verbosity == 0:
                    self.axes[axes_i].text(
                        x,
                        y,
                        f"R={round(node.Fx, 2)}",
                        color=self.plot_colors["reaction_force_text"],
                        fontsize=9,
                        zorder=10,
                    )

            if not math.isclose(node.Fy, 0, rel_tol=1e-5, abs_tol=1e-9):
                # y direction
                scale = abs(node.Fy) / max_force * h
                sol = self.__arrow_patch_values(0, node.Fy, node, scale)
                x = sol[0]
                y = sol[1]
                len_x = sol[2]
                len_y = sol[3]

                self.axes[axes_i].arrow(
                    x,
                    y,
                    len_x,
                    len_y,
                    head_width=h * 0.15,
                    head_length=0.2 * scale,
                    ec=self.plot_colors["reaction_force_arrow_edge"],
                    fc=self.plot_colors["reaction_force_arrow_fill"],
                    zorder=11,
                )

                if verbosity == 0:
                    self.axes[axes_i].text(
                        x,
                        y,
                        f"R={round(node.Fy, 2)}",
                        color=self.plot_colors["reaction_force_text"],
                        fontsize=9,
                        zorder=10,
                    )

            if not math.isclose(node.Tz, 0, rel_tol=1e-5, abs_tol=1e-9):
                # '$...$': render the strings using mathtext
                if node.Tz > 0:
                    self.axes[axes_i].plot(
                        node.vertex.x,
                        node.vertex.y,
                        marker=r"$\circlearrowleft$",
                        ms=25,
                        color=self.plot_colors["reaction_force_arrow_fill"],
                    )
                if node.Tz < 0:
                    self.axes[axes_i].plot(
                        node.vertex.x,
                        node.vertex.y,
                        marker=r"$\circlearrowright$",
                        ms=25,
                        color=self.plot_colors["reaction_force_arrow_fill"],
                    )

                if verbosity == 0:
                    self.axes[axes_i].text(
                        node.vertex.x + h * 0.2,
                        node.vertex.y + h * 0.2,
                        f"T={round(node.Tz, 2)}",
                        color=self.plot_colors["reaction_force_text"],
                        fontsize=9,
                        zorder=10,
                    )
        if show:
            self.plot()
            return None
        return self.fig

    def displacements(  # pylint: disable=arguments-renamed
        self,
        factor: Optional[float] = None,
        figsize: Optional[Tuple[float, float]] = None,
        verbosity: int = 0,
        scale: float = 1,
        offset: Sequence[float] = (0, 0),
        show: bool = True,
        linear: bool = False,
        gridplot: bool = False,
        axes_i: int = 0,
    ) -> Optional["Figure"]:
        """Plots the displacements.

        Args:
            factor (Optional[float], optional): Scaling factor. Defaults to None.
            figsize (Optional[Tuple[float, float]], optional): Figure size. Defaults to None.
            verbosity (int, optional): 0: show values, 1: show displacements only. Defaults to 0.
            scale (float, optional): Scale of the plot. Defaults to 1.
            offset (Sequence[float], optional): Offset of the plot. Defaults to (0, 0).
            show (bool, optional): If True, plt.figure will plot. Defaults to False.
            linear (bool, optional): If True, the bending in between the elements is determined. Defaults to False.
            gridplot (bool, optional): If True, the plot will be added to a grid of plots. Defaults to False.
            axes_i (int, optional): Which set of axes to plot on (for multi-plot windows). Defaults to 0.

        Returns:
            Optional[Figure]: Returns figure object if in testing mode, else None
        """
        self.plot_structure(
            figsize, 1, scale=scale, offset=offset, gridplot=gridplot, axes_i=axes_i
        )
        if factor is None:
            # needed to determine the scaling factor
            max_displacement = max(
                map(
                    lambda el: (
                        np.sqrt(
                            (el.max_extension or 0) ** 2
                            + (el.max_total_deflection or 0) ** 2
                        )
                    ),
                    self.system.element_map.values(),
                )
            )
            factor = det_scaling_factor(max_displacement, self.max_val_structure)

        for el in self.system.element_map.values():
            axis_values = plot_values_deflection(el, factor, linear)
            self.plot_result(
                axis_values,
                node_results=False,
                fill_polygon=False,
                color=self.plot_colors["displaced_elem"],
                axes_i=axes_i,
            )

            if el.type == "general":
                assert el.deflection is not None
                # index of the max deflection
                x = np.linspace(el.vertex_1.x, el.vertex_2.x, el.deflection.size)
                y = np.linspace(el.vertex_1.y, el.vertex_2.y, el.deflection.size)
                xd, yd = plot_values_deflection(el, 1.0, linear)
                deflection = ((xd - x) ** 2 + (yd - y) ** 2) ** 0.5
                index = int(np.argmax(np.abs(deflection)))

                if verbosity == 0:
                    if index != 0 or index != el.deflection.size:
                        self._add_element_values(
                            axis_values[0],
                            axis_values[1],
                            deflection[index],
                            index,
                            3,
                            axes_i=axes_i,
                        )
        if show:
            self.plot()
            return None
        return self.fig

    def results_plot(
        self,
        figsize: Optional[Tuple[float, float]],
        verbosity: int,
        scale: float,
        offset: Sequence[float],
        show: bool,
    ) -> Optional["Figure"]:
        """Plots all the results in one gridded figure.

        Args:
            figsize (Optional[Tuple[float, float]]): Figure size
            verbosity (int): 0: show values, 1: show arrows and polygons only
            scale (float): Scale of the plot
            offset (Sequence[float]): Offset of the plot
            show (bool): If True, plt.figure will plot

        Returns:
            Optional[Figure]: Returns figure object if in testing mode, else None
        """
        plt.close("all")
        self.axes = []
        self.fig = plt.figure(figsize=figsize)
        a = 320
        self.axes.append(self.fig.add_subplot(a + 1))
        plt.title("structure")
        self.plot_structure(
            figsize,
            verbosity,
            show=False,
            scale=scale,
            offset=offset,
            gridplot=True,
            axes_i=0,
        )
        self.axes.append(self.fig.add_subplot(a + 2))
        print(self.axes)
        plt.title("bending moment")
        self.bending_moment(
            factor=None,
            figsize=figsize,
            verbosity=verbosity,
            scale=scale,
            offset=offset,
            show=False,
            gridplot=True,
            axes_i=1,
        )
        self.axes.append(self.fig.add_subplot(a + 3))
        plt.title("shear force")
        self.shear_force(
            factor=None,
            figsize=figsize,
            verbosity=verbosity,
            scale=scale,
            offset=offset,
            show=False,
            gridplot=True,
            axes_i=2,
        )
        self.axes.append(self.fig.add_subplot(a + 4))
        plt.title("axial force")
        self.axial_force(
            factor=None,
            figsize=figsize,
            verbosity=verbosity,
            scale=scale,
            offset=offset,
            show=False,
            gridplot=True,
            axes_i=3,
        )
        self.axes.append(self.fig.add_subplot(a + 5))
        plt.title("displacements")
        self.displacements(
            factor=None,
            figsize=figsize,
            verbosity=verbosity,
            scale=scale,
            offset=offset,
            show=False,
            linear=False,
            gridplot=True,
            axes_i=4,
        )
        self.axes.append(self.fig.add_subplot(a + 6))
        plt.title("reaction force")
        self.reaction_force(
            figsize=figsize,
            verbosity=verbosity,
            scale=scale,
            offset=offset,
            show=False,
            gridplot=True,
            axes_i=5,
        )

        if show:
            self.plot()
            return None
        return self.fig
