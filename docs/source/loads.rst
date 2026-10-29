Loads
=====

anaStruct allows the following loads on a structure. There are loads on nodes and loads on elements. Element loads are
implicitly placed on the loads and recalculated during post processing.

Node loads
----------

Point loads
###########

Point loads are defined in x- and/ or y-direction, or by defining a load with an angle.

.. automethod:: anastruct.fem.system.SystemElements.point_load

Example
.......

.. code-block:: python

    ss.add_element(location=[0, 1])
    ss.point_load(ss.id_last_node, Fx=10, rotation=45)
    ss.show_structure()

.. image:: img/loads/point.png

Bending moments
###############

Moment loads apply a rotational force on the nodes.

.. automethod:: anastruct.fem.system.SystemElements.moment_load

Example
.......

.. code-block:: python

    ss.add_element([5, 0])
    ss.moment_load(node_id=ss.id_last_node, Ty=20)
    ss.show_structure()

.. image:: img/loads/moment.png

Element loads
-------------

Q-loads are distributed loads. They can act perpendicular to the elements direction, parallel to the elements direction,
and in global x and y directions.

q-loads
#######

.. automethod:: anastruct.fem.system.SystemElements.q_load

Example
.......

.. code-block:: python


    ss.add_element([5, 0])
    ss.q_load(q=-1, element_id=ss.id_last_element, direction='element')
    ss.show_structure()

.. image:: img/loads/q-load.png


thermal loads
#############

A uniform temperature change gives a free axial strain ``alpha * delta_t``, a temperature gradient over the section
depth gives a free curvature ``alpha * delta_t_gradient / h``. A positive gradient means a warmer upper face of a
horizontal element. Thermal loads can also be used in a :class:`.LoadCase`.

.. automethod:: anastruct.fem.system.SystemElements.thermal_load

Example
.......

.. code-block:: python

    ss = SystemElements(EA=1e6, EI=1e4)
    ss.add_element([[0, 0], [10, 0]])
    ss.add_support_fixed([1, 2])
    ss.thermal_load(element_id=1, delta_t=20, alpha=1.2e-5, delta_t_gradient=15, h=0.5)
    ss.solve()


Remove loads
------------

.. automethod:: anastruct.fem.system.SystemElements.remove_loads
