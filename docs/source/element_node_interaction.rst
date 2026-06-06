Element/ node interaction
=========================

Once you structures will get more and more complex, it will become harder to keep count of element id and node ids.
The `SystemElements` class therefore has several methods that help you:

* Find a node id based on a x- and y-coordinate
* Find the nearest node id based on a x- and y-coordinate
* Get all the coordinates of all nodes.

Find node id based on coordinates
#################################

.. automethod:: anastruct.fem.system.SystemElements.find_node_id

Find nearest node id based on coordinates
#########################################

.. automethod:: anastruct.fem.system.SystemElements.nearest_node

Query node coordinates
######################

.. automethod:: anastruct.fem.system.SystemElements.nodes_range