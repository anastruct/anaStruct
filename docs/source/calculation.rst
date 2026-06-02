Calculation
===========

Once all the elements, supports and loads are in place, solving the calculation is as easy as calling the `solve`
method.

.. automethod:: anastruct.fem.system.SystemElements.solve

Non linear
##########

The model will automatically do a non linear calculation if there are non linear nodes present in the
SystemElements state. You can however force the model to do a linear calculation with the `force_linear` parameter.

Geometrical non linear
######################

To start a geometrical non linear calculation you'll need to set the `geometrical_non_linear` to True. It is also wise
to pass a `discretize_kwargs` dictionary.

.. code-block:: python

    ss.solve(geometrical_non_linear=True, discretize_kwargs=dict(n=20))

With this dictionary you can set the amount of discretization elements
generated during the geometrical non linear calculation. This calculation is an approximation and gets more accurate
with more discretization elements.