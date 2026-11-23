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

To start a geometrical non linear calculation you'll need to set the `geometrical_non_linear` to True. The second
order effects within a member are only included if the member is divided into several elements, e.g. with
`discretize` (or `add_multiple_elements`). This calculation is an approximation and gets more accurate with more
elements.

.. code-block:: python

    ss.discretize(n=20)
    ss.solve(geometrical_non_linear=True)

The calculation also determines the buckling factor, `ss.buckling_factor`. If you only need a more accurate buckling
factor, pass a `discretize_kwargs` dictionary instead. It divides a copy of the model that is used for the buckling
factor only; the model itself, and its second order results, are unchanged.

.. code-block:: python

    ss.solve(geometrical_non_linear=True, discretize_kwargs=dict(n=20))