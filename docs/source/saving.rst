Saving
======

What do you need to save? You've got a script that represents your model. Just run it!

If you do need to save a model, you can save it with standard python object pickling.

.. code-block:: python

    import pickle
    from anastruct import SystemElements

    ss = SystemElements()

    # save
    with open('my_structure.pkl', 'wb') as f:
        pickle.dump(ss, f)

    # load
    with open('my_structure.pkl', 'rb') as f:
        ss = pickle.load(f)