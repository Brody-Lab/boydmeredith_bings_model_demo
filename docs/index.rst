.. Bings Model Demo documentation master file, created by
   sphinx-quickstart on Thu Jan 26 17:52:10 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bing's Model Demo documentation!
============================================

**Bing's Model Demo** creates an interactive jupyter notebook designed to help build intuition about Bing's model, an elaboration of the drift diffusion model.
Check out the :doc:`usage` section for further information, including how to :ref:`install <installation>` the project


Contents 
--------
.. toctree::

    usage
    source/modules
    ../Bings Model Demo

Let's put a widget here for fun

.. jupyter-execute::

    from ipywidgets import VBox, jsdlink, IntSlider, Button
    s1, s2 = IntSlider(max=200, value=100), IntSlider(value=40)
    b = Button(icon='legal')
    jsdlink((s1, 'value'), (s2, 'max'))
    VBox([s1, s2, b])

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
