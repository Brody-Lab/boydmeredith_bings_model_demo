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

    import matplotlib.pyplot as plt
    import numpy as np
    from ipywidgets import widgets, VBox, jsdlink, IntSlider, Button
    s1, s2 = IntSlider(max=200, value=100), IntSlider(value=40)
    b = Button(icon='legal')
    jsdlink((s1, 'value'), (s2, 'max'))
    out = widgets.Output()
    def eventhandler(change):
        out.clear_output()
        with out:
            plt.plot(np.arange(5),change.new*np.arange(5))
            plt.show()
   s1.observe(eventhandler, names='value')
   display(widgets.HBox([s1, out]))
    
Let's try the model ui

.. jupyter-execute:: 

    import os
    import sys
    print(sys.path)
    sys.path.insert(0, os.path.relpath('..'))
    import src.model_ui


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
