from ipywidgets import widgets, AppLayout, interact, interactive, fixed
from IPython.display import display, Math, Latex
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from bings_model_demo import model as bm
from bings_model_demo import plots as dp

sns.set_context('notebook')

def clicks_eventhandler(change):
    global bups
    bups = bm.make_clicktrain(total_rate=rate_slider.value, gamma=gamma_slider.value,
                               duration=dur_slider.value, rng=seed_slider.value, stereo_click=stereo_check.value)
    adaptation_eventhandler(change)

def adaptation_eventhandler(change):
    bm.make_adapted_clicks(bups, phi=phi_slider.value, tau_phi=tau_phi_slider.value)
    integration_eventhandler(change)

def integration_eventhandler(change):
    global a
    a = bm.integrate_adapted_clicks(bups, lam=lambda_slider.value, s2s=s2s_slider.value,
                             s2a=s2a_slider.value, s2i=s2i_slider.value, bias=bias_slider.value,
                             B=B_slider.value, nagents=nagent_slider.value, rng=seed_slider.value)
    choice_eventhandler(change)

def choice_eventhandler(change):
    choice_params = {"bias":bias_slider.value, "lapse":lapse_slider.value, "B":B_slider.value}
    plot_out.clear_output(wait=True)
    with plot_out:
        dp.plot_process(bups, a, choice_params)
        plt.show()

plot_out = widgets.Output()

bups_text = widgets.Output()
with bups_text:
    print("Click train parameters\n")

adaptation_text = widgets.Output()
with adaptation_text:
    print("\nAdaptation parameters\n")
    #display(Math(r'$dC = \frac{1-C}{\tau_{\phi}}dt + (\phi - 1) C (\delta_{t,t_R} + \delta_{t,t_L})$'))

integration_text = widgets.Output()
with integration_text:
    print("\n\nAccumulation parameters \n")
    #display(Math(r'$da = (\eta C \delta_R - \eta C \delta_L)dt + \lambda a dt + \sigma_adW $'))
    #display(Math('$a_0 = \mathcal{N}(0, \sigma_i^2)$'))

choice_text = widgets.Output()
with choice_text:
    print("\nChoice parameters\n")

# Define click parameters
stereo_check = widgets.Checkbox(value=True, description="Stereo Click")
seed_slider = widgets.IntSlider(value=1, min=0, max=10, description="Seed")
rate_slider = widgets.IntSlider(value=40,min=5,max=50,step=5,description=r'$r_L+r_R$ (Hz)')
gamma_slider = widgets.FloatSlider(value=1.5,min=-5,max=5,step=.25,description=r"$\gamma = \log \frac{r_1}{r_2}$")
dur_slider = widgets.FloatSlider(value=1,min=.25,max=5,step=.25,description="T (s)")

seed_slider.observe(clicks_eventhandler, names='value')
stereo_check.observe(clicks_eventhandler, names='value')
rate_slider.observe(clicks_eventhandler, names='value')
gamma_slider.observe(clicks_eventhandler, names='value')
dur_slider.observe(clicks_eventhandler, names='value')

# Define adaptation parameters
phi_slider = widgets.FloatSlider(value=.1,min=.001,max=1.5,step=.05,description=r"$\phi$")
tau_phi_slider = widgets.FloatSlider(value=.15,min=.001,max=1,step=.05,description=r"$\tau_{\phi}$")

phi_slider.observe(clicks_eventhandler, names='value')
tau_phi_slider.observe(clicks_eventhandler, names='value')

# Define integration parameters
lambda_slider = widgets.FloatSlider(value=0., min = -5., max=5., step=.25, description=r"$\lambda$")
s2s_slider = widgets.FloatSlider(value=.0, min = 0, max=50., step=.25, description=r"$\sigma^2_s$")
s2a_slider = widgets.FloatSlider(value=.0, min = 0, max=10., step=.25, description=r"$\sigma^2_a$")
s2i_slider = widgets.FloatSlider(value=0, min = 0, max=5., step=.25, description=r"$\sigma^2_i$")
B_slider = widgets.FloatSlider(value=10., min = 0., max=25., step=1, description=r"$B$")
nagent_slider = widgets.IntSlider(value=10, min = 1, max=50, description=r"N samples")

lambda_slider.observe(clicks_eventhandler, names='value')
s2s_slider.observe(clicks_eventhandler, names='value')
s2a_slider.observe(clicks_eventhandler, names='value')
s2i_slider.observe(clicks_eventhandler, names='value')
B_slider.observe(clicks_eventhandler, names='value')
nagent_slider.observe(clicks_eventhandler, names='value')

# Define choice parameters
bias_slider = widgets.FloatSlider(value=0, min = -5, max=5., step=.05, description=r"Bias")
lapse_slider = widgets.FloatSlider(value=.05, min = 0., max=1., step=.01, description=r"Lapse")
bias_slider.observe(clicks_eventhandler, names='value')
lapse_slider.observe(clicks_eventhandler, names='value')

bup_inputs = widgets.VBox([bups_text, stereo_check, rate_slider, gamma_slider, dur_slider])
adaptation_inputs = widgets.VBox([adaptation_text, phi_slider, tau_phi_slider])
integration_inputs = widgets.VBox([integration_text, lambda_slider, s2s_slider, s2a_slider,
                                   s2i_slider, B_slider, nagent_slider])
choice_inputs = widgets.VBox([choice_text, bias_slider, lapse_slider])
inputs = widgets.VBox([seed_slider, bup_inputs, adaptation_inputs, integration_inputs, choice_inputs])

interface = widgets.HBox([inputs, plot_out])

def draw_gui():
    """Create the gui showing how a trail from the Poisson Clicks task is integrated by Bings model to produce a choice"""
    display(interface)
    clicks_eventhandler([])
