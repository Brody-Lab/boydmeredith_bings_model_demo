
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

##########
# Plotting code
##########
left_color = "purple"
right_color = "green"
model_color = "pink"

def plot_process(bups, a, params):
    """
    Create a figure summarizing accumulation process

    Draw four panels.
    1. The left and right click events
    2. The sensory adaptation process that determines each click's impact on the accumulator value (before sensory noise is applied)
    3. The magnitide of each click after sensory adaptation
    4. Realizations the accumulation process

    Args:
        bups: A dictionary containing information about the click train and adaptation process
        a: An N X T numpy array containing N realizations at T timepoint
        params: The agent's accumulation parameters

    Returns:
        fig: A figure containing each of the subplots

    """
    fig, ax = plt.subplots(4,1, sharex=True,
                           figsize=(5,7),
                           gridspec_kw={'height_ratios': [.2, .2, .45, 1]})

    plot_clicktrain(bups, ax=ax[0])
    ax[0].set_ylabel('')
    for spine in ['left','bottom']:
        ax[0].spines[spine].set_linewidth(1)
    ax[0].axes.get_xaxis().set_visible(False)
    ax[0].tick_params(left=False)

    plot_adaptation_process(bups, ax=ax[1])
    ax[1].set_xlabel('')
    ax[1].axes.get_xaxis().set_visible(False)
    ax[1].spines['left'].set_linewidth(0)

    plot_adapted_clicks(bups, ax=ax[2])
    ax[2].axes.get_xaxis().set_visible(False)
    ax[2].spines['bottom'].set_visible(False)
    ax[2].spines['left'].set_linewidth(0)
    ax[2].spines['bottom'].set_linewidth(.5)

    plot_accumulation(bups, a, params, ax=ax[3])
    fig.tight_layout()

    ax3pos = ax[3].get_position()
    choice_axpos = [ax3pos.x0 + ax3pos.width + .075 , ax3pos.y0, .1, ax3pos.height]
    choice_ax = fig.add_axes(choice_axpos)
    final_a = a[:,-1]
    B = params["B"]
    sns.kdeplot(y = final_a, warn_singular=False, color=model_color, ax=choice_ax, clip=[-B, B], fill=True)
    choice_ax.axhline(params['bias'],ls=":", color="black")
    choice_ax.axhline(params['B'],ls="-", color="black")
    choice_ax.axhline(-params['B'],ls="-", color="black")
    choice_ax.set_ylim(ax[3].get_ylim())
    choice_ax.axes.get_yaxis().set_visible(False)
    choice_ax.set_title("Final $a$")
    choice_ax.set_xlabel("$P(a)$")

    ax[0].set_xlim([-.05, bups['duration']+.025])

    sns.despine()

    #plot_choices(a, bias=params['bias'], lapse=params['lapse'], ax=choice_ax)

    #fig.align_ylabels()
    #plt.show()
    return fig


def plot_clicktrain(bups, ax=[]):
    """Creates a figure containing a plot of the left and right clicks

    Args:
        bups: a dict containing left, right, left_rate, right_rate and duration

    Returns:
        None
    """
    if ax==[]:
        fig, ax = plt.subplots( figsize=(4,1.75))

    left_bups, right_bups = bups['left'], bups['right']
    left_rate, right_rate = bups['left_rate'], bups['right_rate']
    duration = bups['duration']

    ax.eventplot(left_bups,lineoffsets=-.5,color=left_color, alpha=.5)
    ax.eventplot(right_bups,lineoffsets=.5,color=right_color, alpha=.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Click sign")
    ax.set_title(f"Clicks $(r_L={left_rate:.2f}$ Hz, $r_R={right_rate:.2f}$ Hz)" )
    ax.set_xlim([0, duration])
    ax.set_yticks([-1, 1])
    ax.set_yticklabels([r'$\delta_L$',r'$\delta_R$'])
    for spine in ['left','right','top']:
        ax.spines[spine].set_linewidth(0)

    return None

def plot_adaptation_process(bups, ax=[]):
    ms = 4
    alpha = .3
    if ax == []:
        print('no axes supplied')
        fig, ax = plt.subplots(figsize=(4,2))

    left_bups, right_bups = bups['left'], bups['right']
    tvec, Cfull = bups['tvec'], bups['Cfull']
    Cmax = max(np.hstack([bups['left_adapted'], bups['right_adapted']]))

    ax.plot(tvec, Cfull, color = "gray")
    #ax.plot(left_bups, np.ones_like(left_bups) * Cmax, "o", color=left_color, alpha=alpha, ms=ms)
    #ax.plot(right_bups, np.ones_like(right_bups) * Cmax, "o", color=right_color, alpha=alpha, ms=ms)
    ax.set_title("Adaptation process")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("C")
    ax.set_ylim([0, Cmax*1.2])

def plot_adapted_clicks(bups, ax=[]):
    ms = 4
    alpha = .3
    if ax == []:
        fig, ax = plt.subplots(figsize=(4,2))

    left_bups, right_bups = bups['left'], bups['right']
    left_adapted, right_adapted = bups['left_adapted'], bups['right_adapted']
    ymax = max(np.hstack([left_adapted, right_adapted]))*1.1
    yl = [-ymax, ymax]
    #yl = [-1, 1]
    ax.plot(np.vstack([left_bups, left_bups]),
             np.vstack([np.zeros_like(left_bups), -left_adapted]), color=left_color, alpha=2*alpha)
    ax.plot(np.vstack([right_bups, right_bups]),
             np.vstack([np.zeros_like(right_bups), right_adapted]), color=right_color, alpha=2*alpha)

    ax.plot(left_bups,-left_adapted, "o", color=left_color, alpha=alpha, ms=ms)
    ax.plot(right_bups,right_adapted, "o", color=right_color, alpha=alpha, ms=ms)
    ax.set_xlabel("Time (s)")
    ax.set_title("Adapted clicks")
    ax.set_ylabel(r"$C \cdot \delta_{R,L}$")
    ax.set_ylim(yl)
    ax.spines['bottom'].set_position(('data',0))

    plt.tight_layout()


def plot_accumulation(bups, a_agents, params, ax=[]):
    tvec, dur = bups['tvec'], bups['duration']
    bias, B = params['bias'], params['B']
    n_agents = np.shape(a_agents)[0]
    alpha = 1 / (n_agents ** .25)
    if ax == []:
        fig, ax = plt.subplots(figsize=(4,2))
    alims = [-1, 1]
    ax.set_xlim([0, dur])
    ax.axhline(bias,color='black',linestyle=':')
    ax.axhline(B,color='black',linestyle='-',lw=1)
    ax.axhline(-B,color='black',linestyle='-',lw=1)
    for a in a_agents:
        ax.plot(tvec, a, color=model_color, alpha=alpha)
        alims[0] = min(alims[0],np.min(a)*1.1)
        alims[1] = max(alims[1],np.max(a)*1.1)
    ax.set_ylim(alims)
    ax.set_xlabel("Time (s)")
    ax.set_title("Accumulation process")
    ax.set_ylabel("a")
    sns.despine()

def plot_choices(a_agents, bias=0, lapse=0, ax=[]):
    if ax == []:
        fig, ax = plt.subplots(figsize=(4,2))
    a = a_agents[:,-1]
    nagents = len(a)
    go_right = a > bias
    is_lapse = np.random.random_sample(len(a)) < lapse
    go_right[is_lapse] = np.random.random_sample(sum(is_lapse)) < .5
    ngoright = sum(go_right)
    nlapses = sum(is_lapse)

    ax.scatter(a[~is_lapse], go_right[~is_lapse], label= "non-lapse", alpha=.5)
    ax.scatter(a[is_lapse], go_right[is_lapse], label = "lapse", alpha=.5)
    xl = np.array(ax.get_xlim())
    xl[0] = min(xl[0], bias-.5)
    xl[1] = max(xl[1], bias+.5)
    ax.plot([xl[0],bias], np.ones(2)*lapse/2, color="gray", label="P(go right)")
    ax.plot([bias,xl[1]], 1-np.ones(2)*lapse/2, color="gray")
    ax.plot([bias, bias], [lapse/2, 1-lapse/2], color="gray", linestyle="--")
    ax.set_xlabel("Accumulation value, a")
    ax.set_ylabel("P(go right)")
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(.5 ,1.5))
    ax.set_title(f'{ngoright}/{nagents} realizations chose right; {nlapses} lapse trials')
