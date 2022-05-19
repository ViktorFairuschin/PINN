import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def plot_initial_distribution(model, save_dir=None, max_x=10.):
    """ Plot initial distribution """

    x = np.linspace(0., max_x, 1000)
    t = np.zeros_like(x)

    plt.figure()
    plt.title('Initial distribution of u(x,t) = f(x) at t = 0')
    plt.plot(x, model.f(x), ':', c='k', lw=1, label='ground truth')
    plt.plot(x, model.predict(x, t), '-', c='r', lw=1, label='model')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'initial_distribution.png'))
    plt.show()


def plot_wave_propagation(model, save_dir=None, max_x=10., max_t=4.):
    """ Plot wave propagation. """

    x = np.linspace(0., max_x, 1000)
    t = np.linspace(0., max_t, 100)

    fig, ax = plt.subplots(figsize=(6, 4))
    ln, = plt.plot([], [], '-', c='k', lw=1)

    def init():
        ax.set_xlim(-0.1, 0.1 + max_x)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        return ln,

    def update(t_):
        ax.set_title("t = {:.2f}".format(t_))
        t_ = np.ones_like(x) * t_
        u = model.predict(x, t_)
        ln.set_data(x, u)
        return ln,

    ani = FuncAnimation(
        fig,
        update,
        frames=t,
        init_func=init,
        interval=50,
        blit=False,
        repeat=False)

    if save_dir:
        writer = PillowWriter(fps=30)
        ani.save(os.path.join(save_dir, 'wave_propagation.gif'), writer=writer)

    plt.show()
