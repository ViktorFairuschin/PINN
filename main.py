import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tensorflow warnings

import tensorflow as tf
import datetime
from pinn.model import PINN
from pinn.utils import plot_initial_distribution, plot_wave_propagation


def main():

    # set random seed

    tf.random.set_seed(42)

    # define models parameters

    units = [100] * 5
    activation = 'sigmoid'
    c = 1
    lr = 0.001
    batch_size = 64
    log_dir = './logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # create model

    model = PINN(units=units, activation=activation, c=c, lr=lr, batch_size=batch_size, log_dir=log_dir)

    # train model

    model.fit(steps=20000)

    # plot initial distribution

    plot_initial_distribution(model, save_dir='./results')

    # plot wave propagation

    plot_wave_propagation(model, save_dir='./results')


if __name__ == '__main__':
    main()
