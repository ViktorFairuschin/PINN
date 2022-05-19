import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tensorflow warnings

import tensorflow as tf
import logging
import sys


console_formatter = logging.Formatter("%(levelname)s in %(name)s : %(message)s")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(console_formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)


class PINN:
    """ Physics informed neural network model for one-dimensional wave equation. """

    def __init__(
            self,
            units: list,
            activation: str,
            c: float,
            lr: float,
            batch_size: int,
            log_dir: str,
            **kwargs
    ):

        self.c = c
        self.units = units
        self.batch_size = batch_size

        # create neural network to approximate u(x,t)

        self.u = tf.keras.models.Sequential(name='pinn')
        self.u.add(tf.keras.layers.InputLayer(input_shape=(2,)))
        for units in self.units:
            self.u.add(tf.keras.layers.Dense(units, activation))
        self.u.add(tf.keras.layers.Dense(1))
        self.u.summary()

        # create optimizer, loss and metrics

        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.loss_metrics = tf.keras.metrics.Mean('loss', dtype=tf.float32)

        # create summary writer to monitor model training with Tensorboard

        self.summary_writer = tf.summary.create_file_writer(log_dir)

    def train_step(self):
        with tf.GradientTape() as g:
            loss = self.compute_loss()
        gradients = g.gradient(loss, self.u.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.u.trainable_variables))

    def fit(self, steps):
        template = "\r" + "Step {}/{}: " + "loss {:.2e}"
        for step in range(1, steps + 1):
            self.train_step()
            with self.summary_writer.as_default():
                tf.summary.scalar('loss', self.loss_metrics.result(), step=step)
            if step % 10 == 0:
                sys.stdout.write(template.format(step, steps, self.loss_metrics.result()))

    def predict(self, x, t):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        t = tf.convert_to_tensor(t, dtype=tf.float32)
        u = self.u(tf.stack([x, t], axis=1))
        return u.numpy()

    @tf.function
    def compute_loss(self):
        pde = self.pde()
        bc = self.bc()
        ic = self.ic()
        loss = pde + bc + ic
        self.loss_metrics.update_state(loss)
        return loss

    @tf.function
    def pde(self):
        """ u_tt(x,t) - c^2 * u_xx(x,t) = 0,    0 < x < 1,     0 < t < 1 """

        x = tf.random.uniform((self.batch_size,), 0., 1.)
        t = tf.random.uniform((self.batch_size,), 0., 1.)
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            g.watch(t)
            with tf.GradientTape(persistent=True) as gg:
                gg.watch(x)
                gg.watch(t)
                u = self.u(tf.stack([x, t], axis=1))
            u_x = gg.gradient(u, x)
            u_t = gg.gradient(u, t)
        u_xx = g.gradient(u_x, x)
        u_tt = g.gradient(u_t, t)
        del g, gg
        return tf.reduce_mean(tf.square(u_tt - self.c ** 2 * u_xx))

    @tf.function
    def f(self, x):
        """ initial distribution of u(x,t) at t = 0 """

        return tf.exp(- tf.square(x - 0.5) / 0.005)

    @tf.function
    def bc(self):
        """ u(0,t) = u(1,t) = 0,   0 < t < 1 """

        t = tf.random.uniform((self.batch_size,), 0., 1.)
        x = tf.ones_like(t)
        u = self.u(tf.stack([x, t], axis=1))
        x = tf.zeros_like(t)
        u += self.u(tf.stack([x, t], axis=1))
        return tf.reduce_mean(tf.square(u))

    @tf.function
    def ic(self):
        """ u(x,0) = f(x),     u_t(x,0) = 0 """

        x = tf.random.uniform((self.batch_size,), 0., 1.)
        t = tf.zeros_like(x)
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            g.watch(t)
            u = self.u(tf.stack([x, t], axis=1))
        u_t = g.gradient(u, t)
        del g
        f = self.f(x)
        return tf.reduce_mean(tf.square(u_t)) + self.mse(f, u)
