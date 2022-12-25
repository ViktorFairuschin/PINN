# Physics Informed Neural Network

This is a tensorflow implementation of physics informed neural 
network for one-dimensional wave equation.

## A brief introduction to physics informed neural networks

$$
u" - c^2 \ddot{u} = 0
$$

Consider a one-dimensional wave equation 

![equation](equations/light_mode/1.svg)

with two initial conditions

![equation](equations/light_mode/2.svg)

and two boundary conditions

![equation](equations/light_mode/3.svg)

The main idea behind physics informed neural networks (PINN) is to approximate 
the solution, u(x,t), of the wave equation with a deep neural network by 
exploiting the ability of neural networks to act as universal function 
approximators.

![equation](equations/light_mode/4.svg)

![equation](equations/light_mode/5.svg)

![equation](equations/light_mode/6.svg)

![equation](equations/light_mode/7.svg)

![equation](equations/light_mode/8.svg)

# Results

![results](results/initial_distribution.png)

![results](results/wave_propagation.gif)

# References




