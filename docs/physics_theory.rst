Dynamical System Derivation
===========================

This page explains the full dynamical derivation of the Ackermann Car Model
used in :code-link:`ackermann-jax <https://github.com/cornellev/ackermann-jax>`.  The model is a 3D extension of the standard
2D bicycle model, with a full 3D rigid-body state and a simple friction model for the wheels.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

Body Layout and Frame Conventions
---------------------------------

Car and Control States
----------------------

There are two main state representations being used for this system,
one for the control variable :math:`\mathbf{u}` and one for the full state :math:`\mathbf{x}`. 

The car state :math:`\mathbf{x}` is an
:class:`~ackermann_jax.car.AckermannCarState` with fields:

.. list-table::
   :header-rows: 1
   :widths: 15 20 50

   * - Field
     - Symbol
     - Description
   * - ``p_W``
     - :math:`p_W \in \mathbb{R}^3`
     - World-frame position
   * - ``R_WB``
     - :math:`R_{WB} \in SO(3)`
     - Rotation from body to world frame
   * - ``v_W``
     - :math:`v_W \in \mathbb{R}^3`
     - World-frame linear velocity
   * - ``w_B``
     - :math:`\omega_B \in \mathbb{R}^3`
     - Body-frame angular velocity
   * - ``omega_W``
     - :math:`\Omega_W \in \mathbb{R}^4`
     - Wheel angular velocities (FL, FR, RL, RR)

  
Similarly, the car control variable :math:`\mathbf{u}` is an
:class:`~ackermann_jax.car.AckermannCarInput` with fields:

.. list-table::
  :header-rows: 1
  :widths: 15 20 50

  * - Field
    - Symbol
    - Description
  * - ``delta``
    - :math:`\delta \in \mathbb{R}`
    - Steering angle (front wheels)
  * - ``tau_w``
    - :math:`\tau_w \in \mathbb{R}^4`
    - Torque applied to each wheel (FL, FR, RL, RR)


Modeling of Main System
-----------------------

With the frmae conventions above, we first extract all components
from the state and control variables. Then, we assemble the body frame vectors, via the following code::

  t_B = jnp.stack([jnp.cos(delta), jnp.sin(delta), 0.0])
  n_B = jnp.stack([-jnp.sin(delta), jnp.cos(delta), 0.0])


Smaller Physical Models
-----------------------

Tire Contact Model
~~~~~~~~~~~~~~~~~~

Tire Slip Model
~~~~~~~~~~~~~~~

Controllers (Analytical/Simulation)
-----------------------------------

Steering Control
~~~~~~~~~~~~~~~~

Motor Control (& Configuration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
