EKF: Inner Workings
===================

This page explains the error-state Extended Kalman Filter (EKF) used in
``ackermann-jax``.  The filter estimates the full 3-D rigid-body state of the
car by maintaining a *nominal* trajectory and a Gaussian distribution over the
*error* from that trajectory.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

A standard EKF linearises the dynamics around the current estimate and
propagates a covariance matrix to track uncertainty.  Because the car state
contains an :math:`SO(3)` rotation (which lives on a manifold, not a vector space),
a plain vector-space EKF would accumulate large linearisation errors near
singularities.

The **error-state** (or "indirect") formulation avoids this by keeping the
nominal state on the manifold and tracking only a *small* perturbation
:math:`\delta \mathbf{x}` in a local tangent space.  All covariance arithmetic is done
in that 16-dimensional tangent space; the manifold update is applied via
:func:`~ackermann_jax.errorDyn.inject_error`.

State Representation
--------------------

Nominal State
~~~~~~~~~~~~~

The nominal state :math:`x_{\mathrm{nom}}` is an
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

Error-State Vector
~~~~~~~~~~~~~~~~~~

The error-state :math:`\delta \mathbf{x} \in \mathbb{R}^{16}` lives in the tangent
space of the nominal state.  Its layout is defined by
:data:`~ackermann_jax.ekf.ERROR_DIM`:

.. list-table::
   :header-rows: 1
   :widths: 10 15 55 10

   * - Slice Index
     - Symbol
     - Quantity
     - Dim
   * - ``0:3``
     - :math:`\delta p_W`
     - World-frame position error
     - 3
   * - ``3:6``
     - :math:`\delta\theta`
     - :math:`SO(3)` rotation error (axis-angle / exponential map)
     - 3
   * - ``6:9``
     - :math:`\delta v_W`
     - World-frame velocity error
     - 3
   * - ``9:12``
     - :math:`\delta\omega_B`
     - Body-frame angular velocity error
     - 3
   * - ``12:16``
     - :math:`\delta\Omega`
     - Wheel angular velocity errors (FL, FR, RL, RR)
     - 4

The rotation error :math:`\delta\theta` is a 3-vector in the Lie algebra
:math:`\mathfrak{so}(3)`.  Applying it to the nominal rotation uses the
exponential map:

.. math::

   R^+ = R_{\mathrm{nom}} \, \exp(\delta\theta^\times)

where :math:`(\cdot)^\times` is the skew-symmetric (hat) operator.

Predict Step
------------

Given control input :math:`u` and timestep :math:`dt`, the predict step
(:func:`~ackermann_jax.ekf.ekf_predict`) does two things:

1. **Nominal propagation** — integrate the car model forward:

   .. math::

      \hat{\mathbf{x}}_{k+1|k} = f(\hat{\mathbf{x}}_{\mathrm{nom}}, \mathbf{u}, \Delta t)

2. **Covariance propagation** — linearize the error dynamics and advance
   the covariance prediction as well:

   .. math::

      P_{k+1|k} = F P F^\top + Q

   where :math:`F` is the discrete state-transition Jacobian computed by
   :func:`~ackermann_jax.ekf.compute_F` via ``jax.jacobian``:

   .. math::

      F = \left. \frac{\partial\, f_{\delta}(\delta \mathbf{x})}{\partial\, \delta \mathbf{x}}
          \right|_{\delta \mathbf{x} = 0}

   and :math:`f_\delta` is the error-state dynamics function
   :func:`~ackermann_jax.errorDyn.error_dynamics`.

   :math:`Q \in \mathbb{R}^{16 \times 16}` is the process-noise covariance matrix.

Update Step
-----------

When a measurement :math:`z \in \mathbb{R}^m` arrives, the update step
(:func:`~ackermann_jax.ekf.ekf_update`) fuses it into the estimate:

1. **Measurement Jacobian** — computed by
   :func:`~ackermann_jax.ekf.compute_H` via automatic differentiation through
   :func:`~ackermann_jax.errorDyn.inject_error`:

   .. math::

      H = \left.
            \frac{\partial\, h\!\left(
              \mathrm{inject\_error}(\mathbf{x}_{\mathrm{nom}},\, \delta \mathbf{x})
            \right)}{\partial\, \delta \mathbf{x}}
          \right|_{\delta \mathbf{x} = 0}

   This correctly accounts for the :math:`SO(3)` perturbation convention so that
   :math:`H` maps error-state coordinates to measurement space.

2. **Innovation covariance and Kalman gain:**

   .. math::

      S = H P H^\top + R

   .. math::

      K = P H^\top S^{-1}

3. **Nominal state correction** — compute the error correction and inject
   it back onto the manifold:

   .. math::

      \delta x^* = K \,(z - h(x_{\mathrm{nom}}^-))

   .. math::

      x_{\mathrm{nom}}^+ = \mathrm{inject\_error}(x_{\mathrm{nom}}^-,\, \delta x^*)

4. **Covariance update** (Joseph form, preserves positive semi-definiteness):

   .. math::

      P^+ = (I - KH)\, P\, (I - KH)^\top + K R K^\top

Sensor Models
-------------

.. The filter supports multiple measurement sources, each defined by a
.. measurement function :math:`h : \mathcal{X} \to \mathbb{R}^m`.

GPS (2-D position)
~~~~~~~~~~~~~~~~~~

.. math::

   h_{\mathrm{GPS}}(x) = p_W[0:2]

Returns the :math:`x`-:math:`y` components of latitude
and longitude, while :math:`z` is currently assumed to be the height of the vehicle,
and differentiations in height can be achieved by passing in a different measurement.

Wheel Encoders
~~~~~~~~~~~~~~

.. math::

   h_{\mathrm{wheels}}(x) = \Omega_W

Returns all four wheel angular velocities.  RPM readings from the encoders
are converted to :math:`\mathrm{rad/s}` before use.

Multi-Sensor Fusion
~~~~~~~~~~~~~~~~~~~

When multiple sensors fire in one timestep, call :func:`~ackermann_jax.ekf.ekf_predict`
once and then chain :func:`~ackermann_jax.ekf.ekf_update` for each source::

   ekf = ekf_predict(model, ekf, u, Q, dt)
   ekf = ekf_update(ekf, z_gps,    h_gps,    R_gps)
   ekf = ekf_update(ekf, z_wheels, h_wheels, R_wheels)

Each sequential update conditions on the result of the previous one.

.. note::
   In the future, these update steps should be merged into a *single* update for computational efficiency.

JAX Integration
---------------

All filter functions are designed to work natively with JAX:

- :func:`~ackermann_jax.ekf.ekf_predict` and
  :func:`~ackermann_jax.ekf.ekf_update` are decorated with
  :func:`jax.jit` for compiled execution.
- :class:`~ackermann_jax.ekf.EKFState` is a registered pytree, so it passes
  through :func:`jax.jit` and :func:`jax.lax.scan` without extra wrapping.
- Jacobians :math:`F` and :math:`H` are computed via ``jax.jacobian`` rather
  than hand-derived formulas, keeping the implementation concise and correct
  as the dynamics evolve.

.. note::
   On Jetson Orin (JetPack 6.x) the ``jnp.linalg.inv(S)`` call inside
   :func:`~ackermann_jax.ekf.ekf_update` can trigger a cuSolver internal
   error.  Replace it with ``jnp.linalg.solve(S, H @ ekf.P).T`` or set
   ``XLA_FLAGS=--xla_gpu_cusolver_disable=true`` as a workaround.

API Reference
-------------

See :doc:`api/ekf` for the full API documentation.
