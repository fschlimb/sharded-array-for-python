"""
Linear wave equation benchmark

Usage:

Verify solution with 128x128 problem size

.. code-block::

    python wave_equation.py

Run a performance test with 1024x1024 problem size.
Runs a fixed number of steps with a small time step.

.. code-block::

    python wave_equation.py -n 1024 -t

Run with numpy backend

.. code-block::

    python wave_equation.py -b numpy ...

"""
import math
import numpy
import time as time_mod
import argparse


def run(n, backend, benchmark_mode):
    if backend == "ddpt":
        import ddptensor as np
        from ddptensor.numpy import fromfunction
        from ddptensor import init, fini, sync
        all_axes = [0, 1]
    elif backend == "numpy":
        import numpy as np
        from numpy import fromfunction
        init = fini = sync = lambda x = None: None
        all_axes = None
    else:
        raise ValueError(f'Unknown backend: "{backend}"')

    print(f'Using backend: {backend}')
    init(False)

    # constants
    h = 1.0
    g = 9.81

    # domain extent
    # NOTE need to be floats
    xmin = -1.0
    xmax = 1.0
    ymin = -1.0
    ymax = 1.0
    lx = xmax - xmin
    ly = ymax - ymin

    # grid resolution
    nx = n
    ny = n
    # grid spacing
    dx = lx/nx
    dy = lx/ny

    # export interval
    t_export = 0.02
    t_end = 1.0

    # coordinate arrays
    x_t_2d = fromfunction(
        lambda i, j: xmin + i*dx + dx/2, (nx, ny), dtype=np.float64)
    y_t_2d = fromfunction(
        lambda i, j: ymin + j*dy + dy/2, (nx, ny), dtype=np.float64)

    T_shape = (nx, ny)
    U_shape = (nx + 1, ny)
    V_shape = (nx, ny+1)

    dofs_T = int(numpy.prod(numpy.asarray(T_shape)))
    dofs_U = int(numpy.prod(numpy.asarray(U_shape)))
    dofs_V = int(numpy.prod(numpy.asarray(V_shape)))

    print(f'Grid size: {nx} x {ny}')
    print(f'Elevation DOFs: {dofs_T}')
    print(f'Velocity  DOFs: {dofs_U + dofs_V}')
    print(f'Total     DOFs: {dofs_T + dofs_U + dofs_V}')

    # prognostic variables: elevation, (u, v) velocity
    e = np.full(T_shape, 0.0, np.float64)
    u = np.full(U_shape, 0.0, np.float64)
    v = np.full(V_shape, 0.0, np.float64)

    # auxiliary variables for RK time integration
    e1 = np.full(T_shape, 0.0, np.float64)
    u1 = np.full(U_shape, 0.0, np.float64)
    v1 = np.full(V_shape, 0.0, np.float64)
    e2 = np.full(T_shape, 0.0, np.float64)
    u2 = np.full(U_shape, 0.0, np.float64)
    v2 = np.full(V_shape, 0.0, np.float64)

    def exact_elev(t, x_t_2d, y_t_2d, lx, ly):
        """
        Exact solution for elevation field.

        Returns time-dependent elevation of a 2D standing wave in a rectangular
        domain.
        """
        amp = 0.5
        c = (g * h) ** 0.5
        n = 1
        sol_x = np.cos(2 * n * math.pi * x_t_2d / lx)
        m = 1
        sol_y = np.cos(2 * m * math.pi * y_t_2d / ly)
        omega = c * math.pi * ((n/lx)**2 + (m/ly)**2)**0.5
        # NOTE ddpt fails with scalar computation
        sol_t = numpy.cos(2 * omega * t)
        return amp * sol_x * sol_y * sol_t

    # inital elevation
    e[0:nx, 0:ny] = exact_elev(0.0, x_t_2d, y_t_2d, lx, ly)

    # compute time step
    alpha = 0.5
    c = (g * h) ** 0.5
    dt = alpha * dx / c
    dt = t_export / int(math.ceil(t_export / dt))
    nt = int(math.ceil(t_end / dt))
    if benchmark_mode:
        dt = 1e-5
        nt = 100
        t_export = dt*25

    print(f'Time step: {dt} s')
    print(f'Total run time: {t_end} s, {nt} time steps')

    sync()

    def rhs(u, v, e):
        """
        Evaluate right hand side of the equations
        """
        # sign convention: positive on rhs

        # pressure gradient -g grad(elev)
        dudt = -g * (e[1:nx, 0:ny] - e[0:nx-1, 0:ny]) / dx
        dvdt = -g * (e[0:nx, 1:ny] - e[0:nx, 0:ny-1]) / dy

        # velocity divergence -h div(u)
        dedt = -h * ((u[1:nx+1, 0:ny] - u[0:nx, 0:ny]) / dx +
                     (v[0:nx, 1:ny+1] - v[0:nx, 0:ny]) / dy)

        return dudt, dvdt, dedt

    def step(u, v, e, u1, v1, e1, u2, v2, e2):
        """
        Execute one SSPRK(3,3) time step
        """
        dudt, dvdt, dedt = rhs(u, v, e)
        u1[1:nx, 0:ny] = u[1:nx, 0:ny] + dt * dudt
        v1[0:nx, 1:ny] = v[0:nx, 1:ny] + dt * dvdt
        e1[0:nx, 0:ny] = e[0:nx, 0:ny] + dt * dedt

        dudt, dvdt, dedt = rhs(u1, v1, e1)
        u2[1:nx, 0:ny] = 0.75*u[1:nx, 0:ny] + 0.25*(u1[1:nx, 0:ny] + dt*dudt)
        v2[0:nx, 1:ny] = 0.75*v[0:nx, 1:ny] + 0.25*(v1[0:nx, 1:ny] + dt*dvdt)
        e2[0:nx, 0:ny] = 0.75*e[0:nx, 0:ny] + 0.25*(e1[0:nx, 0:ny] + dt*dedt)

        dudt, dvdt, dedt = rhs(u2, v2, e2)
        u[1:nx, 0:ny] = u[1:nx, 0:ny]/3.0 + 2.0/3.0*(u2[1:nx, 0:ny] + dt*dudt)
        v[0:nx, 1:ny] = v[0:nx, 1:ny]/3.0 + 2.0/3.0*(v2[0:nx, 1:ny] + dt*dvdt)
        e[0:nx, 0:ny] = e[0:nx, 0:ny]/3.0 + 2.0/3.0*(e2[0:nx, 0:ny] + dt*dedt)

    t = 0
    i_export = 0
    next_t_export = 0
    initial_v = None
    tic = time_mod.perf_counter()
    for i in range(nt+1):
        sync()
        t = i*dt

        if t >= next_t_export - 1e-8:
            elev_max = float(np.max(e, all_axes))
            u_max = float(np.max(u, all_axes))

            total_v = float(np.sum(e + h, all_axes)) * dx * dy
            if initial_v is None:
                initial_v = total_v
            diff_v = total_v - initial_v

            print(f'{i_export:2d} {i:4d} {t:.3f} elev={elev_max:7.5f} '
                  f'u={u_max:7.5f} dV={diff_v: 6.3e}')
            if elev_max > 1e3 or not math.isfinite(elev_max):
                print(f'Invalid elevation value: {elev_max}')
                break
            i_export += 1
            next_t_export = i_export * t_export
            sync()

        step(u, v, e, u1, v1, e1, u2, v2, e2)

    sync()

    duration = time_mod.perf_counter() - tic
    print(f'Duration: {duration:.2f} s')

    e_exact = exact_elev(t, x_t_2d, y_t_2d, lx, ly)
    err2 = (e_exact - e) * (e_exact - e) * dx * dy / lx / ly
    err_L2 = math.sqrt(float(np.sum(err2, all_axes)))
    print(f'L2 error: {err_L2:7.5e}')

    if nx == 128 and ny == 128 and not benchmark_mode:
        assert numpy.allclose(err_L2, 7.22407e-03)
        print('SUCCESS')

    fini()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run wave equation benchmark',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-n', '--resolution', type=int, default=128,
                        help='Number of grid cells in x and y direction.')
    parser.add_argument('-t', '--benchmark-mode', action='store_true',
                        help='Run a fixed number of time steps.')
    parser.add_argument('-b', '--backend', type=str, default='ddpt',
                        choices=['ddpt', 'numpy'],
                        help='Backend to use.')
    args = parser.parse_args()
    run(args.resolution, args.backend, args.benchmark_mode)