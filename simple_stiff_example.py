import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ============================= General plot parameters =============================
SMALL_SIZE = 13
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

# matplotlib.use("SVG")
# plt.rcParams['svg.fonttype'] = 'none'

# matplotlib.rcParams['font.family'] = 'Avenir'

matplotlib.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
matplotlib.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
matplotlib.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
matplotlib.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.gcf().subplots_adjust(bottom=0.15)

################### Numerical schemes ###################

def explicit_euler_step(y, f, dt):
    return y + dt * f(y)

def implicit_euler_step_linear_system(y, A, dt):
    return np.linalg.solve(np.eye(len(y)) - dt * A, y)


################### Stepper #############################

def simple_stepper(scheme, y0, f, dt, T):
    """Simple stepper for ODEs, no error control / no adaptive step size."""
    y = np.zeros((int(T / dt) + 1, len(y0)))
    t = np.arange(0, T + dt, dt)
    y[0] = y0
    for i in range(1, len(y)):
        y[i] = scheme(y[i - 1], f, dt)
    return (t, y)

################### Example stiff ODE ###################

def stiff_example(y):
    A = np.array([[998, 1998], [-999, -1999]])
    return A @ y

def true_solution(t):
    A = np.array([[2, -1], [-1, 1]])
    return A @ np.array([np.exp(-t), np.exp(-1000*t)])

################### Testing & Plotting ###################

# decrease figure size
plt.rcParams["figure.figsize"] = (6, 4.5)

def expl_test(dt = 8 / 1000):
    y_0 = np.array([1, 0]); T = 1
    (t, y) = simple_stepper(explicit_euler_step, y_0, stiff_example, dt, T)

    y_true = true_solution(t).T
    plt.plot(t, y_true[:, 0], label = r"true $y_1$", linewidth = 3, color = "cornflowerblue")
    plt.plot(t, y_true[:, 1], label = r"true $y_1$", linewidth = 3, color = "orange")
    plt.plot(t, y[:, 0], label = r"$y_1$", linestyle = "dashed", color = "navy")
    plt.plot(t, y[:, 1], label = r"$y_2$", linestyle = "dashed", color = "red")
    plt.xlabel("t in arbitrary units")
    plt.ylabel(r"$y_1, y_2$")
    # plt.title("Solution of a Stiff ODE using the Explicit Euler Scheme, dt = " + str(dt), pad=20)
    plt.title(r"Explicit Euler Scheme, $\Delta t =$ " + str(dt), pad=20)
    plt.legend()
    plt.savefig("figures/stiff_ex" + str(dt).replace(".","-") + ".svg")
    plt.clf()

def expl_test_ax(ax, dt = 8 / 1000, dtstring = ""):
    y_0 = np.array([1, 0]); T = 1
    (t, y) = simple_stepper(explicit_euler_step, y_0, stiff_example, dt, T)

    y_true = true_solution(t).T
    ax.plot(t, y_true[:, 0], label = r"true $y_1$", linewidth = 3, color = "cornflowerblue")
    ax.plot(t, y_true[:, 1], label = r"true $y_1$", linewidth = 3, color = "orange")
    ax.plot(t, y[:, 0], label = r"$y_1$", linestyle = "dashed", color = "navy")
    ax.plot(t, y[:, 1], label = r"$y_2$", linestyle = "dashed", color = "red")
    ax.set_xlabel("t in arbitrary units")
    ax.set_ylabel(r"$y_1, y_2$")
    # plt.title("Solution of a Stiff ODE using the Explicit Euler Scheme, dt = " + str(dt), pad=20)
    ax.set_title(r"$\Delta t = $" + str(dt) + ", so " + dtstring, pad = 20)
    ax.legend()

def impl_test(dt = 8 / 1000):
    y_0 = np.array([1, 0]); T = 1
    A = np.array([[998, 1998], [-999, -1999]])
    (t, y) = simple_stepper(lambda y, f, dt: implicit_euler_step_linear_system(y, A, dt), y_0, stiff_example, dt, T)

    y_true = true_solution(t).T
    plt.plot(t, y_true[:, 0], label = r"true $y_1$", linewidth = 3, color = "cornflowerblue")
    plt.plot(t, y_true[:, 1], label = r" true $y_2$", linewidth = 3, color = "orange")
    plt.plot(t, y[:, 0], label = r"$y_1$", linestyle = "dashed", color = "navy")
    plt.plot(t, y[:, 1], label = r"$y_2$", linestyle = "dashed", color = "red")
    plt.xlabel("t in arbitrary units")
    plt.ylabel(r"$y_1, y_2$")
    # plt.title("Solution of a Stiff ODE using the Implicit Euler Scheme, dt = " + str(dt), pad=20)
    plt.title(r"Implicit Euler Scheme, $\Delta t = $" + str(dt), pad=20)
    plt.legend()
    plt.savefig("figures/stiff_impl" + str(dt).replace(".","-") + ".svg")
    plt.clf()

def nearby_solutions():
    T = 1; dt = 4 / 1000
    A = np.array([[998, 1998], [-999, -1999]])
    for i in range(10):
        y_0 = np.array([i, 1])
        (t, y) = simple_stepper(lambda y, f, dt: implicit_euler_step_linear_system(y, A, dt), y_0, stiff_example, dt, T)
        plt.plot(t, y[:, 0], label = r"$y_1$", linewidth = 3, color = "cornflowerblue")
        plt.plot(t, y[:, 1], label = r"$y_2$", linewidth = 3, color = "orange")
    plt.xlabel("t in arbitrary units")
    plt.ylabel(r"$y_1, y_2$")
    # plt.legend()
    # plt.title("Solution of a Stiff ODE using the Implicit Euler Scheme, dt = " + str(dt), pad=20)
    plt.title("Nearby solutions vary rapidly", pad=20)
    
    plt.show()
    plt.clf()

# Plot the solution in 2D and also the vector field of the ODE
def plot_solution_2D(dt = 8 / 1000):

    y_0 = np.array([1, 0]); T = 1
    A = np.array([[998, 1998], [-999, -1999]])
    f = stiff_example
    (t, y) = simple_stepper(lambda y, f, dt: implicit_euler_step_linear_system(y, A, dt), y_0, stiff_example, dt, T)


    plt.plot(y[:, 0], y[:, 1], label = "solution", linewidth = 3, color = "cornflowerblue")
    plt.xlabel(r"$y_1$")
    plt.ylabel(r"$y_2$")
    plt.title(r"Solution of a Stiff ODE using the Implicit Euler Scheme, $ \Delta t = $" + str(dt), pad=20)
    plt.legend()

    # plot vector field
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    U = np.zeros(X.shape)
    V = np.zeros(Y.shape)
    for i in range(len(x)):
        for j in range(len(y)):
            U[i, j] = f(np.array([X[i, j], Y[i, j]]))[0]
            V[i, j] = f(np.array([X[i, j], Y[i, j]]))[1]
    plt.quiver(X, Y, U, V)
    plt.xlabel(r"$y_1$")
    plt.ylabel(r"$y_2$")
    plt.title("Vector field of the ODE", pad=20)
    plt.show()
    plt.clf()


if __name__ == "__main__":
    expl_test(dt = 0.0005)
    expl_test(dt = 0.002)
    expl_test(dt = 0.004)

    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # expl_test_ax(axs[0], dt = 0.0005, dtstring = r"$\Delta t < 1/1000$")
    # expl_test_ax(axs[1], dt = 0.002, dtstring = r"$1/1000 < \Delta t < 2/1000$")
    # expl_test_ax(axs[2], dt = 0.004, dtstring = r"$2/1000 < \Delta t$")
    # plt.tight_layout()
    # plt.savefig("figures/stiff_expl.svg")

    impl_test(0.01)
    # nearby_solutions()
    # plot_solution_2D(dt = 0.01)