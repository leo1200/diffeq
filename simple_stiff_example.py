import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ============================= General plot parameters =============================
SMALL_SIZE = 13
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

matplotlib.use("SVG")
plt.rcParams['svg.fonttype'] = 'none'

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
    plt.title("Explicit Euler Scheme, dt = " + str(dt), pad=20)
    plt.legend()
    plt.savefig("figures/stiff_ex" + str(dt).replace(".","-") + ".svg")
    plt.clf()

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
    plt.title("Implicit Euler Scheme, dt = " + str(dt), pad=20)
    plt.legend()
    plt.savefig("figures/stiff_impl" + str(dt).replace(".","-") + ".svg")
    plt.clf()

if __name__ == "__main__":
    expl_test(dt = 0.0005)
    expl_test(dt = 0.002)
    expl_test(dt = 0.004)

    impl_test(0.01)