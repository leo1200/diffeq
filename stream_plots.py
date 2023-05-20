import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

g = 9.81
l = 0.1

def pendulum(theta, omega):
    return omega, -g/l*np.sin(theta)

def euler_step(theta, omega, dt):
    dtheta, domega = pendulum(theta, omega)
    theta += dt*dtheta
    omega += dt*domega
    return theta, omega

def euler(theta0, omega0, T, dt):
    N = int(round(T/dt))
    theta = np.zeros((N+1, len(theta0)))
    omega = np.zeros((N+1, len(omega0)))
    time = np.linspace(0, T, N+1)
    theta[0] = theta0
    omega[0] = omega0
    for n in range(N):
        theta[n+1], omega[n+1] = euler_step(theta[n], omega[n], dt)
    return theta, omega, time

def leapfrog(theta0, omega0, T, dt):
    N = int(round(T/dt))
    theta = np.zeros((N+1, len(theta0)))
    omega = np.zeros((N+1, len(omega0)))
    time = np.linspace(0, T, N+1)
    theta[0] = theta0
    omega[0] = omega0
    theta[1], omega[1] = euler_step(theta[0], omega[0], dt)
    for n in range(1, N):
        theta[n+1] = theta[n-1] + 2*dt*omega[n]
        omega[n+1] = omega[n-1] + 2*dt*(-g/l*np.sin(theta[n]))
    return theta, omega, time

def leapfrog_kick_drift_kick(theta0, omega0, T, dt):
    # Kick-drift-kick version of leapfrog
    # Half step in first variable
    # ▁v_(n+1/2)=▁v_n+▁a_n  Δt/2  (kick)
    # Full step in second variable
    # ▁s_(n+1)=▁s_n+▁v_(n+1/2) Δt (drift)
    # Half step in second variable
    # ▁v_(n+1)=▁v_(n+1/2)+▁a_(n+1)  Δt/2  (kick)
    # Possibly change timestep here.
    # ▁v_(n+3/2)=▁v_(n+1)+▁a_(n+1)  Δt/2=▁v_(n+1/2)+▁a_(n+1) Δt

    N = int(round(T/dt))
    theta = np.zeros((N+1, len(theta0)))
    omega = np.zeros((N+1, len(omega0)))
    time = np.linspace(0, T, N+1)
    theta[0] = theta0
    omega[0] = omega0
    theta[1], omega[1] = euler_step(theta[0], omega[0], dt/2)
    for n in range(1, N):
        omega_half = omega[n] + dt/2*(-g/l*np.sin(theta[n]))
        theta[n+1] = theta[n] + dt*omega_half
        omega[n+1] = omega_half + dt/2*(-g/l*np.sin(theta[n+1]))
    return theta, omega, time


def td_between_four_points(p1, p2, p3, p4, N):
    # Draw a shape with lines between four points p1, p2, p3, p4
    # with N points on each line
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    x = np.concatenate((np.linspace(x1, x2, N), np.linspace(x2, x3, N), np.linspace(x3, x4, N), np.linspace(x4, x1, N)))
    y = np.concatenate((np.linspace(y1, y2, N), np.linspace(y2, y3, N), np.linspace(y3, y4, N), np.linspace(y4, y1, N)))
    return x, y
    
# Apply Euler's method to pendulum
# starting with a rectangular shape in
# phase space
theta0, omega0 = td_between_four_points((0.5, 0), (0.5, 2), (1, 2), (1, 0), 100)
T = 5
dt = 0.001

theta_sol_leap, omega_sol_leap, time = leapfrog_kick_drift_kick(theta0, omega0, T, dt)
theta_sol_euler, omega_sol_euler, time = euler(theta0, omega0, T, dt)

def plot_phase_space(ax):
    # Make pendulum streamline plot
    theta = np.linspace(-2*np.pi, 2*np.pi, 100)
    omega = np.linspace(-6*np.pi, 6*np.pi, 100)
    Theta, Omega = np.meshgrid(theta, omega)
    dTheta, dOmega = pendulum(Theta, Omega)

    dt = 0.01

    speed = np.sqrt(dTheta**2 + dOmega**2)
    lw = speed / speed.max()

    ax.streamplot(Theta, Omega, dt * dTheta, dt * dOmega, density = 3, linewidth = lw, color = 'lightsteelblue')

def plot_solution(ax, theta_sol, omega_sol, i, time, method = "leapfrog"):  
    # Plot pendulum solution
    if method == "leapfrog":
        col = "crimson"
    else:
        col = "darkblue"
    sp = ax.plot(theta_sol[i, :], omega_sol[i, :], '-', lw = 2, label = method + " solution at t = {:.2f}".format(time[i]), color = col)
    return sp

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def calc_area(theta_sol, omega_sol):
    # Calculate area of solution
    area = np.zeros(len(theta_sol))
    for i in range(len(theta_sol)):
        theta = theta_sol[i, :]
        omega = omega_sol[i, :]
        area[i] = PolyArea(theta, omega)
    return area

def plot_area(ax, time, area, method = "leapfrog"):
    # Plot area of solution
    if method == "leapfrog":
        col = "crimson"
    else:
        col = "darkblue"
    ax.plot(time, area, lw = 2, color = col, label = "method = " + method)
    ax.set_xlabel("time")
    ax.set_ylabel("area")
    ax.legend(loc = "upper left")
    ax.set_title("Shoelace Approximation \n of the Area of the Polygon")


def animate_phase_space(theta_sol_euler, omega_sol_euler, theta_sol_leap, omega_sol_leap, time):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 8), gridspec_kw={'width_ratios': [3, 1]})
    plot_phase_space(ax1)
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel(r"$\omega$")
    ax1.set_title("Pendulum Streamline Plot")
    ax1.legend(loc = "upper left")

    sp1 = plot_solution(ax1, theta_sol_euler, omega_sol_euler, 0, time, method = "euler")
    sp2 = plot_solution(ax1, theta_sol_leap, omega_sol_leap, 0, time, method = "leapfrog")


    plot_area(ax2, time, calc_area(theta_sol_euler, omega_sol_euler), method = "euler")
    plot_area(ax2, time, calc_area(theta_sol_leap, omega_sol_leap), method = "leapfrog")
    # Plot vertical line at current time
    ax2.axvline(time[0], color = "k", ls = "--", lw = 1)
    
    def animate(i):
        # remove previous solution
        sp1[0].remove()
        sp2[0].remove()

        # plot new solution
        sp1[0] = plot_solution(ax1, theta_sol_euler, omega_sol_euler, i, time, method = "euler")[0]
        sp2[0] = plot_solution(ax1, theta_sol_leap, omega_sol_leap, i, time, method = "leapfrog")[0]

        ax1.legend()

        # Remove old line and plot new
        ax2.clear()
        plot_area(ax2, time, calc_area(theta_sol_euler, omega_sol_euler), method = "euler")
        plot_area(ax2, time, calc_area(theta_sol_leap, omega_sol_leap), method = "leapfrog")
        # Plot vertical line at current time
        ax2.axvline(time[i], color = "k", ls = "--", lw = 1)
        print("Frame " + str(i) + " of " + str(len(time)) + " ✔️")


    anim = FuncAnimation(fig, animate, frames = range(0, len(time), 10), interval = 0.01)
    # save animation as gif
    anim.save("figures/anims/phase_space.gif", fps = 30, dpi = 300)
    # plt.show()

animate_phase_space(theta_sol_euler, omega_sol_euler, theta_sol_leap, omega_sol_leap, time)