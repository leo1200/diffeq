###################### Imports ######################
import numpy as np
import matplotlib.pyplot as plt

###################### RK2 and leapfrog in kick-drift-kick form ######################

def rk2_step(y, f, dt):
    k1 = f(y)
    k2 = f(y + dt*k1)
    return y + dt/2*(k1 + k2)

def euler_step(y, f, dt):
    return y + dt*f(y)

def leapfrog_kdk(r0, v0, f, T, dt):
    N = int(round(T/dt))
    r = np.zeros((N+1, len(r0)))
    v = np.zeros((N+1, len(v0)))
    energy = np.zeros(N+1)
    lmag = np.zeros(N+1)
    time = np.linspace(0, T, N+1)
    r[0] = r0
    v[0] = v0
    energy[0] = grav_energy(r[0], v[0])
    lmag[0] = grav_angular_momentum(r[0], v[0])
    # r[1], v[1] = euler_step(r[0], v[0], dt/2)
    r[1] = r[0] + dt/2*v[0]
    v[1] = v[0] + dt/2*f(r[0])
    energy[1] = grav_energy(r[1], v[1])
    lmag[1] = grav_angular_momentum(r[1], v[1])
    for n in range(1, N):
        v_half = v[n] + dt/2*f(r[n])
        r[n+1] = r[n] + dt*v_half
        v[n+1] = v_half + dt/2*f(r[n+1])
        energy[n+1] = grav_energy(r[n+1], v[n+1])
        lmag[n+1] = grav_angular_momentum(r[n+1], v[n+1])
    return r, v, energy, lmag, time

###################### Gravitational orbit ######################
# G = 1; M = 1

def f_grav(r):
    return -r/np.linalg.norm(r)**3

def f_grav_full(y):
    return np.concatenate((y[2:], f_grav(y[:2])))

def grav_energy(r, v):
    return 1/2*np.linalg.norm(v)**2 - 1/np.linalg.norm(r)

def grav_angular_momentum(r, v):
    # concatenate 0 to get 3D vector
    L = np.cross(np.concatenate((r, [0])), np.concatenate((v, [0])))
    # return norm of 3D vector
    return np.linalg.norm(L)

def grav_orbit_leap(r0, v0, T, dt):
    r, v, energy, lmag, time = leapfrog_kdk(r0, v0, f_grav, T, dt)
    return r, v, energy, lmag, time

def grav_orbit_rk2(r0, v0, T, dt):
    y0 = np.concatenate((r0, v0))
    y = np.zeros((len(y0), int(round(T/dt))+1))
    energy = np.zeros(int(round(T/dt))+1)
    lmag = np.zeros(int(round(T/dt))+1)
    y[:, 0] = y0
    energy[0] = grav_energy(y[:2, 0], y[2:, 0])
    lmag[0] = grav_angular_momentum(y[:2, 0], y[2:, 0])
    time = np.linspace(0, T, int(round(T/dt))+1)
    for n in range(1, len(time)):
        y[:, n] = rk2_step(y[:, n-1], f_grav_full, dt)
        energy[n] = grav_energy(y[:2, n], y[2:, n])
        lmag[n] = grav_angular_momentum(y[:2, n], y[2:, n])
    return y[:2], y[2:], energy, lmag, time

###################### Gravitational orbit plot ######################
# Plot orbit, energy and angular momentum
def plot_orbit(x, y, energy, lmag, time, title):
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    # in the orbit plot encode time by color
    axs[0].scatter(x, y, c=time, cmap='viridis', s=1, rasterized = True)
    axs[0].set_title('Orbit')
    axs[0].set_xlabel('x in arbitrary units')
    axs[0].set_ylabel('y in arbitrary units')
    axs[1].scatter(time, energy, c = time, cmap='viridis', s=1, rasterized = True)
    axs[1].set_title('Energy')
    axs[1].set_xlabel('time in arbitrary units')
    axs[1].set_ylabel('energy in arbitrary units')
    axs[2].scatter(time, lmag, c = time, cmap='viridis', s=1, rasterized = True)
    axs[2].set_title('Angular momentum')
    axs[2].set_xlabel('time in arbitrary units')
    axs[2].set_ylabel('angular momentum in arbitrary units')
    fig.tight_layout()
    # set x and y limits
    axs[1].set_ylim([energy[0]-0.1, energy[0]+0.5])
    axs[2].set_ylim([lmag[0]-0.1, lmag[0]+0.1])
    # fig.suptitle(title)
    plt.savefig('figures/grav_orbit_' + title + '.svg', dpi = 250)
    plt.show()

###################### Main ######################
if __name__ == '__main__':
    r0 = np.array([1, 0])
    v0 = np.array([0, 0.5])
    T = 100
    dt = 0.01
    r, v, energy, lmag, time = grav_orbit_leap(r0, v0, T, dt)
    plot_orbit(r[:, 0], r[:, 1], energy, lmag, time, 'Leapfrog')
    r, v, energy, lmag, time = grav_orbit_rk2(r0, v0, T, dt)
    plot_orbit(r[0], r[1], energy, lmag, time, 'RK2')