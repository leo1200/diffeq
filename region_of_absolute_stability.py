
import numpy as np
import matplotlib.pyplot as plt

# ================== Set nice plots ====================
import matplotlib
SMALL_SIZE = 11
MEDIUM_SIZE = 13
BIGGER_SIZE = 14

# General plot parameters
matplotlib.rcParams['font.family'] = 'Avenir'

matplotlib.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
matplotlib.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
matplotlib.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
matplotlib.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Set the colormap for the histogram plot
# cmap = cm.tab20(np.linspace(0, 1, 12))
# hierarchy.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])

# Fix the random seed for reproducibility
np.random.seed(0)


def explicit_euler_stability():
    # Define the range of values for the real and imaginary parts of z
    real_vals = np.linspace(-3, 2, 400)
    imag_vals = np.linspace(-2, 2, 400)

    # Create a meshgrid from the real and imaginary parts
    real, imag = np.meshgrid(real_vals, imag_vals)

    # Calculate the absolute values of z
    z = real + 1j * imag

    # Compute the stability function for the explicit Euler scheme
    stability = np.abs(1 + z)

    # Plot the stability diagram
    plt.figure(figsize=(10, 8))
    # Plot the region of absolute stability where |R(z)| <= 1
    plt.contourf(real, imag, stability, levels=[0, 1], colors=["#16a085"])
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    # plt.title("Region of Absolute Stability of the Explicit Euler Scheme")
    # Plot the x and y axes
    plt.axhline(y=0, color="k")
    plt.axvline(x=0, color="k")
    # set axis ratio to 1:1
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig("figures/explicit_euler_stability_region.svg")

def implicit_euler_stability():
    # Define the range of values for the real and imaginary parts of z
    real_vals = np.linspace(-3, 2, 400)
    imag_vals = np.linspace(-2, 2, 400)

    # Create a meshgrid from the real and imaginary parts
    real, imag = np.meshgrid(real_vals, imag_vals)

    # Calculate the absolute values of z
    z = real + 1j * imag

    # Compute the stability function for the explicit Euler scheme
    stability = np.abs(1 / (1 - z))

    # Plot the stability diagram
    plt.figure(figsize=(10, 8))
    # Plot the region of absolute stability where |R(z)| <= 1
    plt.contourf(real, imag, stability, levels=[0, 1], colors=["#16a085"])
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    # plt.title("Region of Absolute Stability of the Explicit Euler Scheme")
    # Plot the x and y axes
    plt.axhline(y=0, color="k")
    plt.axvline(x=0, color="k")
    # set axis ratio to 1:1
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig("figures/implicit_euler_stability_region.svg")

# Call the function to generate the stability diagram
implicit_euler_stability()