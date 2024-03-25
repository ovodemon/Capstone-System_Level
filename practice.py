#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:59:09 2024

@author: zhouy24
"""


import jax
import jax.example_libraries.optimizers as opt
import jax.numpy as npj
import matplotlib.pyplot as plt  # plotting
import sax
from tqdm.notebook import trange

def effective_index(D=0.5, n_1=1.44, n_2=3.48):
    n_eff = D*n_2 + (1-D)*n_1
    return n_eff

def coupling_angle(wlen, pitch, D=0.5, n_1=1.44, n_2=3.48):
    n_eff = effective_index(D, n_1, n_2)
    theta_1 = npj.arcsin((pitch*n_eff - wlen)/n_1)*180/npj.pi
    return theta_1

def eta_naught(length, eta_max=0.5, beta=22):
    eta_0 = eta_max*(1-npj.exp(-length/beta))
    return eta_0

def angular_acceptance(length, dtheta_max=6.25, beta=22):
    dtheta = dtheta_max*npj.exp(-length/beta)
    return dtheta

def coupling_efficiency(theta, wlen=1.55, length=25, pitch=0.9, D=0.5, n_1=1.44, n_2=3.48, eta_max=0.5, dtheta_max=6.25, beta=22):
    theta_1 = coupling_angle(wlen, pitch, D, n_1, n_2)
    eta_0 = eta_naught(length, eta_max, beta)
    dtheta = angular_acceptance(length, dtheta_max, beta)
    eta = eta_0*npj.exp(-npj.square(theta-theta_1)*npj.log(2)/npj.square(dtheta))
    return eta, theta_1

def coupler(theta=10) -> sax.SDict:
    eta, theta_1 = coupling_efficiency(theta)
    coupler_dict = sax.reciprocal(
        {
            ("in0", "out0"): eta,
        }
    )
    return coupler_dict


def waveguide(wavelen=1.55, n_eff=3.0, length=1, alpha=0.0):
    """A simple mode of a waveguide with phase change and loss. All lengthscale units
    are in microns."""
    phase = 2*npj.pi*length*n_eff/wavelen
    transmission = npj.power(10, -(alpha * length)/20) * npj.exp(1.j*phase)
    sdict = sax.reciprocal(
        {
            ("in0", "out0"): transmission
        }
    )
    return sdict


wg_ckt, info = sax.circuit(
    netlist={
        "instances": {
            "lft": "coupler",
            "rgt": "waveguide",
        },
        "connections": {
            "lft,out0": "rgt,in0",
        },
        "ports": {
            "in0": "lft,in0",
            "out0": "rgt,out0",
        },
    },
    models={
        "coupler": coupler,
        "waveguide": waveguide,
    },
)

theta = npj.linspace(0, 45, 100)
ce = coupler(theta)[('in0','out0')]

# Plot the angular-dependent coupling
plt.plot(theta, ce*100)
plt.xlabel("Angle [deg]")
plt.ylabel("Coupling Efficiency [%]")
plt.show()

# Print the CE at the peak and FWHM to verify params
print(f"Peak CE: {ce.max()}")
print(f"At theta=theta_1+dtheta, CE: {ce[npj.argwhere(theta >= (10+2.5))[0][0]]}")



'''
wg_sim_ideal = wg_ckt(
                lft={"theta": 45},
                rgt={"length": lengths})

wg_sim_lossy = wg_ckt(
                lft={"theta": 45},
                rgt={"length": lengths, "alpha": 0.025})

wg_ideal_s21_mag = npj.abs(wg_sim_ideal[('in0', 'out0')])
wg_lossy_s21_mag = npj.abs(wg_sim_lossy[('in0', 'out0')])
wg_ideal_s21_phase = npj.angle(wg_sim_ideal[('in0', 'out0')])
wg_lossy_s21_phase = npj.angle(wg_sim_lossy[('in0', 'out0')])


fig, axes = plt.subplots(2)
axes[0].plot(lengths, wg_ideal_s21_mag, label='Ideal')
axes[0].plot(lengths, wg_lossy_s21_mag, label='Lossy')
axes[1].plot(lengths, wg_ideal_s21_phase, label='Ideal')
axes[1].plot(lengths, wg_lossy_s21_phase, label='Lossy')
axes[0].set_ylabel(r"Transmission $|S_{21}|$ [$\%$]")
axes[1].set_ylabel(r"Transmission $\angle S_{21}$ [rad]")
axes[1].set_xlabel(r"Waveguide Length $[\mu m]$")
fig.suptitle("Waveguide Transmission and Phase Shift")
plt.legend()
plt.show()
'''