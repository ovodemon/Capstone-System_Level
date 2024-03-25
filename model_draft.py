#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 14:07:56 2024

@author: zhouy24
"""

import jax
import jax.example_libraries.optimizers as opt
import jax.numpy as npj
import matplotlib.pyplot as plt  # plotting
import sax
from tqdm.notebook import trange


def coupling_efficiency(theta, eta_0=0.3, theta_1=10, dtheta=2.5):
    eta = eta_0*npj.exp(-npj.square(theta-theta_1)*npj.log(2)/npj.square(dtheta))
    return eta


def taper(a=0.4, b=0.5, c=7, z=0.9, wavelen=1.55, n_eff=3.0, length=0.3, alpha=0.0):
    X = a*(b*npj.square(z) + (1-b)*z) + (1-a)*npj.square(npj.sin(c*z*npj.pi/2))
    phase = 2*npj.pi*length*n_eff/wavelen
    transmission = X*npj.exp(1.j*phase)
    sdict = sax.reciprocal(
        {
            ("in0", "out0"): transmission
        }
    )
    return sdict







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



def  power_combiner(ampl_1=0.2, ampl_2=0.03, phase_1=1, phase_2=1, wavelen=1.55, length=0.5, n_eff=3.0):
    ampl = npj.sqrt(npj.square(ampl_1) + npj.square(ampl_2) + 2*ampl_1*ampl_2*npj.exp(1.j*(phase_1-phase_2)))
    phase_combine = npj.arctan((ampl_1*npj.sin(phase_1)+ampl_2*npj.sin(phase_2)) / (ampl_1*npj.cos(phase_1)+ampl_2*npj.cos(phase_2)))
    phase_length = 2*npj.pi*length*n_eff/wavelen
    power_out = npj.square(ampl * npj.exp(1.j*(phase_combine+phase_length)))
    sdict = sax.reciprocal(
        {
            ("in0", "out0"): power_out/(npj.square(ampl_1)),
            ("in1", "out0"): power_out/(npj.square(ampl_2))
        }
        )
    return sdict

'4 folded grating array'
_4_folded_array, info = sax.circuit(
    netlist={
        "instances": {
            "first_coupler": "coupler",
            "first_taper": "taper",
            "first_ninetybend": "ninetybend",
            "first_sbend": "sbend",
            "second_coupler": "coupler",
            "second_taper": "taper",
            "second_ninetybend": "ninetybend",
            "second_sbend": "sbend",
            "third_coupler": "coupler",
            "third_taper": "taper",
            "third_ninetybend": "ninetybend",
            "third_sbend": "sbend",
            "fourth_coupler": "coupler",
            "fourth_taper": "taper",
            "fourth_ninetybend": "ninetybend",
            "fourth_sbend": "sbend",
            "powercombine1": "powercombine",
            "pc1_sbend": "sbend",
            "powercombine2": "powercombine",
            "pc2_sbend": "sbend",
            "powercombine3": "powercombine",
        },
        "connections": {
            "first_coupler,out0": "first_taper,in0",
            "first_taper,out0": "first_ninetybend,in0",
            "first_ninetybend,out0": "first_sbend,in0",
            "second_coupler,out0": "second_taper,in0",
            "second_taper,out0": "second_ninetybend,in0",
            "second_ninetybend,out0": "second_sbend,in0",
            "first_sbend,out0": "powercombine1,in0",
            "second_sbend,out0": "powercombine1,in1",
            "powercombine1,out0": "pc1_sbend,in0",
            "third_coupler,out0": "third_taper,in0",
            "third_taper,out0": "third_ninetybend,in0",
            "third_ninetybend,out0": "third_sbend,in0",
            "third_sbend,out0": "powercombine2,in0",
            "pc1_sbend,out0": "powercombine2,in1",
            "powercombine2,out0": "pc2_sbend,in0",
            "fourth_coupler,out0": "fourth_taper,in0",
            "fourth_taper,out0": "fourth_ninetybend,in0",
            "fourth_ninetybend,out0": "fourth_sbend,in0",
            "fourth_sbend,out0": "powercombine3,in0",
            "pc2_sbend,out0": "powercombine3,in1",
        },
        "ports": {
            "in0": "first_coupler,in0",
            "in1": "second_coupler,in0",
            "in2": "third_coupler,in0",
            "in3": "fourth_coupler,in0",
            "out0": "powercombine3,out0",
        },
    },
    models={
        "coupler": coupler,
        "taper": taper,
        "ninetybend": ninetybend,
        "sbend": sbend,
        "powercombine": powercombine,
    },
)



theta = npj.linspace(0, 45, 100)

mzi_array_theta = _4_folded_array(
                        first_coupler={"theta": theta},
                        second_coupler={"theta": theta},
                        third_coupler={"theta": theta},
                        fourth_coupler={"theta": theta})
mzi_array_S11 = npj.abs(mzi_array_theta[('in0', 'out0')])
mzi_array_S12 = npj.abs(mzi_array_theta[('in3', 'out0')])

fig, ax1 = plt.subplots(1)
ax2 = ax1.twinx()
ax1.plot(theta, mzi_array_S11, color="blue", label="S11")
ax2.plot(theta, mzi_array_S12, color="red", label="S13")
plt.legend()
plt.show()



