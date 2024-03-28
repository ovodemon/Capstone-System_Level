#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:19:24 2024

@author: zhouy24
"""

import jax
import jax.example_libraries.optimizers as opt
import jax.numpy as npj
import matplotlib.pyplot as plt  # plotting
import sax
from tqdm.notebook import trange



'Coupler'
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

def coupling_efficiency(theta, wlen=1.55, length=15, pitch=0.85, D=0.5, n_1=1.44, n_2=3.48, eta_max=0.5, dtheta_max=6.25, beta=22):
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

'Taper'
def taper(a=0.4, b=0.5, c=7, z=0.9, wavelen=1.55, n_eff=3.0, length=0.3, alpha=0.0):
    X = a*(b*npj.square(z) + (1-b)*z) + (1-a)*npj.square(npj.sin(c*z*npj.pi/2))
    phase = 2*npj.pi*length*n_eff/wavelen
    transmission = (-X)*npj.exp(-1.j*phase)
    sdict = sax.reciprocal(
        {
            ("in0", "out0"): transmission
        }
    )
    return sdict

'90 degree bend'
def ninetybend(wavelen=1.55, n_eff=3.0, length=0.2, alpha=0.0):
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

'S bend'
def sbend(wavelen=1.55, n_eff=3.0, length=1, alpha=0.0):
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
'''
'Power Combiner: need determine in1 and in2'
def  powercombine(ampl_1=0.2, ampl_2=0.1, phace_1=1, phace_2=1, wavelen=1.55, length=0.5, n_eff=3.0):
    ampl = npj.sqrt(npj.square(ampl_1) + npj.square(ampl_2) + 2*ampl_1*ampl_2*npj.exp(1.j*(phace_1-phace_2)))
    phase_combine = npj.arctan((ampl_1*npj.sin(phace_1)+ampl_2*npj.sin(phace_2)) / (ampl_1*npj.cos(phace_1)+ampl_2*npj.cos(phace_2)))
    phase_length = 2*npj.pi*length*n_eff/wavelen
    power_out = npj.square(ampl * npj.exp(1.j*(phase_combine+phase_length)))
    sdict = sax.reciprocal(
        {
            ("in0", "out0"): -npj.sqrt(power_out/(npj.square(ampl_1))),
            ("in1", "out0"): -npj.sqrt(power_out/(npj.square(ampl_2)))
        }
        )
    return sdict
'''
def  powercombine():
    sdict = sax.reciprocal(
        {
            ("in0", "out0"): 0.99,
            ("in1", "out0"): 0.99
        }
        )
    return sdict


num_stacks = 2



'The first part of grating array, including coupler, taper and 90-degree bend'
coupler_model,info1 = sax.circuit(
        netlist={
            "instances": {
                "cp": "coupler",
                "tp": "taper",
                "nb": "ninetybend",
            },
            "connections": {
                "cp,out0": "tp,in0",
                "tp,out0": "nb,in0",
            },
            "ports": {
                "in0": "cp,in0",
                "out0": "nb,out0",
            },
        },
        models={
            "coupler": coupler,
            "taper": taper,
            "ninetybend": ninetybend,
        },
    )

'''The rest parts of grating array. Each part include coupler, taper, 90-degree bend,
two s-bends and power combiner'''
rest_model, info2 = sax.circuit(
        netlist={
            "instances": {
                "cp": "coupler",
                "tp": "taper",
                "nb": "ninetybend",
                "sb01": "sbend",
                "sb02": "sbend",
                "pc": "powercombiner",
            },
            "connections": {
                "cp,out0": "tp,in0",
                "tp,out0": "nb,in0",
                "nb,out0": "sb01,in0",
                "sb01,out0": "pc,in0",
                "sb02,out0": "pc,in1",
            },
            "ports": {
                "in0": "cp,in0",
                "in1": "sb02,in0",
                "out0": "nb,out0",
            },
        },
        models={
            "coupler": coupler,
            "taper": taper,
            "ninetybend": ninetybend,
            "sbend": sbend,
            "powercombiner": powercombine,
        },
    )




'X folded grating array'
_X_folded_array, info = sax.circuit(
    netlist={
        "instances": {
            "cm": "coupler_model",
            **{f"rm{i}": "rest_model" for i in range(num_stacks)},
        },
        "connections": {
            "cm,out0": f"rm0,in1",
            **{f"rm{i},out0": f"rm{i+1},in1" for i in range(num_stacks - 1)},
        },
        "ports": {
            "in0": "cm,in0",
            **{f"in{i+1}": f"rm{i},in0" for i in range(num_stacks - 1)},
            "out0": f"rm{num_stacks-2},out0",
        },
    },
    models={
        "coupler_model": coupler_model,
        "rest_model": rest_model,
    },
)

theta = npj.linspace(0, 90, 100)




'''
"Determine input wave to each power combiner"
mzi_array_theta = _4_folded_array(
                        first_coupler={"theta": theta},
                        second_coupler={"theta": theta},
                        third_coupler={"theta": theta},
                        fourth_coupler={"theta": theta})
"Determine input wave to the first power combiner"
pc1_ampl1 = npj.abs(mzi_array_theta[('in0', 'out1')])
pc1_phace1 = npj.angle(mzi_array_theta[('in0', 'out1')])
pc1_ampl2 = npj.abs(mzi_array_theta[('in1', 'out2')])
pc1_phace2 = npj.angle(mzi_array_theta[('in1', 'out2')])
mzi_array_theta = _4_folded_array(
                        first_coupler={"theta": theta},
                        second_coupler={"theta": theta},
                        powercombine1={"ampl_1": pc1_ampl1, "ampl_2": pc1_ampl2,
                                       "phace_1": pc1_phace1, "phace_2": pc1_phace2},
                        third_coupler={"theta": theta},
                        fourth_coupler={"theta": theta})

"Determine input wave to the second power combiner"
pc2_ampl1 = npj.abs(mzi_array_theta[('in0', 'out3')])/2
pc2_phace1 = npj.angle(mzi_array_theta[('in0', 'out3')])
pc2_ampl2 = npj.abs(mzi_array_theta[('in2', 'out4')])
pc2_phace2 = npj.angle(mzi_array_theta[('in2', 'out4')])
mzi_array_theta = _4_folded_array(
                        first_coupler={"theta": theta},
                        second_coupler={"theta": theta},
                        powercombine1={"ampl_1": pc1_ampl1, "ampl_2": pc1_ampl2,
                                       "phace_1": pc1_phace1, "phace_2": pc1_phace2},
                        third_coupler={"theta": theta},
                        powercombine2={"ampl_1": pc2_ampl1, "ampl_2": pc2_ampl2,
                                       "phace_1": pc2_phace1, "phace_2": pc2_phace2},
                        fourth_coupler={"theta": theta})

"Determine input wave to the third power combiner"
pc3_ampl1 = npj.abs(mzi_array_theta[('in0', 'out5')])/3
pc3_phace1 = npj.angle(mzi_array_theta[('in0', 'out5')])
pc3_ampl2 = npj.abs(mzi_array_theta[('in3', 'out6')])
pc3_phace2 = npj.angle(mzi_array_theta[('in3', 'out6')])
mzi_array_theta = _4_folded_array(
                        first_coupler={"theta": theta},
                        second_coupler={"theta": theta},
                        powercombine1={"ampl_1": pc1_ampl1, "ampl_2": pc1_ampl2,
                                       "phace_1": pc1_phace1, "phace_2": pc1_phace2},
                        third_coupler={"theta": theta},
                        powercombine2={"ampl_1": pc2_ampl1, "ampl_2": pc2_ampl2,
                                       "phace_1": pc2_phace1, "phace_2": pc2_phace2},
                        fourth_coupler={"theta": theta},
                        powercombine3={"ampl_1": pc3_ampl1, "ampl_2": pc3_ampl2,
                                       "phace_1": pc3_phace1, "phace_2": pc3_phace2})


mzi_array_S11 = npj.abs(mzi_array_theta[('in0', 'out0')])
mzi_array_S12 = npj.abs(mzi_array_theta[('in2', 'out0')])

fig, ax1 = plt.subplots(1)
'ax2 = ax1.twinx()'
ax1.plot(theta, mzi_array_S11, color="red", label="S11")
ax1.set_ylabel(r"Transmission Effciency")
ax1.set_xlabel(r"Angle [deg]")
fig.suptitle("4-coupler Folded Array")
"""
ax2.plot(theta, mzi_array_S12, color="red", label="S13")
"""
plt.legend()
plt.show()

'''

