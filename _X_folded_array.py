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
import time
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
def ninetybend(wavelen=1.55, n_eff=3.0, length=0.5, alpha=0.0):
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

'Power Combiner: need determine in1 and in2'
def  powercombine(ampl_1=0.2, ampl_2=0.1, phace_1=1, phace_2=1, wavelen=1.55, length=0.5, n_eff=3.0):
    ampl = npj.sqrt(npj.square(ampl_1) + npj.square(ampl_2) + 2*ampl_1*ampl_2*npj.exp(1.j*(phace_1-phace_2)))
    phase_combine = npj.arctan((ampl_1*npj.sin(phace_1)+ampl_2*npj.sin(phace_2)) / (ampl_1*npj.cos(phace_1)+ampl_2*npj.cos(phace_2)))
    phase_length = 2*npj.pi*length*n_eff/wavelen
    power_out = npj.square(ampl * npj.exp(1.j*(phase_combine+phase_length)))
    sdict = sax.reciprocal(
        {
            ("in0", "out0"): npj.sqrt(power_out/(npj.square(ampl_1))),
            ("in1", "out0"): npj.sqrt(power_out/(npj.square(ampl_2))),
        }
        )
    return sdict


num_stacks = int(input("The number of stacks: "))


start = time.time()


'The first part of grating array, including coupler, taper, 90-degree bend and sbend'
coupler_model,info1 = sax.circuit(
        netlist={
            "instances": {
                "cp": "coupler",
                "tp": "taper",
                "nb": "ninetybend",
                "sb": "sbend"
            },
            "connections": {
                "cp,out0": "tp,in0",
                "tp,out0": "nb,in0",
                "nb,out0": "sb,in0"
            },
            "ports": {
                "in0": "cp,in0",
                "out0": "sb,out0",
            },
        },
        models={
            "coupler": coupler,
            "taper": taper,
            "ninetybend": ninetybend,
            "sbend": sbend
        },
    )


'X folded grating array'
_X_folded_array, info = sax.circuit(
    netlist={
        "instances": {
            **{f"cm{i}": "coupler_model" for i in range(num_stacks)},
            **{f"pc{i}": "powercombiner" for i in range(num_stacks)},
            **{f"sb{i}": "sbend" for i in range(num_stacks-1)},
        },
        "connections": {
            "cm0,out0": "pc0,in1",
            **{f"cm{i+1},out0": f"pc{i},in0" for i in range(num_stacks - 1)},
            **{f"pc{i},out0": f"sb{i},in0" for i in range(num_stacks - 1)},
            **{f"sb{i},out0": f"pc{i+1},in1" for i in range(num_stacks - 1)},
        },
        "ports": {
            **{f"in{i}": f"cm{i},in0" for i in range(num_stacks)},
            "out0": "cm0,out0",
            **{f"out{i+1}": f"sb{i},out0" for i in range(num_stacks - 1)},
            **{f"out{i+num_stacks}": f"cm{i+1},out0" for i in range(num_stacks - 1)}
        },
    },
    models={
        "coupler_model": coupler_model,
        "powercombiner": powercombine,
        "sbend": sbend,
    },
)

theta = npj.linspace(0, 90, 100)





"Determine input wave to each power combiner"
"Initial setting"
_X_folded_array_theta = _X_folded_array(
                            **{f"cm{i}": {'cp': {'theta': theta}} for i in range (num_stacks)},
                            )


"Loop to modify every power combiner"


ampl1_pc = dict()
phace1_pc = dict()
ampl2_pc = dict()
phace2_pc = dict()
ctr = 1

for i in range (num_stacks - 1):
    ampl1_pc[i] = npj.abs(_X_folded_array_theta[('in0','out'+str(i))])/((i+1)**2)
    phace1_pc[i] =  npj.angle(_X_folded_array_theta[('in0','out'+str(i))])
    ampl2_pc[i] = npj.abs(_X_folded_array_theta[('in'+str(i+1),'out'+str(i+num_stacks))])
    phace2_pc[i] = npj.angle(_X_folded_array_theta[('in'+str(i+1),'out'+str(i+num_stacks))])
    _X_folded_array_theta = _X_folded_array(
                                **{f"cm{j}": {'cp': {'theta': theta}} for j in range (num_stacks)},
                                **{f'pc{k}': {'ampl_1': ampl1_pc[i], 'phace_1': phace1_pc[i],
                                              'ampl_2': ampl2_pc[i], 'phace_2': phace2_pc[i]}
                                               for k in range (ctr)
                                   }
                                )
    ctr += 1




_X_folded_array_S11 = npj.abs(_X_folded_array_theta[('in0', 'out'+str(num_stacks-1))])
_X_folded_array_S13 = npj.abs(_X_folded_array_theta[('in2', 'out'+str(num_stacks-1))])

fig, ax1 = plt.subplots(1)
'ax2 = ax1.twinx()'
ax1.plot(theta, _X_folded_array_S11, color="red", label="S11")
ax1.set_ylabel(r"Transmission Effciency")
ax1.set_xlabel(r"Angle [deg]")
fig.suptitle("X-coupler Folded Array")
"""
ax2.plot(theta, mzi_array_S12, color="red", label="S13")
"""
plt.legend()
plt.show()

end= time.time()
print(f"time: {end - start} s")

