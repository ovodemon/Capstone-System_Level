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
from model_draft import *


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

