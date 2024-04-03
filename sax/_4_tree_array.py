#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 19:34:43 2024

@author: zhouy24
"""

import jax
import jax.example_libraries.optimizers as opt
import jax.numpy as npj
import matplotlib.pyplot as plt  # plotting
import sax
from tqdm.notebook import trange
from model_draft import *


'4 folded grating array'
_4_tree_array, info = sax.circuit(
    netlist={
        "instances": {
            "first_coupler": "coupler",
            "first_taper": "taper",
            "first_sbend": "sbend",
            "second_coupler": "coupler",
            "second_taper": "taper",
            "second_sbend": "sbend",
            "third_coupler": "coupler",
            "third_taper": "taper",
            "third_sbend": "sbend",
            "fourth_coupler": "coupler",
            "fourth_taper": "taper",
            "fourth_sbend": "sbend",
            "powercombine1": "powercombine",
            "pc1_sbend": "sbend",
            "powercombine2": "powercombine",
            "pc2_sbend": "sbend",
            "powercombine3": "powercombine",
        },
        "connections": {
            "first_coupler,out0": "first_taper,in0",
            "first_taper,out0": "first_sbend,in0",
            "second_coupler,out0": "second_taper,in0",
            "second_taper,out0": "second_sbend,in0",
            "first_sbend,out0": "powercombine1,in0",
            "second_sbend,out0": "powercombine1,in1",
            "powercombine1,out0": "pc1_sbend,in0",
            "third_coupler,out0": "third_taper,in0",
            "third_taper,out0": "third_sbend,in0",
            "fourth_coupler,out0": "fourth_taper,in0",
            "fourth_taper,out0": "fourth_sbend,in0",
            "third_sbend,out0": "powercombine2,in0",
            "fourth_sbend,out0": "powercombine2,in1",
            "powercombine2,out0": "pc2_sbend,in0",
            "pc1_sbend,out0": "powercombine3,in0",
            "pc2_sbend,out0": "powercombine3,in1",
        },
        "ports": {
            "in0": "first_coupler,in0",
            "in1": "second_coupler,in0",
            "in2": "third_coupler,in0",
            "in3": "fourth_coupler,in0",
            "out0": "powercombine3,out0",
            "out1": "first_sbend,out0",
            "out2": "second_sbend,out0",
            "out3": "pc1_sbend,out0",
            "out4": "third_sbend,out0",
            "out5": "fourth_sbend,out0",
            "out6": "pc2_sbend,out0",
        },
    },
    models={
        "coupler": coupler,
        "taper": taper,
        "sbend": sbend,
        "powercombine": powercombine,
    },
)

theta = npj.linspace(0, 90, 100)

"Determine input wave to each power combiner"
mzi_array_theta = _4_tree_array(
                        first_coupler={"theta": theta},
                        second_coupler={"theta": theta},
                        third_coupler={"theta": theta},
                        fourth_coupler={"theta": theta})

"Determine input wave to the first power combiner"
pc1_ampl1 = npj.abs(mzi_array_theta[('in0', 'out1')])
pc1_phace1 = npj.angle(mzi_array_theta[('in0', 'out1')])
pc1_ampl2 = npj.abs(mzi_array_theta[('in1', 'out2')])
pc1_phace2 = npj.angle(mzi_array_theta[('in1', 'out2')])

"Determine input wave to the second power combiner"
pc2_ampl1 = npj.abs(mzi_array_theta[('in2', 'out4')])
pc2_phace1 = npj.angle(mzi_array_theta[('in2', 'out4')])
pc2_ampl2 = npj.abs(mzi_array_theta[('in3', 'out5')])
pc2_phace2 = npj.angle(mzi_array_theta[('in3', 'out5')])

mzi_array_theta = _4_tree_array(
                        first_coupler={"theta": theta},
                        second_coupler={"theta": theta},
                        powercombine1={"ampl_1": pc1_ampl1, "ampl_2": pc1_ampl2,
                                       "phace_1": pc1_phace1, "phace_2": pc1_phace2},
                        third_coupler={"theta": theta},
                        powercombine2={"ampl_1": pc2_ampl1, "ampl_2": pc2_ampl2,
                                       "phace_1": pc2_phace1, "phace_2": pc2_phace2},
                        fourth_coupler={"theta": theta})

"Determine input wave to the third power combiner"
pc3_ampl1 = npj.abs(mzi_array_theta[('in0', 'out3')])/2
pc3_phace1 = npj.angle(mzi_array_theta[('in0', 'out3')])
pc3_ampl2 = npj.abs(mzi_array_theta[('in2', 'out6')])/2
pc3_phace2 = npj.angle(mzi_array_theta[('in2', 'out6')])
mzi_array_theta = _4_tree_array(
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
fig.suptitle("4-coupler Tree Array")
"""
ax2.plot(theta, mzi_array_S12, color="red", label="S13")
"""
plt.legend()
plt.show()
