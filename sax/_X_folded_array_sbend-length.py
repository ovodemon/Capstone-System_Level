#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:19:24 2024

@author: zhouy24
"""

import time
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
length = npj.linspace(1.5, 2.1, 100)


"Determine input wave to each power combiner"
"Initial setting"
setting = sax.get_settings(_X_folded_array)
for i in range (num_stacks):
    setting[f'cm{i}']['cp']['theta']=theta
    
_X_folded_array_theta = _X_folded_array(**setting)


"Loop to modify every power combiner"
ampl1_pc = dict()
phace1_pc = dict()
ampl2_pc = dict()
phace2_pc = dict()
_X_folded_array_S11 = []
theta_max = []

for a in range (len(length)):
    for i in range (num_stacks - 1):
        ampl1_pc[i] = npj.abs(_X_folded_array_theta[('in0','out'+str(i))]) / (i+1)**2
        phace1_pc[i] =  npj.angle(_X_folded_array_theta[('in0','out'+str(i))])
        ampl2_pc[i] = npj.abs(_X_folded_array_theta[('in'+str(i+1),'out'+str(i+num_stacks))])
        phace2_pc[i] = npj.angle(_X_folded_array_theta[('in'+str(i+1),'out'+str(i+num_stacks))])
        for j in range (i+1):
            setting[f'pc{j}']['ampl_1']=ampl1_pc[i]
            setting[f'pc{j}']['phace_1']=phace1_pc[i]
            setting[f'pc{j}']['ampl_2']=ampl2_pc[i]
            setting[f'pc{j}']['phace_2']=phace2_pc[i]
        for k in range (num_stacks -1):
            setting[f'sb{k}']['length']=length[a]
                    
        _X_folded_array_theta = _X_folded_array(**setting)
        
    te = npj.abs(_X_folded_array_theta[('in0', 'out'+str(num_stacks-1))])
    peak = te.max()
    theta_max.append(int(npj.where(te == peak)[0][0]) * 90/100)
    _X_folded_array_S11.append(float(peak))

'''
_X_folded_array_S11 = npj.abs(_X_folded_array_theta[('in0', 'out'+str(num_stacks-1))])
_X_folded_array_S13 = npj.abs(_X_folded_array_theta[('in2', 'out'+str(num_stacks-1))])
'''


fig, ax1 = plt.subplots(1)
'ax2 = ax1.twinx()'
ax1.plot(length, _X_folded_array_S11, color="red", label="S11")
ax1.set_ylabel(r"Transmission Effciency")
ax1.set_xlabel(r"Length")
fig.suptitle("X-coupler Folded Array")
"""
ax2.plot(theta, mzi_array_S12, color="red", label="S13")
"""
plt.legend()
plt.show()

end= time.time()
print(f"time: {end - start} s")
