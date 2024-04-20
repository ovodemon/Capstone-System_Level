#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:24:40 2024

@author: zhouy24
"""

from scipy.optimize import minimize, rosen, rosen_der
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




def transmission_1(sets):
    """
    Parameters
    sets : array_like
                Array(theta, s-bend length)
                
    Returns
            transmission effeciency (output/input_1)
    """
    
    "Determine input wave to each power combiner"
    "Initial setting"
    setting = sax.get_settings(_X_folded_array)
    for i in range (num_stacks):
        setting[f'cm{i}']['cp']['theta']= sets[0]
        setting[f'cm{i}']['sb']['length']=sets[1]
        setting[f'cm{i}']['nb']['length']=sets[2]
    for k in range (num_stacks -1):
        setting[f'sb{k}']['length']=sets[1]
    _X_folded_array_theta = _X_folded_array(**setting)
    
    "Loop to modify every power combiner"
    ampl1_pc = dict()
    phace1_pc = dict()
    ampl2_pc = dict()
    phace2_pc = dict()
    
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
                    
        _X_folded_array_theta = _X_folded_array(**setting)
    
    te = npj.abs(_X_folded_array_theta[('in0', 'out'+str(num_stacks-1))])

    return te

def plot_te(sets):
    theta = npj.linspace(0, 90, 100)
    "Determine input wave to each power combiner"
    "Initial setting"
    setting = sax.get_settings(_X_folded_array)
    for i in range (num_stacks):
        setting[f'cm{i}']['cp']['theta']=theta
        setting[f'cm{i}']['sb']['length']=sets[1]
        setting[f'cm{i}']['nb']['length']=sets[2]
    for k in range (num_stacks -1):
        setting[f'sb{k}']['length']=sets[1]
    _X_folded_array_theta = _X_folded_array(**setting)

    "Loop to modify every power combiner"
    ampl1_pc = dict()
    phace1_pc = dict()
    ampl2_pc = dict()
    phace2_pc = dict()
    ctr = 1

    for i in range (num_stacks - 1):
        ampl1_pc[i] = npj.abs(_X_folded_array_theta[('in0','out'+str(i))]) / (i+1)**2
        phace1_pc[i] = npj.angle(_X_folded_array_theta[('in0','out'+str(i))])
        ampl2_pc[i] = npj.abs(_X_folded_array_theta[('in'+str(i+1),'out'+str(i+num_stacks))])
        phace2_pc[i] = npj.angle(_X_folded_array_theta[('in'+str(i+1),'out'+str(i+num_stacks))])
        for k in range (ctr):
            setting[f'pc{k}']['ampl_1']=ampl1_pc[i]
            setting[f'pc{k}']['phace_1']=phace1_pc[i]
            setting[f'pc{k}']['ampl_2']=ampl2_pc[i]
            setting[f'pc{k}']['phace_2']=phace2_pc[i]
        
        _X_folded_array_theta = _X_folded_array(**setting)
        ctr += 1
    _X_folded_array_S21 = npj.abs(_X_folded_array_theta[('in0', 'out'+str(num_stacks-1))])
    
    "Calculate FOV."
    fov_theta=[]
    for i in range(len(_X_folded_array_S21)):
        if _X_folded_array_S21[i] >= 0.45:
            fov_theta.append(theta[i])
    if len(fov_theta)>=1:
        fov=fov_theta[-1]-fov_theta[0]
        print(f"FOV={fov}")
    else: print("FOV=0")
    
    fig, ax1 = plt.subplots(1)
    'ax2 = ax1.twinx()'
    ax1.plot(theta, _X_folded_array_S21, color="red", label="S11")
    ax1.set_ylabel(r"Transmission Effciency")
    ax1.set_xlabel(r"Theta")
    fig.suptitle("X-coupler Folded Array")
    plt.legend()
    plt.show()
    return



def loss_1(sets):
    return 1-transmission_1(sets)


sets = [20.36036036, 1.6,0.7]
x0 = [21.6, 1.7,0.7]

loss_1_nelder = minimize(loss_1, x0, method='nelder-mead',options={'xatol': 1e-8, 'disp': True})
end= time.time()
print(f"Nelder-mead's time: {end - start} s\r")
start = time.time()
loss_1_BFGS = minimize(loss_1, x0, method='BFGS', options={'gtol': 1e-8,'disp': True})
end= time.time()
print(f"BFGS's time: {end - start} s\r")
    
