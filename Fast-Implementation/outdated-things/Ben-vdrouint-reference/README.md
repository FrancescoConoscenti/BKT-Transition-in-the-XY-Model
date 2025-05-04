main scripts:

* 1_simulate_observables.py: full simulation + creation of observables plots.
* 2_visualize_vortices.py: only thermalization + creation of L=16 snapshot plots for 3 temperatures

helper scripts:

* all_data_process.py: helper
* functions_mcstep.py: helper
* mcpt-xy.py: simulation; used in '1_simulate_observables.py'
* mcpt-thermalize.py: same as 'mcpt-xy.py' but ONLY thermalization (much faster); used in '2_visualize_vortices.py'.