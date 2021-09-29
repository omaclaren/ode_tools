from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import copy
import ode_tools.ode_lib as ode_lib

N =  5e6 # T
I0 = 1./N
S0 = 1- I0
R0 = 0
t_sim = [0, 200]
t_eval = np.arange(0, 201, 1)

compartments = OrderedDict([['S', S0], ['I', I0], ['R', R0]])

# note: copy of above but above will be updated with current values so keep this for re-running/reference
initial_conditions = OrderedDict([['S', S0], 
                                  ['I', I0], 
                                  ['R', R0]])

parameters = OrderedDict([['beta', 0.3], 
                          ['gamma', 0.1]])

#stoichiometry_wide = [['S',['J_inf',-1]],
#                      ['I',['J_inf',1],['J_rec',-1]],
#                      ['R',['J_rec',1]]]

#stoichiometry = ode_lib.stoichiometry_wide_to_tall(stoichiometry_wide)

# now using reaction scheme form
# either
#reaction_system = ['J_inf: S -> I',
#                   'J_rec: I -> R'
# or
reaction_system = ['J_inf: S + I -> 2I',
                   'J_rec: I -> R']

# maybe need to think a bit better about passing by copy or ref but oh well
stoichiometry = OrderedDict()
stoichiometry = ode_lib.parse_reaction_system(reaction_system=reaction_system,d=stoichiometry)

# will give expected form:
# stoichiometry = OrderedDict([[('S', 'J_inf'), -1],
#                              [('I', 'J_inf'), 1],
#                              [('I', 'J_rec'), -1],
#                              [('R', 'J_rec'), 1]])

# save figure
dot = ode_lib.stoichiometry_to_dot(stoichiometry,dot_name='output/SIR_example_usage',save=True)

fluxes = OrderedDict([['J_inf', lambda c, p: p['beta'] * c['S'] * c['I']],
                      ['J_rec', lambda c, p: p['gamma'] * c['I']]])

problem_data = {'compartments':compartments,
                'parameters':parameters,
                'fluxes':fluxes,
                'stoichiometry':stoichiometry,
                'initial_conditions':initial_conditions}


S,I,R = ode_lib.simulate_and_eval(problem_data,t_sim,t_eval)

# total infections
plt.figure()
plt.plot(R*N)
plt.show()

# total infections logged
plt.figure()
plt.plot(R*N)
plt.yscale('log')
plt.show()

# empirical Reff estimates
plt.figure()
plt.plot(1+(1/parameters['gamma'])*np.gradient(np.log(np.diff(R))))
plt.show()

plt.figure()
plt.plot(1+(1/parameters['gamma'])*np.gradient(np.log(I)))
plt.show()

# all compartments
plt.figure()
plt.plot(S*N, label='S')
plt.plot(I*N, label='I')
plt.plot(R*N, label='R')
plt.legend()
plt.show()
