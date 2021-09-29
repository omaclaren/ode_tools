from scipy.integrate import solve_ivp
import numpy as np
import graphviz 
from collections import OrderedDict
import re

def parse_reaction(reaction_process,d={}):
    '''
    Works with unordered dict or ordred dict coz not crucial to be ordered...
    Also, updates an existing dictionary or returns a newly created dictionary.
    Form of a reaction process is 'J: A+2B -> C' etc. Can include <-> reactions, treats as separate.
    '''
    r_nospace = reaction_process.replace(' ','')
    flux, reaction = r_nospace.split(':')
    reactions = [reaction.replace('<->','->')]
    if re.search('<->',reaction):
        reverse_products = reaction.split('<->')[0]
        reverse_reactants = reaction.split('<->')[1]
        reverse = reverse_reactants + '->' + reverse_products
        reactions.append(reverse)
   
    if len(reactions) > 1:
        flux_name = flux + '_f'
    else:
        flux_name = flux
    for i, reaction_i in enumerate(reactions):
        if i == 1:
            flux_name = flux + '_r'
        reactants, products = reaction_i.split('->')
        react_parts = reactants.split('+')
        prod_parts = products.split('+')
        print(reaction_i)
        for react_part in react_parts:
            coeff = re.match(r'\d{0,}',react_part)[0]
            species = react_part.replace(coeff,'')
            if coeff == '':
                coeff = 1
            if (species, flux_name) in d.keys():
                d[(species,flux_name)] += -int(coeff)
            else:
                d[(species, flux_name)] = -int(coeff)
        for prod_part in prod_parts:
            coeff = re.match(r'\d{0,}',prod_part)[0]
            species = prod_part.replace(coeff,'')
            if coeff == '':
                coeff = 1
            if (species, flux_name) in d.keys():
                d[(species,flux_name)] += int(coeff)
            else:
                d[(species, flux_name)] = int(coeff)
    return d

def parse_reaction_system(reaction_system, d={}):
    '''
    Parses a whole system of reactions
    '''
    for ri in reaction_system:
        d = parse_reaction(reaction_process=ri,d=d)
    
    return d

def stoichiometry_wide_to_tall(wide):
    '''
    Note: takes in list (?), outputs OrderedDict. Probably not necessary for this one to be ordered.

    Takes stoichiometry in form e.g.
    
    wide = [['SU',['J_1',1],['J_2',-1]],
            ['IU',['J_1',-1]]]

    and returns tall in form 

    tall = OrderedDict([(('SU', 'J_1'), 1), 
                        (('SU', 'J_2'), -1), 
                        (('IU', 'J_1'), -1)])
    '''
    tall = OrderedDict()
    for wide_i in wide:
        for flux_i in wide_i[1:]:
            tall[((wide_i[0], flux_i[0]))] = flux_i[1]
    return tall

def stoichiometry_to_dot(stoichiometry,dot_name='Reaction Scheme',save=False):
    '''
    Assumes in tall form, as for ODE.
    '''
    dot = graphviz.Digraph(dot_name)
    for cr,s in stoichiometry.items():
        dot.node(cr[0],shape='circle')
        dot.node(cr[1],shape='box')
        if s > 0:
            dot.edge(cr[1],cr[0])
        else:
            dot.edge(cr[0],cr[1])
    if save==True:
        dot.render()
    return dot

def ode_template(t, y, compartments, parameters, fluxes, stoichiometry):
    ''' 
    Note: generally all ordered dicts. 
    Key one to have ordered is compartments (ordering needed for simulation).

    - compartments: species, amount/conc
    - parameters: parm, value
    - fluxes: process, constit. expression
    - stoichiometry: (species, flux), integer
    
    '''

    n_state = len(compartments.keys())
    ode_rhs = np.zeros((n_state,))

    # update compartment values with y values
    for i, c in enumerate(compartments.keys()):
        compartments[c] = y[i]

    # update RHS
    for i, c in enumerate(compartments.keys()):
        ode_rhs[i] = np.sum([fluxes[cj[1]](compartments,parameters) * sto for cj,sto in stoichiometry.items() if cj[0] == c])
    
    #print(ode_rhs)
    return ode_rhs


def simulate_and_eval(problem_data, t_sim, t_eval, method='RK45', return_sim=False):
    '''
    OrderedDicts: in particular compartments and initial conditions. Others less important to be ordered.
    '''
    compartments = problem_data['compartments']
    parameters = problem_data['parameters'] 
    fluxes = problem_data['fluxes']
    stoichiometry = problem_data['stoichiometry']
    initial_conditions = problem_data['initial_conditions']

    full_sol = solve_ivp(lambda t, y: ode_template(t, y, 
                         compartments = compartments,
                         parameters = parameters,
                         fluxes = fluxes,
                         stoichiometry = stoichiometry),
                         t_span=t_sim, y0=list(initial_conditions.values()), 
                         method=method,
                         dense_output=True)
    sol = full_sol.sol
    if return_sim:
        return sol(t_eval), sol
    return sol(t_eval)
    

