#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tellurium as te
import simplesbml
import time

'''
Useful tellurium functions:
        te.getODEsFromModel(te_model)
        te.getEigenvalues(te_model)


        Basis = te.nullspace(A, atol=1e-13, rtol=0)
        rk = te.rank(A, atol=1e-13, rtol=0)
'''


def MLE_tellurium(Lambda,
                  te_model,
                  tfinal,
                  n_timestep,
                  u,
                  scaling_c=None,
                  err_tol=None,
                  return_runtime=False,
                  check_convergence=False,
                  maxNumOfIntegrations=5,
                  check_compatibility_class=False,
                  eigenvalueAnalysis=True,
                  plot_traj=False,
                  verbose=False):
    '''
    Computes MLE of a log-linear model M_{A} given:
        - matrix Lambda (size m * s) defining te_model
        - vector of counts u (np.array, size (m,))
        - scaling vector scaling_c (np.array, size (m,))
    ! Currently no convergence criteria implemented.

    OUTPUT:  MLE (np.array,  size (m,))

    Parameters
    ----------
    te_model : tellurium model
        Contains all species, reactions, etc. Used as part of tellurium package.
    tfinal : scalar
        final time of integration
    n_timestep : integer
        Determines meshsize: dt = tfinal/n_timesteps.
    u : np.array, size (m,)
        vector of counts. Default is None.
        Uses normalized u as IC in te_model
    scaling_c (optional) : np.array, size (m,)
        Scaling vector. The default is None, translated as (1,...,1).
    err_tol (optional) : TYPE, optional
        Could be implemented to ensure |x(tfinal) - x(tfinal-1)| < err_tol
        ! Not yet implemented !
    plot_traj (optional) : Boolean
        If True, plots the trajectories resulting from odeint
        The default is False.
    return_runtime (optional) : Boolean
        If True, returns time it took to integrate via tic-toc.
        If False, returns an empty array []
        The default is False.
    check_convergence (optional) : Boolean
        If True, checks cnvg by checking DB flux < err_tol.
        If False, returns an empty array []
        The default is False.
    maxNumOfIntegrations (optional): integer
        # times of repeated integrations even if hasn't converged.
        If check_convergence = True, maximum
    check_compatibility_class (optional) : Boolean
        If True, checks if u and x(tfinal) are in same stoichiometric compatibility class.
        If False, returns an empty array []
        The default is False.
    eigenvalueAnalysis (optional) : Boolean
        If True, ...............................
        If False, returns an empty array []
        The default is False.

    Returns
    -------
    MLE : np.array, size (m,)
        Final state from initial state u_normalized.
        MLE if all entries positive.
    runtime (if return_runtime=True) : scalar
        Time it took odeint to integrate; uses time.time()

    '''

    #variables
    dim_m = Lambda.shape[0]
    #reversible pairs
    dim_s = Lambda.shape[1]

    if u is not None:
        u = np.array(u)
        u = (u.flatten()).reshape(-1)

        if len(u) != dim_m:
            print('\n Error: mismatch dims of Lambda and u')
        else:
            u = u / u.sum()
            te_model = new_IC(te_model, u)
    # else if no u given: # use existing IC in te_model

    if scaling_c is not None:
        scaling_c = np.array(scaling_c)
        scaling_c = (scaling_c.flatten()).reshape(-1)
        if len(scaling_c) != dim_m:
            print('\n Error: mismatch dims of Lambda and scaling_c ')
    else:
        scaling_c = np.ones(dim_m)

    if err_tol is None:
        err_tol = 1e-4

    tic = time.time();
    results = te_model.simulate(0, tfinal, n_timestep)
    toc = time.time();
    runtime = toc - tic;

    if check_convergence:
        cnvg_flag, DB_residual_n = check_detailed_balancing(
            Lambda,
            results[n_timestep - 1, 1:-1],
            scaling_c=scaling_c,
            err_tol=err_tol)
        reintegrate_counter = 1
        # integrates again up to 4 more times
        while DB_residual_n > err_tol and reintegrate_counter < maxNumOfIntegrations:
                         tic = time.time();
            results = te_model.simulate(0, tfinal, n_timestep)
                         toc = time.time();
                         runtime += toc - tic;
 
            cnvg_flag, DB_residual_n = check_detailed_balancing(
                Lambda,
                results[n_timestep - 1, 1:-1],
                scaling_c=scaling_c,
                err_tol=err_tol)
            reintegrate_counter += 1
        if verbose:
            print(
                f'\n Checking convergence: integrated {reintegrate_counter} # of times.'
            )
    else:
        cnvg_flag = []
        DB_residual_n = []
    MLE = results[n_timestep - 1, 1:-1]

    if check_compatibility_class:
        same_cmptb_class_flag = check_compatibility_class_te_rank(
            Lambda, MLE, u, verbose=verbose)
    else:
        same_cmptb_class_flag = []

    if eigenvalueAnalysis:
        eig = te_model.getFullEigenValues()
        sorted_ind = np.argsort(np.real(eig))
        eig = eig[sorted_ind]
    else:
        eig = []

    if plot_traj == True:
        tfinalminusone = np.where(results[:, 0] < tfinal - 1)[0]
        tfinalminusone = max(tfinalminusone)
        diff_norm = np.linalg.norm(MLE - results[tfinalminusone - 1,
                                                 1:dim_m + 1]) / dim_m

        plt.plot(results[:, 0], results[:, 1:dim_m + 1])
        plt.title('|x(T)-x(T-1)|/m = %.2e' % diff_norm)
        plt.show()

    return MLE, runtime, cnvg_flag, DB_residual_n, same_cmptb_class_flag, eig


def write_tellurium_model_via_SBML(Lambda,
                                   scaling_c=None,
                                   verbose=False,
                                   return_runtime=False):
    '''
    Outputs a tellurium model with reversible MAS with stoichiometric matrix Lambda:
        gamma_neg[:,rr] -- scaling_c ** gamma_pos[:,rr] --> gamma_pos[:,rr]
        gamma_pos[:,rr] -- scaling_c ** gamma_neg[:,rr] --> gamma_neg[:,rr]

    Parameters
    ----------
    Lambda : np.array, size m * s.
        Stoichiometric matrix; each column = reaction vector of FORWARD reaction 
    scaling_c (optional) : np.array, size (m,)
        defines rate constants. If None, scaling_c = (1,...,1).
    verbose (optional) : Boolean
        The default is False.
        If true, print human readable string of tellurium model
    return_runtime (optional) : Boolean
        If True, returns time it took to load model via tic-toc.
        If False, no returns.
        The default is False.

    Returns
    -------
    te_model : tellurium model 
        To be used in tellurium's integrator
    '''

    dim_m = Lambda.shape[0]
    dim_s = Lambda.shape[1]

    Lambda_positive = np.maximum(Lambda, 0)
    Lambda_negative = np.maximum(-Lambda, 0)

    if scaling_c is not None:
        scaling_c = np.array(scaling_c)
        scaling_c = (scaling_c.flatten()).reshape(-1)
        if len(scaling_c) != dim_m:
            print('\n Error: mismatch dims of Lambda and scaling_c ')
    else:
        scaling_c = np.ones(dim_m)
    rate_cnsts_fwd, rate_cnsts_bck = define_rate_constants(Lambda,
                                                           scaling_c=scaling_c)

    SimSBML = simplesbml.SbmlModel()

    # extra species (for zero complex)
    # adding species with concentrations 0
    for ss in range(dim_m): 
        SimSBML.addSpecies('x' + str(ss), 0.0)
    SimSBML.addSpecies('ext_S', 0.0)
    

    for rr in range(dim_s):  ### adding each reversible reactions
        ## SimSBML.addReaction(Reactant_list, Product_list, Rate_law, local_params=params_dict, rxn_id=rxn_id_str)
        reactants_ind = np.where(Lambda_negative[:, rr] > 0)[0]
        products_ind = np.where(Lambda_positive[:, rr] > 0)[0]
        reactants_stoich = Lambda_negative[:, rr]
        products_stoich = Lambda_positive[:, rr]

        reactants_list = []
        products_list = []
        rate_law_fwd_expr = 'f'
        rate_law_bck_expr = 'b'

        if len(reactants_ind) == 0:
            reactants_list.append('ext_S')
        else:
            for ss in reactants_ind:
                rate_law_fwd_expr += '*'
                if reactants_stoich[ss] == 1:
                    reactants_list.append('x' + str(ss))
                    rate_law_fwd_expr += 'x' + str(ss)
                else:  ## need to add stoichiometric coefficient (>1), and powers to rate law
                    reactants_list.append(
                        str(reactants_stoich[ss]) + ' x' + str(ss))
                    rate_law_fwd_expr += 'x' + str(ss) + '^' + str(
                        reactants_stoich[ss])

        if len(products_ind) == 0:
            products_list.append('ext_S')
        else:
            for ss in products_ind:
                rate_law_bck_expr += '*'
                if products_stoich[ss] == 1:
                    products_list.append('x' + str(ss))
                    rate_law_bck_expr += 'x' + str(ss)
                else:
                    products_list.append(
                        str(products_stoich[ss]) + ' x' + str(ss))
                    rate_law_bck_expr += 'x' + str(ss) + '^' + str(
                        products_stoich[ss])

        rate_law_expr = rate_law_fwd_expr + ' - ' + rate_law_bck_expr
        rxn_id_str = 'J' + str(rr)

        param_dict = {}
        param_dict['f'] = rate_cnsts_fwd[rr]
        param_dict['b'] = rate_cnsts_bck[rr]

        SimSBML.addReaction(reactants_list,
                            products_list,
                            rate_law_expr,
                            local_params=param_dict,
                            rxn_id=rxn_id_str)


     tic = time.time();
    te_model = te.loads(SimSBML.toSBML())
         toc = time.time();
         runtime = toc - tic;


    if verbose:
        print(te_model.getCurrentAntimony())
        ## produces readable strign with models + parameters

    if return_runtime:
        return te_model, runtime
    else:
        return te_model


def define_rate_constants_all(Lambda, scaling_c=None):
    ''' 
    Return rate constants of both forward + backward reactions given 
        stoichiometric matrix Lambda, and scaling_c vector c:
        Forward rate constant = scaling_c ** (gamma_pos[r])
        Backward rate constant = scaling_c ** (gamma_neg[r])

    Parameters
    ----------
    Lambda : np.array, size m * s.
        Stoichiometric matrix; each column = reaction vector of FORWARD reaction 
    scaling_c (optional): np.array, size (m,).
        Defines rate constants of reversible reactions:  fwd = scaling_c ** gamma_pos[r],  bck = scaling_c ** gamma_neg[r]
        The default is None -- will be set as (1,...,1).

    Returns
    -------
    rate_cnsts :  np.array, size (2s,)
    '''

    dim_m = Lambda.shape[0]
    dim_s = Lambda.shape[1]

    if scaling_c is not None:
        scaling_c = np.array(scaling_c)
        scaling_c = (scaling_c.flatten()).reshape(-1)
        if len(scaling_c) != dim_m:
            print('\n Error: mismatch dims of Lambda and scaling_c ')
    else:
        scaling_c = np.ones(dim_m)

    Lambda_positive = np.maximum(Lambda, 0)
    Lambda_negative = np.maximum(-Lambda, 0)

    rate_cnsts_fwd = np.array([
        np.prod(np.power(scaling_c, Lambda_positive[:, rr]))
        for rr in range(dim_s)
    ])
    rate_cnsts_bck = np.array([
        np.prod(np.power(scaling_c, Lambda_negative[:, rr]))
        for rr in range(dim_s)
    ])
    rate_cnsts = np.hstack((rate_cnsts_fwd, rate_cnsts_bck))

    return rate_cnsts


def define_rate_constants(Lambda, scaling_c=None):
    ''' 
    Return rate constants of both forward + backward reactions (separately) given 
        stoichiometric matrix Lambda, and scaling_c vector c:
        Forward rate constant = scaling_c ** (gamma_pos[r])
        Backward rate constant = scaling_c ** (gamma_neg[r])

    Parameters
    ----------
    Lambda : np.array, size m * s.
        Stoichiometric matrix; each column = reaction vector of FORWARD reaction 
    scaling_c (optional): np.array, size (m,).
        Defines rate constants of reversible reactions:  fwd = scaling_c ** gamma_pos[r],  bck = scaling_c ** gamma_neg[r]
        The default is None -- will be set as (1,...,1).

    Returns
    -------
    rate_cnsts_fwd :  np.array, size (s,)
    rate_cnsts_bck :  np.array, size (s,)
    '''

    dim_m = Lambda.shape[0]
    dim_s = Lambda.shape[1]

    if scaling_c is not None:
        scaling_c = np.array(scaling_c)
        scaling_c = (scaling_c.flatten()).reshape(-1)
        if len(scaling_c) != dim_m:
            print('\n Error: mismatch dims of Lambda and scaling_c ')
    else:
        scaling_c = np.ones(dim_m)

    Lambda_positive = np.maximum(Lambda, 0)
    Lambda_negative = np.maximum(-Lambda, 0)

    rate_cnsts_fwd = np.array([
        np.prod(np.power(scaling_c, Lambda_positive[:, rr]))
        for rr in range(dim_s)
    ])
    rate_cnsts_bck = np.array([
        np.prod(np.power(scaling_c, Lambda_negative[:, rr]))
        for rr in range(dim_s)
    ])

    return rate_cnsts_fwd, rate_cnsts_bck


def new_IC(te_model, u):
    '''
    Update Tellurium model te_model with new initial conditions set by vector u

    Parameters
    ----------
    te_model : tellurium model (with rxns, rate cnsts, current states)
        
    u : np.array, size (m,)
        new initial condition to be inputted into the system 

    Returns
    -------
    te_model : tellurium model with updated IC 

    '''

    u = np.array(u)
    u = (u.flatten()).reshape(-1)
    dim_m = u.shape[0]

    list_of_species = [f"init(x{i})" for i in range(dim_m)]
    te_model.setValues(list_of_species, u)

    return te_model


def new_rate_cnsts(te_model, rate_cnsts):
    '''### TO BE IMPLEMENTED LATER ### '''
    return te_model


def check_detailed_balancing(Lambda,
                             ss_val,
                             scaling_c=None,
                             err_tol=None,
                             verbose=False):
    '''
    Check if ss_val is in detailed-balanced for the system, by computing 
        flux_fwd = (scaling_c ** Lambda_positive[rr]) * (ss_val ** Lambda_negative[rr])
        flux_bck = (scaling_c ** Lambda_negative[rr]) * (ss_val ** Lambda_positive[rr])
    If any entry in abs(flux_fwd - flux_bck) > err_tol, NOT DB.
    If all entry in abs(flux_fwd - flux_bck) < err_tol, DB. 

    Parameters
    ----------
    Lambda : np.array, size m * s.
        Stoichiometric matrix; each column = reaction vector of FORWARD reaction 
    ss_val : np.array, size (m,)
        steady state value to be checked
    scaling_c (optional) : np.array, size (m,)
        defines rate constants. If None, scaling_c = (1,...,1).
    err_tol (optional) :  
        If None, default is 1e-4.
    verbose (optional) : Boolean 
        Default = False.
        If true, prints flux_fwd - flux_bck as a vector.

    Returns
    -------
    is_detailed_balancing_flag : boolean 
        TRUE if abs(flux_fwd - flux_bcK) < err_tol  for ALL reactions 
        FALSE if any flux > err_tol.
        
    residual_n : scalar 
        || flux_fwd - flux_bck ||_2 / dim_s 
    '''

    dim_m = Lambda.shape[0]
    dim_s = Lambda.shape[1]

    ss_val = np.array(ss_val)
    ss_val = (ss_val.flatten()).reshape(-1)
    if len(ss_val) != dim_m:
        print('\n Error: mismatch dims of Lambda and ss_val')

    if scaling_c is not None:
        scaling_c = np.array(scaling_c)
        scaling_c = (scaling_c.flatten()).reshape(-1)
        if len(scaling_c) != dim_m:
            print('\n Error: mismatch dims of Lambda and scaling_c ')
        else:
            rate_cnsts_fwd, rate_cnsts_bck = define_rate_constants(
                Lambda, scaling_c=scaling_c)
    else:
        rate_cnsts_fwd = np.ones(dim_s)
        rate_cnsts_bck = np.ones(dim_s)

    if err_tol is None:
        err_tol = 1e-4

    Lambda_positive = np.maximum(Lambda, 0)
    Lambda_negative = np.maximum(-Lambda, 0)

    flux_fwd = rate_cnsts_fwd * np.array([
        np.prod(np.power(ss_val, Lambda_negative[:, rr])) for rr in range(dim_s)
    ])
    flux_bck = rate_cnsts_bck * np.array([
        np.prod(np.power(ss_val, Lambda_positive[:, rr])) for rr in range(dim_s)
    ])
    flux_diff = flux_fwd - flux_bck

    if np.any(abs(flux_diff) > err_tol):
        is_detailed_balancing_flag = False
    else:
        is_detailed_balancing_flag = True

    residual_n = np.linalg.norm(
        flux_diff) / dim_s  # normalized residual norm(flux_diff)/dim_s

    if verbose:
        print(f'\n flux across each pair = {flux_diff}.')
        print(f'\n residual_n = |flux_diff|/# of reactions = {residual_n}.')

    return is_detailed_balancing_flag, residual_n


def check_compatibility_class_te_rank(Lambda, ss_val, u, verbose=False):
    ''''  
    Check if ss_val is in same compatibility class as u, by checking if 
    (u-ss_val) is in the colspace(Lambda), i.e., Lambda @ x = u-ss_val for some x.
    This is done using Tellurium's rank approximation method (based on SVD).
    If rank(Lambda) = rank([Lambda, ss_val-u]), then returns in_same_compatibility_class_flag = TRUE.
    # Else, return in_same_compatibility_class_flag = FALSE. 
    
    By default, absolute tolerance for 0 SV is 1e-13.

    Parameters
    ----------
    Lambda : np.array, size m * s.
        Stoichiometric matrix; each column = reaction vector of FORWARD reaction 
    ss_val : np.array, size (m,)
        steady state value to be checked
    u : np.array, size (m,)
        initial condition defining compatibility class.
    verbose (optional) : Boolean 
        Default = False.
        If true, prints more message on what went wrong.

    Returns
    -------
    in_same_compatibility_class_flag : boolean 
        
    '''

    dim_m = Lambda.shape[0]
    dim_s = Lambda.shape[1]

    ss_val = np.array(ss_val)
    ss_val = (ss_val.flatten()).reshape(-1)
    if len(ss_val) != dim_m:
        print('\n Error: mismatch dims of Lambda and ss_val')
    u = np.array(u)
    u = (u.flatten()).reshape(-1)
    if len(u) != dim_m:
        print('\n Error: mismatch dims of Lambda and u')

    Lambda_rk = te.rank(Lambda)
    Lambda_aug_rk = te.rank(
        np.hstack((Lambda, np.reshape(u - ss_val, (u.shape[0], 1)))))

    if Lambda_rk == Lambda_aug_rk:
        in_same_compatibility_class_flag = True
        if verbose:
            print('\n ss_val and u are in same compatibility class :)')
    else:
        in_same_compatibility_class_flag = False
        if verbose:
            print('\n ss_val and u NOT in same compatibility class.')

    return in_same_compatibility_class_flag


def timescales(Lambda,
               u=None,
               tfinal=1e3,
               n_timestep=750,
               u_rand_min=10,
               u_rand_max=50):
    
    '''
    Estimate the timescales of a system based on the eigenvalues of the Jacobian,
    using the function `MLE_tellurium` to trace the trajectories of the system.
    
    If no initial condition u is provided, a random vector u is generated within 
    the range [u_rand_min, u_rand_max].
    
    Parameters
    ----------
    Lambda : np.array, size m * m.
        A square matrix that defines the system's dynamics.
    u : np.array, size (m,), optional.
        Initial condition for the system. If None, a random vector will be generated.
    tfinal : float, optional.
        Final time for the simulation. Default is 1e3.
    n_timestep : int, optional.
        Number of time steps for the simulation. Default is 750.
    u_rand_min : int, optional.
        Minimum value for generating random initial conditions if u is not provided. Default is 10.
    u_rand_max : int, optional.
        Maximum value for generating random initial conditions if u is not provided. Default is 50.

    Returns
    -------
    timescales : np.array
        Array of timescales (inverse of the real parts of nonzero eigenvalues).
    '''

    Lambda_rk = te.rank(Lambda)

    if u is None:
        u = np.random.randint(u_rand_min,
                              u_rand_max,
                              size=(1, Lambda.shape[0]))

    te_model = write_tellurium_model_via_SBML(Lambda, return_runtime=False)

    _, _, _, _, _, eig = MLE_tellurium(Lambda,
                                       te_model,
                                       tfinal,
                                       n_timestep,
                                       u,
                                       eigenvalueAnalysis=True)

    eig_real = eig.real
    eig_imag = eig.imag
    eig_real_nonzero = eig_real[0:Lambda_rk]

    return 1 / abs(eig_real_nonzero)


#     '''
#     Example usage with independence model:
#     Requires from independence_models import *
# k = 4
# d = 6
# Lambda = independence_model_kernel(k,d)
# u = np.random.randint(1,15, size=(Lambda.shape[0]));
# # u = np.array([[12,  8, 11,  9,  1, 14,  2,  2,  7,  5,  8,  1,  8,  6,  6, 10,
# #          3, 12,  4,  1, 14,  5,  6, 10]])
# MLE_theory = independence_model_birch_point(k,d, u);

# tfinal = 250
# n_timestep = 100

# te_model, loadtime = write_tellurium_model_via_SBML(Lambda, scaling_c=None, verbose=False, return_runtime=True)
# MLE_te, runtime, cnvg_flag, DB_residual_n, same_cmptb_class_flag, eig = MLE_tellurium(Lambda, te_model, tfinal, n_timestep, u, scaling_c=None, err_tol=None,
#                   return_runtime=True, check_convergence=True, check_compatibility_class=True,
#                   eigenvalueAnalysis=True,
#                   plot_traj=False, verbose=False)
# print(np.linalg.norm(MLE_te - MLE_theory)/Lambda.shape[0])
# print(f'te runtime = {runtime} s')

#     '''

# k = 10
# d = 15
# Lambda = independence_model_kernel(k,d)
# u = np.random.randint(1,15, size=(Lambda.shape[0]));
# # u = np.array([[12,  8, 11,  9,  1, 14,  2,  2,  7,  5,  8,  1,  8,  6,  6, 10,
# #          3, 12,  4,  1, 14,  5,  6, 10]])
# MLE_theory = independence_model_birch_point(k,d, u);

# tfinal = 300
# n_timestep = 50

# te_model, loadtime = write_tellurium_model_via_SBML(Lambda, scaling_c=None, verbose=False, return_runtime=True)
# MLE_te, runtime, cnvg_flag, DB_residual_n, same_cmptb_class_flag, eig = MLE_tellurium(Lambda, te_model, tfinal, n_timestep, u, scaling_c=None, err_tol=None,
#                   return_runtime=True, check_convergence=True, check_compatibility_class=True,
#                   eigenvalueAnalysis=False,
#                   plot_traj=True, verbose=False)
# print(np.linalg.norm(MLE_te - MLE_theory)/Lambda.shape[0])
# print(f'te runtime = {runtime} s')
