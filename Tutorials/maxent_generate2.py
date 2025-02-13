#!/usr/bin/env python
# coding: utf-8
r"""
# Actualization Criteria and Self-Consistent Mean-Field Projections

In this tutorial, we will build upon the concepts introduced in the previous
tutorial, where we explored different actualization criteria for Adaptive
Max-Ent. Notably, we discovered that the **Partial Sum criterion** provides
the tightest control over errors, in contrast to the
**Lieb-Robinson criterion**, which is lestightnt.

However, as the complexity of the bases involved in the actualization g
linearly with the number of actualizations and with the depth of the basis,
$\ell$rows, managing this complexity becomes critical. To address this
challenge, a Self-Consistent **Mean-Field (MF) projection** is applied,
to reduce and organize the basis while retaining its essential features.

In this tutorial, we will continue the discussion while maintaining the
same initial conditions for the system.

### Objectives:

We will introduce two examples of **MF-aided Adaptive Max-Ent simulations**:
1. The basis of observables is constructed as a **$m_0$-body projected
   Hierarchical Basis**, where the projection ensures that all elements of
   the basis remain manageable.
2. We will demonstrate how Mean-Field projections enable efficient
   simulations while preserving the accuracy of the Adaptive Max-Ent scheme.

"""
# In[1]:

from typing import Callable, Optional
import pickle
from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import fsolve
import qutip


from alpsqutip import (
    restricted_maxent_toolkit as me,
)  # # custom library including basic linear algebra functions
from alpsqutip.alpsmodels import model_from_alps_xml
from alpsqutip.geometry import graph_from_alps_xml
from alpsqutip.model import SystemDescriptor
from alpsqutip.proj_evol import (
    safe_exp_and_normalize,
)  # # function used to safely and robustly map K-states to states


MODELS_LIB_FILE = "../alpsqutip/lib/models.xml"
LATTICE_LIB_FILE = "../alpsqutip/lib/lattices.xml"
SIMULATIONS_FILE_PREFIX = "simulations"


OBSERVABLES = {}
PARAMS = {}

PARAMS["size"] = 7
PARAMS["Jx"] = 1.0
PARAMS["Jy"] = 0.75 * PARAMS["Jx"]
PARAMS["Jz"] = 1.05 * PARAMS["Jx"]
PARAMS["phi0"] = np.array(
    [0.0, 0.25, 0.25, -10.0]
)  # No podemos ir más allá de |phi|< 10,
# ya que la matriz de Gram se hace singular.


# long term ev

# Configuration du style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 12,  # Taille de police
        "axes.labelsize": 14,  # Taille des labels des axes
        "axes.titlesize": 16,  # Taille des titres
        "legend.fontsize": 12,  # Taille des légendes
        "xtick.labelsize": 12,  # Taille des labels des ticks sur l'axe X
        "ytick.labelsize": 12,  # Taille des labels des ticks sur l'axe Y
        "font.family": "serif",  # Police de type "serif" pour
        # un rendu professionnel
        "axes.linewidth": 1.5,  # Largeur des bordures des axes
        "grid.alpha": 0.5,  # Transparence des grilles
    }
)


# functions used for Mean-Field projections


def lieb_robinson_speed(parameters):
    """Compute the Lieb Robinson speed from the parameters"""

    f_factor = np.real(
        max(
            np.roots(
                np.poly1d(
                    [
                        1,
                        0,
                        -(
                            parameters["Jx"] * parameters["Jy"]
                            + parameters["Jx"] * parameters["Jy"]
                            + parameters["Jy"] * parameters["Jz"]
                        ),
                        -2 * (
                            parameters["Jx"] *
                            parameters["Jy"] *
                            parameters["Jz"]
                        ),
                    ]
                )
            )
        )
    )
    chi_y = fsolve(
        lambda x, y: x * np.arcsinh(x) - np.sqrt(x**2 + 1) - y, 1e-1, args=(0)
    )[0]
    return 4 * f_factor * chi_y


# # Load the set of previous simulations

# In[2]:

def load_simulations(filename_prefix=None) -> dict:
    """Load a set of simulations"""
    # Here we store the results of the simulations
    if filename_prefix is None:
        filename_prefix = SIMULATIONS_FILE_PREFIX
    try:
        with open(f"{filename_prefix}.pkl", "br") as in_file:
            simulations = pickle.load(in_file)

        with open(f"{filename_prefix}_{str(datetime.now())}.bkp",
                  "bw") as out_file:
            pickle.dump(simulations, out_file)
    except Exception:
        simulations = {}

    return initialize_simulations_dict(simulations)


# # Define the system and objects required for the simulation
# In[3]%:

def setup_simulation_context() -> dict:
    """
    Setup the system using the PARAMS.
    """
    v_lr = lieb_robinson_speed(PARAMS)
    system = SystemDescriptor(
        model=model_from_alps_xml(MODELS_LIB_FILE, "spin"),
        graph=graph_from_alps_xml(
            LATTICE_LIB_FILE, "open chain lattice",
            parms={"L": PARAMS["size"], "a": 1}
        ),
        parms={"h": 0, "J": PARAMS["Jx"]},
    )

    sites = list(system.sites)
    sx_ops = [
        system.site_operator("Sx", "1[" + str(a) + "]")
        for a in range(len(system.sites))
    ]
    sy_ops = [
        system.site_operator("Sy", "1[" + str(a) + "]")
        for a in range(len(system.sites))
    ]
    sz_ops = [
        system.site_operator("Sz", "1[" + str(a) + "]")
        for a in range(len(system.sites))
    ]

    hamiltoniano = (
        PARAMS["Jx"] * sum(sx_ops[i] * sx_ops[i + 1]
                           for i in range(PARAMS["size"] - 1))
        + PARAMS["Jy"] * sum(sy_ops[i] * sy_ops[i + 1]
                             for i in range(PARAMS["size"] - 1))
        + PARAMS["Jz"] * sum(sz_ops[i] * sz_ops[i + 1]
                             for i in range(PARAMS["size"] - 1))
    )
    idop = system.site_operator("identity", sites[0])
    hbb0 = [
        idop,
        system.site_operator("Sx", "1[0]"),
        system.site_operator("Sy", "1[0]"),
        system.site_operator("Sz", "1[0]"),
    ]

    phi0 = PARAMS["phi0"]
    k0 = me.k_state_from_phi_basis(phi0, hbb0)
    sigma0 = safe_exp_and_normalize(k0)[0]
    phi0[0] = np.log(sigma0.tr())
    k0 = me.k_state_from_phi_basis(phi0, hbb0)
    sigma0, _ = safe_exp_and_normalize(k0)

    OBSERVABLES.update({
        "obs_SxA": sum(sx_ops[i] for i in range(3)),
        "obs_SzA": sum(sz_ops[i] for i in range(3)),
        "obs_Sx3Sx4": (sx_ops[2] * sy_ops[3]),
    })

    timespan = np.linspace(0.0, 650.1 / max(10, v_lr), 650)
    return {
        "OBSERVABLES": OBSERVABLES,
        "system": system,
        "H": hamiltoniano,
        "timespan": timespan,
        "phi0": phi0,
        "k_0": k0,
        "sigma0": sigma0,
    }


# ## Define the basis and the initial state

# In[4]:


# ## Define the observables we are going to track

# In[5]:


def estimate_error_by_partial_sum(phi_local, ws):
    """Compute the error estimation based on partial sums"""
    return me.m_th_partial_sum(phi=phi_local, m=2) / me.m_th_partial_sum(
        phi=phi_local, m=0
    )


def estimate_error_by_weights(phi, ws):
    """Compute the error estimation based on loss weights"""
    return sum(abs(phi_a * w_a) for phi_a, w_a in zip(phi, ws))


# ## Check consistency and get the exact evolution
# * Verify that the parameters coincides with the parameters
#   stored in the previous run.
# * Run the exact simulation if needed.

# In[6]:
# exact dynamics
# Uncomment the next line to start a new set of simulations
# simulations = {}

def initialize_simulations_dict(simulations: Optional[dict] = None) -> dict:
    """
    Parameters
    ----------
    simulations : Optional[dict], optional
        DESCRIPTION. The default is {}.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    if simulations is None:
        simulations = {}

    if "_params" in simulations:
        stored_parms = simulations["_params"]
        for key in stored_parms:
            if key == "phi0":
                assert list(stored_parms[key]) == list(PARAMS[key])
            else:
                assert stored_parms[key] == PARAMS[key]
        assert all(key in stored_parms for key in PARAMS), "keys missing"
        # simulations["parms"] = PARAMS
    else:
        simulations["_params"] = PARAMS
        simulations["_observables"] = OBSERVABLES

    return simulations


def run_exact_simulation(simulation: dict, simulation_context: dict):
    """
    Solve the exact Schödinger's dynamic for the parameters of the
    simulation.
    """
    system = simulation_context["system"]
    timespan = simulation_context["timespan"]
    hamiltonian = simulation_context["H"]
    k_0 = simulation_context["k_0"]

    expect_values = {key: [] for key in OBSERVABLES}
    expect_values["time"] = []
    names_to_pos = {name: i for i, name in enumerate(system.sites)}

    def callback_compute_obs(t, k):
        rho, _ = safe_exp_and_normalize(k)
        expect_values["time"].append(t)
        for key, obs_op_ss in OBSERVABLES.items():
            ss = tuple(sorted((names_to_pos[site]
                       for site in obs_op_ss.acts_over())))
            op = obs_op_ss.to_qutip(
                tuple(
                    sorted(
                        (site for site in obs_op_ss.acts_over()),
                        key=lambda lbl: names_to_pos[lbl],
                    )
                )
            )
            if ss is not None:
                rho_loc = rho.ptrace(ss)
                expect_values[key].append(np.real(qutip.expect(rho_loc, op)))
            else:
                expect_values[key].append(np.real(qutip.expect(rho, op)))

    qutip.mesolve(
        H=hamiltonian.to_qutip(),
        rho0=k_0.to_qutip(tuple(system.sites)),
        tlist=timespan,
        e_ops=callback_compute_obs,
    )
    simulation["ev_obs_ex"] = expect_values


def compute_exact_dynamics(simulations: dict,
                           simulation_context: dict,
                           filename_prefix: Optional[str] = None,):
    """
    If was not previously stored in simulations, compute the exact dynamics.
    Store the result in filename_prefix.
    """
    timespan = simulation_context["timespan"]
    if "exact" not in simulations:
        simulations["exact"] = {
            "parms": PARAMS,
            "date": datetime.now(),
            "name": "exact",
            "ev_obs_ex": [],
            "times": timespan,
        }
        run_exact_simulation(simulations["exact"], simulation_context)

        with open(f"{SIMULATIONS_FILE_PREFIX}.pkl", "bw") as out_file:
            pickle.dump(simulations, out_file)


# ### Mean-Field Projections for Basis Construction
# To construct a new basis and compute the matrix elements of ${\cal G}$ and
# ${\cal H}$, it is necessary to approximate the instantaneous state
# ${\sigma}(t) \propto \exp(-{\bf K}(t))$. While exact computation of
# ${\bf K}(t)$ is often impractical, except in special cases such as when
# ${\bf K}(t)$ is a one-body operator, a suitable approximation suffices
# to compute the evolution.
#
# Mean-Field Approach Overview
# The Mean-Field (MF) approach is a widely used method in fields like condensed
# matter physics (e.g., superconductive models) and quantum information for
# simplifying complex systems. Its central idea is to approximate $n$-body
# interactions in a quantum system using effective one-body (or higher-order)
# averages.
#
# This approach can be formalized through Mean-Field projections:
#
# MF projections, denoted as
# $\pi^{\rm MF}_{B}: {\cal A} \rightarrow {\cal A}_B$, map the full algebra of
# observables, ${\cal A}$, to a subalgebra, ${\cal A}B$, relative to a
# suitable basis $B{\rm MF}$.
# For product-state MF, the basis $B_{\rm MF} = B_{\rm prod}$ consists of local
# observables, such that $B_{\rm prod} = \bigsqcup_i B_i$, where $B_i$ spans a
# local subalgebra ${\cal A}_{B_i}$. The resulting Max-Ent states are product
# states: $\sigma^{\rm MF} = \bigotimes_i \sigma_i$.
# Similarly, bosonic and fermionic MF states can be defined using quadratic
# forms (e.g., creation and annihilation operators) with the basis
# $B_{\rm MF} = B_{\rm quad}$.
# Mean-Field Projection Formula
# The MF projection of an observable ${\bf O}$ is given by:
#
# $$\pi^{\rm MF}_{\tilde{B}, \sigma^{\rm MF}}({\bf O}) =
#      \sum_{{\bf Q} \in \tilde{B}} \big({\bf Q} -
#     \langle {\bf Q} \rangle_{\sigma^{\rm MF}}\big)
#     \frac{\partial \langle {\bf O} \rangle_{\sigma^{\rm MF}}}
#                {\partial \langle {\bf Q} \rangle_{\sigma^{\rm MF}}}
#     + \langle {\bf O} \rangle_{\sigma^{\rm MF}}$$
#
# This projection ensures consistency with the mean-field state,
# $\sigma^{\rm MF}$.
#
# The mean-field state, $\sigma^{\rm MF}$, satisfies the self-consistent
# equation:
#
# $$\sigma^{\rm MF} = \frac{\exp(- \pi^{\rm MF}_{\tilde{B},
#                                   \sigma^{\rm MF}}({\bf O}))}{{\rm Tr}
#                 \exp(- \pi^{\rm MF}_{\tilde{B}, \sigma^{\rm MF}}({\bf O}))}$$
#
# The MF projection $\sigma^{\rm MF} = \exp(-\tilde{\bf K})$ of a state
# $\exp(-{\bf K})$ can be computed iteratively using:
#
# $$\tilde{\bf K}^{(i+1)} = \pi^{\rm MF}_{\tilde{B},
#                                   \sigma_i^{\rm MF}}({\bf K})$$
#
# Here, $\sigma_i^{\rm MF} = \exp(-\tilde{\bf K}^{(i)})$, and the initial guess
# $\sigma^{\rm MF (0)}$ can be the system's initial state $\sigma(0)$ or a
# prior MF state $\sigma^{\rm MF}(T_{n-1})$.
#
# #### Benefits and Challenges
# Advantages: The self-consistent MF approach often converges rapidly,
# providing manageable approximations of complex states.
#
# Caveats: In ill-conditioned scenarios, convergence may fail or lead to
# non-trivial fixed points, requiring further refinements beyond this method's
# scope.
#
# ### Modified Hierarchical Basis
#
# The **Modified Hierarchical Basis** is constructed from the evolved state
# just before actualization. This state is then projected onto $m_0$-body
# observables, ensuring that the basis remains efficient and adapted to the
# system's current dynamics.

# In[7]:


def run_maxent_simulation(current_simulation: dict,
                          estimate_error: Callable,
                          simulation_context: dict):
    """
    Run the simulation of a Max-Ent dynamic.
    Results are stored as entries in `current_simulation`.
    """
    current_simulation_parms = current_simulation["parms"]
    chosen_depth = current_simulation_parms["chosen_depth"]
    eps_tol = current_simulation_parms["eps"]
    m_0 = current_simulation_parms["m0"]
    timespan = simulation_context["timespan"]
    k_0 = simulation_context["k_0"]
    sigma_0 = simulation_context["sigma0"]
    hamiltonian = simulation_context["H"]

    # Initializetimespan variables to track errors, saved cut times,
    # expectation values, and commutators
    actualizations = 0
    bases_deep = 0

    saved_cut_times_index = current_simulation.setdefault(
        "saved_cut_times_index", [0]
    )
    ev_obs_maxent = current_simulation.setdefault(
        "ev_obs_maxent", {key: [] for key in OBSERVABLES}
    )
    ev_obs_maxent["time"] = []
    no_acts = current_simulation.setdefault("no_acts", [0])
    number_of_commutators = current_simulation.setdefault(
        "number_of_commutators", []
    )

    # to be used in storing the values of the partial sum
    local_bound_error = current_simulation.setdefault(
        "local_bound_error", [])
    # to be used in storing the spectral norm of the Hij tensor at each
    # actualization of the (orthonormalized) basis
    spectral_norm_hij_tensor = current_simulation.setdefault(
        "spectral_norm_hij_tensor", []
    )
    # Norm of the orthogonal component of the commutators
    instantaneous_w_errors = current_simulation.setdefault(
        "instantaneous_w_errors", [])

    # Start the computation

    # Start the computation

    def callback_compute_obs_maxent(t, rho):
        ev_obs_maxent["time"].append(t)
        exp_vals_dict = rho.expect(OBSERVABLES)
        for key, exp_val in exp_vals_dict.items():
            ev_obs_maxent[key].append(exp_val)

    print("    * compute initial expectation values")
    # Compute the initial observables
    callback_compute_obs_maxent(0, sigma_0)

    # Compute the scalar product operator used for orthogonalization

    print("    * build scalar product from", type(sigma_0))
    sp_local = me.fetch_covar_scalar_product(sigma=sigma_0)
    local_t_value = 0.0

    # Build the initial Krylov basis and orthogonalize it
    print("    * build initial hierarchical basis with :", chosen_depth,"elements")
    hbb_act = me.build_hierarchical_basis(
        generator=hamiltonian, seed_op=k_0, deep=chosen_depth
    )
    print("    * orthogonalize the basis :")
    orth_basis_act = me.orthogonalize_basis(basis=hbb_act, sp=sp_local)
    bases_deep = len(orth_basis_act)
    print("       - basis length:", bases_deep)
    number_of_commutators.append(bases_deep)

    # Compute the Hamiltonian tensor for the basis
    print("    * build Hij tensor and errors")
    hij_tensor_act, w_errors = me.fn_hij_tensor_with_errors(
        generator=hamiltonian, basis=orth_basis_act, sp=sp_local
    )
    instantaneous_w_errors.append(np.real(w_errors))

    spectral_norm_hij_tensor.append(linalg.norm(hij_tensor_act))

    # Initial condition
    print("    * build initial coordinates.")
    phi0_proj_act = me.project_op(k_0, orth_basis_act, sp_local)

    # Initialize lists to store time-evolved values
    # phi_at_timet = [phi0_proj_act]
    # K_at_timet = [K0.to_qutip()]
    # sigma_at_timet = [me.safe_expm_and_normalize(K_at_timet[0])]

    # Iterate through the time steps
    print("    * start loop")
    for time in timespan[1:]:
        # Evolve the state phi(t) for a small time window
        phi_local = np.real(
            linalg.expm(hij_tensor_act * (time - local_t_value)
                        ) @ phi0_proj_act
        )

        # Compute the new K-state from the orthogonal basis and phi(t)
        k_local = me.k_state_from_phi_basis(
            phi=-phi_local, basis=orth_basis_act)

        # Normalize to obtain the updated density matrix sigma(t)
        sigma_local = safe_exp_and_normalize(k_local)[0]
        # Record expectation values of the observable
        callback_compute_obs_maxent(time, sigma_local)

        # Calculate the local error bound using partial sums
        local_bound_error.append(estimate_error(phi_local, w_errors))

        # Check if the local error exceeds the threshold
        if abs(local_bound_error[-1]) >= eps_tol:
            print(
                "         error bound=",
                local_bound_error[-1],
                f".\n          Updating basis with deep {chosen_depth}",
                datetime.now(),
            )
            # If positive, perform actualization
            actualizations = actualizations + 1

            # Log errors at specific intervals for debugging
            if list(timespan).index(time) % 50 == 0:
                print("         error", time)

            # Update the local time value and save the cut time index
            local_t_value = time
            saved_cut_times_index.append(list(timespan).index(time))

            # Map the K-local state onto a Mean-Field state, retaining only
            # its one-body correlations, to be used in sp
            print("    * rebuild mean field state")
            _, sigma_act = me.mft_state_it(k_local, sigma_local, max_it=10)

            print("          now sigma_act is of type", type(sigma_act))
            # Recompute the scalar product using the MF state
            sp_local = me.fetch_covar_scalar_product(sigma=sigma_act)

            # The new basis is spanned from the k_local state
            print("    * rebuild hierarchical basis")
            hbb_act = me.build_hierarchical_basis(
                generator=hamiltonian, seed_op=k_local, deep=chosen_depth
            )

            # The growth in complexity of the basis is arrested by projecting
            # this basis onto simpler basis
            # composed of $nmax$-body observables only, with $nmax$ much
            # smaller than the size of the system.

            print(f"      * project to {m_0}-bodies")
            hbb_act = [me.project_operator_to_m_body(
                op, m_0, sigma_act) for op in hbb_act]

            print("      * orthogonalizing")
            orth_basis_act = me.orthogonalize_basis(basis=hbb_act, sp=sp_local)
            bases_deep = len(orth_basis_act)

            # Recompute the Hamiltonian tensor and project the state
            print("      * building H")
            hij_tensor_act, w_errors = me.fn_hij_tensor_with_errors(
                generator=hamiltonian, basis=orth_basis_act, sp=sp_local
            )
            print(f"          bases updated with a depth of {
                  bases_deep}", datetime.now())
            instantaneous_w_errors.append(np.real(w_errors))
            spectral_norm_hij_tensor.append(linalg.norm(hij_tensor_act))
            phi0_proj_act = me.project_op(k_local, orth_basis_act, sp_local)

        number_of_commutators.append(number_of_commutators[-1])
        no_acts.append(actualizations)

    current_simulation["velocity_mu"] = np.array(
        spectral_norm_hij_tensor)
    times_act = np.array(saved_cut_times_index)
    current_simulation["times_act"] = times_act
    current_simulation["velocity_PS"] = np.array(
        [
            1 / (times_act[i + 1] - times_act[i])
            for i in range(len(times_act) - 1)
        ]
    )


# ### Test the simulation with different choices of the parameters.
#

# In[ ]:


def main():
    """
    main routine.
    """

    print("load previous simulations")
    simulations = load_simulations()
    print("  loaded:", tuple(simulations.keys()))
    print("loading context:")
    simulation_context = setup_simulation_context()
    print("  done")
    print("computing exact dynamics")
    compute_exact_dynamics(simulations, simulation_context)
    print("  done")
    
    cases = []
    for m_0 in [1, 2, 3]:
        for depth in [5, 3]:
            for epstol in [0.1, 0.01, 0.001]:
                cases.append(
                    {
                        "m0": m_0,
                        "chosen_depth": depth,
                        "eps": epstol,
                    }
                )

    for estimate_error in (estimate_error_by_weights,
                           estimate_error_by_partial_sum):
        for approx_parms in cases:
            simulation_name = (
                f"({approx_parms['m0']},"
                f"{approx_parms['chosen_depth']},"
                f"{approx_parms['eps']},"
                f"{repr(estimate_error)[28:][:-19]})"
            )
            simulations[simulation_name] = {
                "parms": approx_parms,
                "date": datetime.now(),
                "name": simulation_name,
            }
            try:
                print("Simulation", simulation_name)
                run_maxent_simulation(
                    simulations[simulation_name],
                    estimate_error,
                    simulation_context)
            except Exception as e:
                print(e)
                raise
            # except linalg.ArpackNoConvergence:
            #    continue

            with open(f"{SIMULATIONS_FILE_PREFIX}.pkl", "bw") as out_file:
                pickle.dump(simulations, out_file)


if __name__ == "__main__":
    main()
