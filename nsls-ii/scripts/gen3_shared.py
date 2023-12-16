import sext_tools as st
import numpy as np
from apsopt.util.pydantic.vocs import GVOCS
import sys

def prepare(MODE):
    # Already has scaling for SM2B
    dfknobs = st.KNOBS_DATAFRAMES[MODE]

    # DO NOT CHANGE - USED TO CALCULATE ABSOLUTE K2L
    initial_values_ref = st.get_initial_values_ref(MODE)

    # CHANGE THIS TO VALUES FROM RING IF NEEDED
    initial_values = initial_values_ref.copy()
    initial_values['SM1A'] *= 1.0
    initial_values['SM1B'] *= 1.0

    # optimizer variables in units of K2L
    variables_list = st.MODE_GROUPS[MODE]
    variables = {}
    for k in variables_list:
        if k == 'SVD0':
            variables[k] = [-1.0,1.0]
        else:
            variables[k] = [-np.abs(initial_values_ref[k])*0.15,
                            np.abs(initial_values_ref[k])*0.15]
    objectives_active = {'LT': 'MAXIMIZE', 'EFF':'MAXIMIZE'}
    initial_variable_values = {k:0.0 for k in variables.keys()}

    # Manual modifications to spoil quality
    if MODE == 'BARE_SH1_SH3_SH4_SL1_SL2_SL3_SVD0':
        initial_variable_values['SH1'] -= 0.2
        initial_variable_values['SH3'] -= 0.08
        initial_variable_values['SL2'] -= 0.5
        initial_variable_values['SL3'] -= 0.5

    gvocs = GVOCS(variables=variables,
                variables_active=variables,
                objectives=objectives_active,
                objectives_active=objectives_active)
    
    return dfknobs, initial_values_ref, initial_values, gvocs, initial_variable_values


def compute_family_k2l_from_knob_k2l(dfknobs, knobs_dict, ivals, debug=False):
    group_relative_strengths = st.knob_strengths_to_group_strengths(dfknobs, knobs_dict)
    for k in group_relative_strengths:
        if debug:
            print(f'{k:7s}: {ivals[k]:+.3f} + {group_relative_strengths[k]:+.3f} -> {ivals[k]+group_relative_strengths[k]:+.3f}')
        group_relative_strengths[k] = group_relative_strengths[k] + ivals[k]
    return group_relative_strengths

YOSHI_PATH = "/nsls2/users/yhidaka/git_repos/nsls2scripts3/shifts/2023-12-16_APSU_DA_MA"


def get_eval_f(TEST_MODE, gvocs, dfknobs):
    if TEST_MODE:
        from opt_funcs_nsls import make_dummy_eval
        eval_f = make_dummy_eval(gvocs)
    else:
        sys.path.insert(0, YOSHI_PATH)
        import opt_funcs

        def knobs_to_family(inputs_dict):
            d = compute_family_k2l_from_knob_k2l(dfknobs, inputs_dict, True)
            return opt_funcs.master_eval_function(d, meas_bxb_tunes=True, meas_lifetime=True, meas_inj_eff=True)

        eval_f = knobs_to_family
    return eval_f


def get_raw_eval_f():
    sys.path.insert(0, YOSHI_PATH)
    import opt_funcs
    return opt_funcs.master_eval_function