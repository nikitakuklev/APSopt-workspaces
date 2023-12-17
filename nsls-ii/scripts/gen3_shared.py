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
    if 'SM1A' in initial_values:
        initial_values['SM1A'] *= 1.0
    if 'SM1B' in initial_values:
        initial_values['SM1B'] *= 1.0

    # optimizer variables in units of K2L
    variables_list = st.MODE_GROUPS[MODE]
    variables = {}
    for k in variables_list:
        if k == 'SVD0':
            variables[k] = [-1.0,1.0]
        else:
            variables[k] = [-np.abs(initial_values_ref[k])*0.12,
                            np.abs(initial_values_ref[k])*0.12]
    objectives_active = {'LT': 'MAXIMIZE', 'EFF':'MAXIMIZE'}
    initial_variable_values = {k:0.0 for k in variables.keys()}

    # Manual modifications to spoil quality
    if MODE == 'BARE_SH1_SH3_SH4_SL1_SL2_SL3_SVD0':
        initial_variable_values['SH1'] -= 0.2
        initial_variable_values['SH3'] -= 0.08
        initial_variable_values['SL2'] -= 0.5
        initial_variable_values['SL3'] -= 0.5
    elif MODE == 'DW_SH1N_SH3N_SH4N_SL1_SL2_SL3_SH1DW081828_SH3DW081828_SH4DW081828':
        initial_variable_values['SH1N'] -= 0.08
        initial_variable_values['SH3N'] -= 0.04
        initial_variable_values['SH4-DW08'] += 0.02
        initial_variable_values['SH3-DW08'] += 0.02
        initial_variable_values['SH1-DW08'] += 0.02
        initial_variable_values['SL2'] -= 0.15
        initial_variable_values['SL3'] -= 0.15
    elif MODE == 'DW_SH1N_SH3N_SH4N_SL1_SL2P12345_SL3P12345_SH1DW081828_SH3DW081828_SH4DW081828':
        initial_variable_values['SH1N'] -= 0.05
        initial_variable_values['SH3N'] += 0.05
        initial_variable_values['SH4N'] -= 0.04
        initial_variable_values['SL1'] += 0.08
        for i in range(1,6):
            initial_variable_values[f'SL2-P{i}'] -= 0.20 + i*0.07
        for i in range(1,6):
            initial_variable_values[f'SL3-P{i}'] += 0.20 - i*0.07
        initial_variable_values['SH1-DW08'] += 0.025
        initial_variable_values['SH3-DW08'] += 0.025
        initial_variable_values['SH4-DW08'] += 0.025
        initial_variable_values['SH1-DW18'] -= 0.025
        initial_variable_values['SH3-DW18'] -= 0.025
        initial_variable_values['SH4-DW18'] -= 0.025
        initial_variable_values['SH1-DW28'] -= 0.025
        initial_variable_values['SH3-DW28'] -= 0.025
        initial_variable_values['SH4-DW28'] -= 0.025     
        

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


def get_eval_f(TEST_MODE, gvocs, dfknobs, ivals):
    if TEST_MODE:
        from opt_funcs_nsls import make_dummy_eval
        eval_f = make_dummy_eval(gvocs)
    else:
        sys.path.insert(0, YOSHI_PATH)
        import opt_funcs

        def knobs_to_family(inputs_dict):
            d = compute_family_k2l_from_knob_k2l(dfknobs, inputs_dict, ivals, True)
            return opt_funcs.master_eval_function(d, meas_bxb_tunes=True, meas_lifetime=True, meas_inj_eff=True)

        eval_f = knobs_to_family
    return eval_f


def get_raw_eval_f():
    sys.path.insert(0, YOSHI_PATH)
    import opt_funcs
    return opt_funcs.master_eval_function
