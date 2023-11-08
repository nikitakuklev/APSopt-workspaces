import logging
import subprocess
import numpy as np
import traceback
from apsopt.util.pydantic.vocs import GVOCS

logger = logging.getLogger(__name__)

global SEXTUPOLES_EXTRA, SEXTUPOLES_SCRIPT, variables_ch, kgroups

MODE = 'direct'
LIFETIME_SCRIPT = "measLifeTimeAdaptive.py"
EFF_SCRIPT = "/home/oxygen/SHANG/oag/apps/src/tcltkapp/oagapp/measInjEffic648"
kgroups = {}

def get_settings_from_groups(knob_groups):
    global SEXTUPOLES_EXTRA, SEXTUPOLES_SCRIPT, variables_ch
    variables_ch = []
    var_dict = {}
    ival_dict = {}
    objectives_ch = ['LT', 'EFF']
    obj_dict = {obj: 'MAXIMIZE' for obj in objectives_ch}

    for k,v in knob_groups.items():
        name = k#f'Family{k+1}'
        variables_ch.append(name)
        var_dict[name] = [-20.0, 20.0]
        ival_dict[name] = 0.0

    gvocs_global = GVOCS(**{
        'variables': var_dict,'variables_active': var_dict.copy(),
        'objectives': obj_dict,'objectives_active': obj_dict.copy()
        }
    )
    ref_point = {'EFF': 0.2, 'LT': 1.0}
    SEXTUPOLES_EXTRA = ['-setPVRelative','1','-dryRun','0']
    SEXTUPOLES_SCRIPT = "/home/oxygen/SHANG/oag/apps/src/tcltkapp/oagapp/changeSRSextupoles"
    return gvocs_global, ref_point, ival_dict


def exec_tcl(cmd):
    prefix = ''
    p = subprocess.run(prefix + cmd, shell=True, capture_output=True, check=True, encoding='ascii')
    if 'error' in p.stdout.lower() or 'error' in p.stderr.lower():
        raise ValueError(f"Script error {p.stdout=} {p.stderr=}")
    return p

def exec_no_shell(cmd):
    p = subprocess.run(cmd, capture_output=True, check=True, encoding='ascii')
    if 'error' in p.stdout.lower() or 'error' in p.stderr.lower():
        raise ValueError(f"Script error {p.stdout=} {p.stderr=}")
    return p


def setter_direct(inputs_dict):
    global kgroups
    assert len(inputs_dict) == len(kgroups), f'{len(inputs_dict)} {len(kgroups)}'
    vars = []
    vals = []
    final_vals = {}
    for k,v in inputs_dict.items():
        gr = kgroups[k]
        for mag,mag_scaling in gr.items():
            if mag not in final_vals:
                final_vals[mag] = 0.0
            final_vals[mag] += v*mag_scaling

    vars = list(final_vals.keys())
    vals = list(final_vals.values())

    vars_str = ' '.join([f'{k}' for k in vars])
    values_str = ' '.join([f'{v:10.5e}' for v in vals])
    arg_list = [SEXTUPOLES_SCRIPT] + SEXTUPOLES_EXTRA + ['-tagList', f'"{vars_str}"',
            '-valueList', f'"{values_str}"']
    final_command = ' '.join(arg_list)
    #print(final_command)
    p = exec_tcl(final_command)
    

def eval_lifetime(inputs_dict):
    logger.info(f'Setting sextupoles to {inputs_dict}')
    if 'fidelity' in inputs_dict:
        fidelity = inputs_dict.pop('fidelity')
        assert 0.0 <= fidelity <= 1.0
    else:
        fidelity = None
    vars_str = ' '.join([f'{k}' for k, v in inputs_dict.items()])
    values_str = ' '.join([f'{v:21.15e}' for k, v in inputs_dict.items()])
    arg_list = [SEXTUPOLES_SCRIPT, SEXTUPOLES_EXTRA, '-tagList', f'"{vars_str}"',
            '-valueList', f'"{values_str}"']
    final_command = ' '.join(arg_list)
    p = exec_tcl(final_command)

     # Lifetime
    logger.info('Lifetime measurement START')
    if fidelity is not None:
        ft = 10 + (20*fidelity)
        arg_list = ['python', LIFETIME_SCRIPT, '-minTime',ft,'-maxTime',ft]
        p = exec_no_shell(arg_list)
    else:
        arg_list = ['python ' + LIFETIME_SCRIPT]
        final_command = ' '.join(arg_list)
        p = exec_tcl(final_command)

    try:
        objective_lifetime = float(p.stdout.strip())
    except:
        logger.error('Lifetime calc failed')
        logger.error(f'{traceback.format_exc()}')
        objective_lifetime = np.nan

    return {'LT': objective_lifetime}


def master_eval_function(inputs_dict):
    """ Evaluation function that maps variables to objectives """
    
    # Set sextupoles
    logger.info(f'Setting sextupoles to {inputs_dict}')
    setter_direct(inputs_dict)

    # Lifetime
    logger.info('Lifetime measurement START')
    arg_list = ['python ' + LIFETIME_SCRIPT]
    final_command = ' '.join(arg_list)
    try:
        p = exec_tcl(final_command)
        objective_lifetime = float(p.stdout.strip())
    except:
        logger.error('Lifetime calc failed')
        logger.error(f'{traceback.format_exc()}')
        objective_lifetime = np.nan

    
    # Injection efficiency
    logger.info('Efficiency measurement START')
    arg_list = [EFF_SCRIPT, '-cycles 3']
    final_command = ' '.join(arg_list)
    p = exec_tcl(final_command)
    try:
        objective_eff = float(p.stdout.strip())
    except:
        logger.error('Efficiency script failed')
        logger.error(f'{traceback.format_exc()}')
        objective_eff = np.nan

    return {'LT': objective_lifetime, 'EFF': objective_eff}


def master_eval_function_dummy(inputs_dict):
    """ Test function """
    global variables_ch
    assert len(inputs_dict) == len(variables_ch)
    f = 1.0
    for k,v in inputs_dict.items():
        f *= (v/10-1)**2
    efficiency = f
    for k,v in inputs_dict.items():
       efficiency *= np.sin(0.5*v*np.pi)
    lifetime = f
    for k,v in inputs_dict.items():
       lifetime *= np.sin(0.5*v*np.pi) if k != 'Knob5' else np.cos(0.5*v*np.pi)

    return {'LT': sum(np.array(list(inputs_dict.values()))-5), 'EFF': sum(np.array(list(inputs_dict.values()))-2)**2}
    #return {'LT': -lifetime, 'EFF': -efficiency}





