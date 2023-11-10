import pysdds
import pandas as pd
import re, subprocess, sys, pathlib
import numpy as np

"""
A total of 9 sextupole families (3 chromatic [SM*], 6 harmonic): SH1, SH3, SH4, SL1, SL2, SL3, SM1A, SM1B, SM2B

A total of 54 independent setpoint PVs for sextupoles:

SH[134]-DW[08|18|28] = 3x3 = 9 knobs
SH[134]-P[1-5] = 3x5 = 15 knobs
SL[123]-P[1-5] = 3x5 = 15 knobs
SM[1A|1B|2B]-P[1-5] = 3x5 = 15 knobs

PS current limits:
0-151 A for SM2B
0-110 A for the rest

Pentant 1: C23-C28
Pentant 2: C29-C30, C01-C04
Pentant 3: C05-C10
Pentant 4: C11-C16
Pentant 5: C17-C22

"""

ELEGANT_WIN = r"C:\Users\boss\Downloads\Elegant-x64\elegant.exe"
ELEGANT_UNIX = "elegant"

ele_get_beamline = f"""
&run_setup
    default_order = 3,
    use_beamline = RING,
    lattice = 20230915_aphla_bare_w_xbpms_MAG_bare.lte,
    p_central_mev = 3e3,
    concat_order = 0,
    parameters = %s.param
&end

&run_control &end

&bunched_beam &end

&track &end
"""

def run_elegant(taskfile):
    cwd = pathlib.Path.cwd()
    if sys.platform == 'win32':
        ELEGANT = ELEGANT_WIN
        rpnpath = f"-rpnDefns={str(cwd/'lattice'/'defns.rpn')}".replace('\\','\\\\')
    else:
        ELEGANT = ELEGANT_UNIX
        rpnpath = f"-rpnDefns={str(cwd/'lattice'/'defns.rpn')}"    
    p = subprocess.run([ELEGANT, taskfile, rpnpath],
                        capture_output=True,
                        cwd=cwd/'lattice'
                        )
    if p.returncode != 0:
        print(p.stdout.decode())
        print(p.stderr.decode())
        raise Exception(f'Error running elegant {taskfile}')
    return p    

def get_lattice_beamsheet():
    with open('lattice/sext.ele','w') as f:
        f.write(ele_get_beamline)
    run_elegant('sext.ele')

    sdds = pysdds.read('lattice/sext.param')
    return sdds


def get_twiss(param_file, n_pages=1):
    ele_get_sext = ""
    ele_get_sext += f"""
    &run_setup
        default_order = 3,
        use_beamline = RING,
        lattice = 20230915_aphla_bare_w_xbpms_MAG_bare.lte,
        p_central_mev = 3e3,
        concat_order = 0,
    &end
    """
    if param_file is not None:
        ele_get_sext += f"""    
        &load_parameters
            allow_missing_elements = 1,
            filename = {param_file},
            allow_missing_parameters = 1,
            change_defined_values = 0,
        &end
        """
    ele_get_sext += f"""
    &run_control
        n_steps = {n_pages},
    &end

    &twiss_output
        output_at_each_step = 1,
        filename = sext.twi,
        matched = 1,
        radiation_integrals = 0,
    &end

    &bunched_beam &end

    &track &end
    """
    with open('lattice/sext_chroma.ele','w') as f:
        f.write(ele_get_sext)

    run_elegant('sext_chroma.ele')
    sdds = pysdds.read('lattice/sext.twi')
    return sdds


def make_df(sextupoles, delta=1.0):
    df = pd.DataFrame({'ElementName':sextupoles,'ElementParameter':'K2',
                       'ParameterValue':delta,'ParameterMode':'differential'})
    return df

def sort_by_sector(v):
    order = [rf'.*?{i:02d}A|B$' for i in range(1,30)]
    out = []
    for o in order:
        for x in v:
            if re.match(o,x) is not None:
                #print(o,x)
                out.append(x)
                break
    assert len(out) == len(v), f'{len(out)} {len(v)} {len(order)} {v}'
    return out


def make_knobs(mode):
    """
    NSLS-II
    SL1G2C01A: KSEXT, L=0.2, K2=-13.27160605
    SL2G2C01A: KSEXT, L=0.2, K2=35.67792145
    SL3G2C01A: KSEXT, L=0.2, K2=-29.46086061
    SM1G4C01A: KSEXT, L=0.2, K2=-23.68063424
    SM2G4C01B: KSEXT, L=0.25, K2=28.64315469
    SM1G4C01B: KSEXT, L=0.2, K2=-25.94603546
    SH4G6C01B: KSEXT, L=0.2, K2=-15.82090071
    SH3G6C01B: KSEXT, L=0.2, K2=-5.85510841
    SH1G6C01B: KSEXT, L=0.2, K2=19.8329121
    SH1G2C02A: KSEXT, L=0.2, K2=19.8329121
    SH3G2C02A: KSEXT, L=0.2, K2=-5.85510841
    SH4G2C02A: KSEXT, L=0.2, K2=-15.82090071
    SM1G4C02A: KSEXT, L=0.2, K2=-23.68063424
    SM2G4C02B: KSEXT, L=0.25, K2=28.64315469
    SM1G4C02B: KSEXT, L=0.2, K2=-25.94603546
    SL3G6C02B: KSEXT, L=0.2, K2=-29.46086061
    SL2G6C02B: KSEXT, L=0.2, K2=35.67792145
    SL1G6C02B: KSEXT, L=0.2, K2=-13.27160605
    """    
            
    groups = {}
    groups_direct = {}

    family_patterns = {
        'SL1' : lambda i: [f'SL1G2C{i:02d}A',f'SL1G6C{i+1:02d}B'],
        'SL2' : lambda i: [f'SL2G2C{i:02d}A',f'SL2G6C{i+1:02d}B'],
        'SL3' : lambda i: [f'SL3G2C{i:02d}A',f'SL3G6C{i+1:02d}B'],
        'SH1' : lambda i: [f'SH1G6C{i:02d}B',f'SH1G2C{i+1:02d}A'],
        'SH3' : lambda i: [f'SH3G6C{i:02d}B',f'SH3G2C{i+1:02d}A'],
        'SH4' : lambda i: [f'SH4G6C{i:02d}B',f'SH4G2C{i+1:02d}A'],
        'SM1A' : lambda i: [f'SM1G4C{i:02d}A',f'SM1G4C{i+1:02d}A'],
        'SM1B' : lambda i: [f'SM1G4C{i:02d}B',f'SM1G4C{i+1:02d}B'],
        'SM2B' : lambda i: [f'SM2G4C{i:02d}B',f'SM2G4C{i+1:02d}B'],
    }

    if mode == '9knobs': 
        # family_regexes = {
        #     'SL1' : r'SL1G2C\d{2}A|SL1G6C\d{2}B',
        #     'SL2' : r'SL2G2C\d{2}A|SL2G6C\d{2}B',
        #     'SL3' : r'SL3G2C\d{2}A|SL3G6C\d{2}B',
        # }
        for i, (f,p) in enumerate(family_patterns.items()):
            subg = [x for i in range(1,30,2) for x in p(i)]
            groups[f] = subg
    elif mode == '6and3':
        # Only use chromatic sextupoles for null space
        for g in ['SM1A','SM1B','SM2B']:
            subg = [x for i in range(1,30,2) for x in family_patterns[g](i)]
            groups[g] = subg
        for g in ['SH1','SH3','SH4','SL1','SL2','SL3']:
            subg = [x for i in range(1,30,2) for x in family_patterns[g](i)]
            groups_direct[g] = subg
    else:
        raise Exception(f'Unknown mode {mode}')
    
    cnt = sum(len(g) for g in groups.values()) + sum(len(g) for g in groups_direct.values())
    assert cnt == 30*9, f'{cnt} {30*9}'

    #for k in groups:
    #    groups[k] = sort_by_sector(list(set(groups[k])))

    return groups, groups_direct, {k:i for i,k in enumerate(family_patterns.keys())}

def calculate_null_knobs(mode):
    groups, groups_direct, indices = make_knobs(mode)
    g1 = list(groups.values())[0]
    df_list = [make_df(g1,delta=0.0)] + [make_df(g, delta=0.1) for g in groups.values()]  

    sdds = pysdds.SDDSFile.from_df(df_list)
    #sdds.set_mode('ascii')
    pysdds.write(sdds, 'lattice/nullknobs.param', overwrite=True)

    twi = get_twiss('nullknobs.param', len(df_list))
    parameters = ['dnux/dp','dnuy/dp']
    data = {}
    for par in parameters:
        data[par] = twi[par].data

    dCdS = np.array([data['dnux/dp'][1:]-data['dnux/dp'][0], data['dnuy/dp'][1:]-data['dnuy/dp'][0]])/0.1
    u,s,vh = np.linalg.svd(dCdS, full_matrices=True)
    null_knobs = vh.T[:,2:]

    # Move null knobs to the overall matrix
    arr = np.zeros((len(indices),null_knobs.shape[1]))
    local_idx = {k:i for i,k in enumerate(groups)}
    for k,v in groups.items():
        arr[indices[k],:] = null_knobs[local_idx[k],:]

    # Add direct knobs as single-family columns
    if len(groups_direct) > 0:
        arr2 = np.zeros((len(indices),len(groups_direct)))
        for i,(k,v) in enumerate(groups_direct.items()):
            idx = indices[k]
            arr2[idx,i] = 1.0
        arr = np.hstack((arr,arr2))

    dfknobs = pd.DataFrame(arr, index=indices.keys(), columns=[f'SVD{i}' for i in range(null_knobs.shape[1])]
                           +list(groups_direct.keys()))

    return dfknobs


def get_chroma(knob: dict[str, float]):
    """ Apply null knob and verify that the chromaticity is constant """
    df = pd.DataFrame({'ElementName':knob.keys(),'ElementParameter':'K2',
                       'ParameterValue':knob.values(),'ParameterMode':'differential'})
    
    df2 = df.copy()
    df2.loc[:,'ParameterValue'] = 0.0
    df_list = [df2, df]

    sdds = pysdds.SDDSFile.from_df(df_list)
    #sdds.set_mode('ascii')
    pysdds.write(sdds, 'lattice/nullknobs.param', overwrite=True)
    twi = get_twiss('nullknobs.param', len(df_list))

    parameters = ['dnux/dp','dnuy/dp']
    data = {}
    for par in parameters:
        data[par] = twi[par].data

    return data

def make_knob_groups(groups: dict[str, list[str]], knob_matrix: pd.DataFrame, postfix=''):
    """ Take groups dict and knob dataframe, make knob groups """
    assert len(groups) == knob_matrix.shape[0]
    
    knob_groups = {}
    for c in knob_matrix.columns:
        data = {}
        for g,sextupoles in groups.items():
            for s in sextupoles:
                data[s+postfix] = knob_matrix.loc[g,c]
        knob_groups[c] = data
    return knob_groups




