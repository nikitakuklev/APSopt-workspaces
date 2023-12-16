import pysdds
import pandas as pd
import re, subprocess, sys, pathlib
import numpy as np

ELEGANT_WIN = r"C:\Users\boss\Downloads\Elegant-x64\elegant.exe"
ELEGANT_UNIX = "elegant"

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

MODES = {'BARE_SH1P12345_SH3P12345_SH4P12345_SL1_SL2_SL3_SVD0':19,# 5+5+5+3+1 = 19
         'BARE_SH1P12345_SH3P12345_SH4P12345_SL1_SL2_SL3':18,
         'BARE_SH1P12345_SH3P12345_SH4_SL1_SL2_SL3_SVD0':15, # 5+5+1+3+1 = 15
         'BARE_SH1P12345_SH3P12345_SH4_SL1_SL2_SL3':14,
         'BARE_SH1P12345_SH3_SH4_SL1_SL2_SL3_SVD0':11, # 5+1+1+3+1 = 11
         'BARE_SH1_SH3_SH4_SL1_SL2_SL3_SVD0':7, # 1+1+1+3+1 = 7
         'DW_SH1P12345_SH3N_SH4N_SL1N_SL2N_SL3N_SH1DW081828_SH3DW081828_SH4DW081828_SVD0':20, # 5+1+1+1+1+1+3+3+3+1 = 20
         'DW_SH1P12345_SH3N_SH4N_SL1N_SL2N_SL3N_SH1DW081828_SH3DW081828_SH4DW081828':19,
         'DW_SH1N_SH3N_SH4N_SL1N_SL2N_SL3N_SH1DW081828_SH3DW081828_SH4DW081828_SVD0':16, # 1+1+1+1+1+1+3+3+3+1 = 16
         'DW_SH1N_SH3N_SH4N_SL1N_SL2N_SL3N_SH1DW081828_SH3DW081828_SH4DW081828': 15,
         }

MODE_GROUPS = {}
for m,cnt in MODES.items():
    groups = []
    parts = m.split('_')[1:]
    for p in parts:
        if p in ['SH1','SH3','SH4','SL1','SL2','SL3']:
            groups.append(p)
        elif p in ['SH1N','SH3N','SH4N','SL1N','SL2N','SL3N']:
            groups.append(p)
        elif p == 'SVD0':
            groups.append(p)
        elif p.endswith('P12345'):
            groups.extend(f"{p.split('P12345')[0]}-P{i}" for i in [1,2,3,4,5])
        elif p.endswith('DW081828'):
            groups.extend(f"{p.split('DW081828')[0]}-DW{i:02d}" for i in [8,18,28])
    
    MODE_GROUPS[m] = groups
    assert len(groups) == cnt, f'{m} {groups} {len(groups)} {cnt}'

PENTANT_NUMBERS = {1:[23,24,25,26,27,28],
                   2:[29,30,1,2,3,4],
                   3:[5,6,7,8,9,10],
                   4:[11,12,13,14,15,16],
                   5:[17,18,19,20,21,22]}

INITIAL_VALUES_REF_K2L_BARE = {
    'SM1A': -23.68063424*0.2,
    'SM2B': 28.64315469*0.25,
    'SM1B': -25.94603546*0.2,
    'SL3': -29.46086061*0.2,
    'SL2': 35.67792145*0.2,
    'SL1': -13.27160605*0.2,
    'SH4': -15.82090071*0.2,
    'SH3': -5.85510841*0.2,
    'SH1': 19.8329121*0.2,
}
INITIAL_VALUES_REF_K2L_DW = INITIAL_VALUES_REF_K2L_BARE.copy()

HARMONIC_TOP_GROUPS = ['SH1','SH3','SH4','SL1','SL2','SL3']
SUBGROUP_MAP = {
    'SL1': ['SL1-P1','SL1-P2','SL1-P3','SL1-P4','SL1-P5',],
    'SL2': ['SL2-P1','SL2-P2','SL2-P3','SL2-P4','SL2-P5',],
    'SL3': ['SL3-P1','SL3-P2','SL3-P3','SL3-P4','SL3-P5',],
    'SH1': ['SH1-P1','SH1-DW28','SH1-P2','SH1-P3','SH1-DW08','SH1-P4','SH1-DW18','SH1-P5',],
    'SH3': ['SH3-P1','SH3-DW28','SH3-P2','SH3-P3','SH3-DW08','SH3-P4','SH3-DW18','SH3-P5',],
    'SH4': ['SH4-P1','SH4-DW28','SH4-P2','SH4-P3','SH4-DW08','SH4-P4','SH4-DW18','SH4-P5',],
    'SM1A': ['SM1A-P1','SM1A-P2','SM1A-P3','SM1A-P4','SM1A-P5',],
    'SM1B': ['SM1B-P1','SM1B-P2','SM1B-P3','SM1B-P4','SM1B-P5',],
    'SM2B': ['SM2B-P1','SM2B-P2','SM2B-P3','SM2B-P4','SM2B-P5',],
}

SUBGROUP_MAP_INV = {}
for k,v in SUBGROUP_MAP.items():
    for x in v:
        SUBGROUP_MAP_INV[x] = k


KNOBS_DATAFRAMES = {}
for m in MODES:
    index = MODE_GROUPS[m].copy()
    indexnosvd = index.copy()
    columns = MODE_GROUPS[m].copy()
    if 'SVD0' in columns:
        index.remove('SVD0')
        indexnosvd.remove('SVD0')
        index.extend(['SM1A', 'SM1B', 'SM2B'])
    else:
        pass
    df = pd.DataFrame(np.zeros((len(index), len(columns))), index=index, columns=columns)
    if 'SVD0' in df.columns:
        df.loc[['SM1A', 'SM1B', 'SM2B'],'SVD0'] = [-0.663666, 0.744816, -0.069260]
    for i in indexnosvd:
        df.loc[i,i] = 1.0
    if 'SVD0' in df.columns:
        # impact of K2L difference
        df.loc['SM2B',:] *= 0.25/0.2
    KNOBS_DATAFRAMES[m] = df

def get_initial_values_ref(mode):
    """ Get initial k2l values for families in mode"""
    if 'BARE' in mode:
        ivals = INITIAL_VALUES_REF_K2L_BARE
    elif 'DW' in mode:
        ivals = INITIAL_VALUES_REF_K2L_DW
    else:
        raise
    if mode not in MODES:
        raise Exception(f'Unknown mode {mode}')
    vals = {}
    for g in MODE_GROUPS[mode]:
        gname = g
        if gname in ['SH1N','SH3N','SH4N','SL1N','SL2N','SL3N']:
            gname = gname[:-1]
        if gname in ivals:
            vals[g] = ivals[gname]
        elif gname  == 'SVD0':
            for sg in ['SM1A', 'SM1B', 'SM2B']:
                vals[sg] = ivals[sg]
        else:
            vals[g] = ivals[SUBGROUP_MAP_INV[g]]
    assert len(vals) >= len(MODE_GROUPS[mode])
    return vals
    
def verify_funcs():
    for m in MODES:
        get_initial_values_ref(m)


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

def knob_strengths_to_group_strengths(knob_matrix: pd.DataFrame, knob_dict: dict[str,float]):
    """ Take knob dataframe, output per-family strengths """
    assert len(knob_dict) == knob_matrix.shape[1]

    data = {k:0.0 for k in knob_matrix.index}
    for c in knob_matrix.columns:        
        for g in knob_matrix.index:
            data[g] += knob_matrix.loc[g,c]*knob_dict[c]
    return data




