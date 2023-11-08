import pysdds
import pandas as pd
import re, subprocess
import numpy as np

param_file = 'sext.param'

def make_df(sextupoles, delta=1.0):
    df = pd.DataFrame({'ElementName':sextupoles,'ElementParameter':'K2',
                       'ParameterValue':delta,'ParameterMode':'differential'})
    return df

def wrap_40(x):
    y = x if x <= 40 else x % 40
    return y

def sort_by_sector(v):
    order = [rf'^S{i}\D.*?' for i in range(1,41)]
    out = []
    for o in order:
        for x in v:
            if re.match(o,x) is not None:
                #print(o,x)
                out.append(x)
                break
    assert len(out) == len(v), f'{len(out)} {len(v)} {len(order)} {v}'
    return out


def make_knobs(mode='xiaobiao_10fold', keepS40=True):
    groups = {}


    if mode == 'eachcell':
        # 7 sextupole knobs without sector 40
        for gn,g in enumerate(['S{i}A:S1','S{i}A:S2','S{i}A:S3','S{i}A:S4','S{i}B:S1','S{i}B:S2','S{i}B:S2']):
            subg = [g.format(i=i) for i in range(1,41)]
            groups[gn] = subg

    elif mode == 'xiaobiao':
        # 20 fold symmetry
        groups = {i:[] for i in range(7)}
        postfixes = {0:'S1',1:'S2',2:'S3',3:'S1',4:'S2',5:'S3',6:'S4'}
        for n in range(1,21):
            for k,v in groups.items():
                p = postfixes[k]
                if k in [0,1,2]:  
                    v.append(f'S{wrap_40(2*n)}B:{p}')
                    v.append(f'S{wrap_40(2*n+1)}A:{p}')
                elif k in [3,4,5]:
                    v.append(f'S{wrap_40(2*n-1)}B:{p}')
                    v.append(f'S{wrap_40(2*n)}A:{p}')
                elif k == 6:
                    v.append(f'S{wrap_40((n+1)+20)}A:{p}')
                    v.append(f'S{wrap_40(n+1)}A:{p}')

    elif mode == 'xiaobiao_10fold':
        # 10 fold symmetry
        groups = {i:[] for i in range(14)}
        postfixes = {0:('S{i}B:S1','S{i}A:S1'),
                    1:('S{i}B:S2','S{i}A:S2'),
                    2:('S{i}B:S3','S{i}A:S3'),
                    3:('S{i}A:S4','S{i}A:S4'),
                    4:('S{i}A:S3','S{i}B:S3'),
                    5:('S{i}A:S2','S{i}B:S2'),
                    6:('S{i}A:S1','S{i}B:S1'),
                    7:('S{i}B:S1','S{i}A:S1'),
                    8:('S{i}B:S2','S{i}A:S2'),
                    9:('S{i}B:S3','S{i}A:S3'),
                    10:('S{i}A:S4','S{i}A:S4'),
                    11:('S{i}A:S3','S{i}B:S3'),
                    12:('S{i}A:S2','S{i}B:S2'),
                    13:('S{i}A:S1','S{i}B:S1'),
                    }
        for n in range(1,11):
            for k,v in groups.items():
                p = postfixes[k]
                if k in [0,1,2,3,4,5,6]:
                    idx_left, idx_right = 4*n-2, 4*n+1-2
                    v.append(p[0].format(i=wrap_40(idx_left)))
                    v.append(p[1].format(i=wrap_40(idx_right)))
                elif k in [7,8,9,10,11,12,13]:
                    idx_left, idx_right = 4*n-1-2, 4*n+2-2
                    v.append(p[0].format(i=wrap_40(idx_left)))
                    v.append(p[1].format(i=wrap_40(idx_right)))

    for k in groups:
        groups[k] = sort_by_sector(list(set(groups[k])))

    if keepS40:
        for k in groups:
            subg = groups[k]
            subg = [x for x in subg if '40' not in x]
            subg = [x for x in subg if '39' not in x]
            subg = sort_by_sector(subg)
            groups[k] = subg

    df_list = [make_df(groups[0],delta=0.0)] + [make_df(g, delta=0.1) for g in groups.values()]

    cnt = sum(len(g) for g in groups.values())
    if keepS40:
        assert cnt == 40*7 - 7 - 7, f'{cnt} {40*7}'
    else:
        assert cnt == 40*7, f'{cnt} {40*7}'

    sdds = pysdds.SDDSFile.from_df(df_list)
    sdds.set_mode('ascii')
    pysdds.write(sdds, param_file, overwrite=True)

    ele_part_1 = f"""
        &run_setup
            default_order = 3,
            use_beamline = RING,
            lattice = /home/helios/oagData/sr/lattices/RHB-S7S37/aps.lte,
            p_central_mev = 6e3,
            concat_order = 0,
        &end

        &alter_elements 
            name=*
            type=SEXT
            item=ORDER
            value=3
        &end

        &load_parameters
            allow_missing_elements = 1,
            filename = {param_file},
            allow_missing_parameters = 1,
            change_defined_values = 0,
        &end

        &run_control
            n_steps = {len(df_list)},
        &end
        
        &twiss_output
            output_at_each_step = 1,
            filename = sext.twi,
            matched = 1,
            radiation_integrals = 0,
        &end
        
        &bunched_beam
            n_particles_per_bunch = 1,
        &end
        
        &track &end
        
        &stop &end
    """

    with open('sext.ele','w') as f:
        f.write(ele_part_1)

    p = subprocess.run(['elegant','sext.ele'], check=True, capture_output=True)

    twi = pysdds.read('sext.twi')
    parameters = ['dnux/dp','dnuy/dp']
    data = {}
    for par in parameters:
        data[par] = twi[par].data

    dCdS = np.array([data['dnux/dp'][1:]-data['dnux/dp'][0], data['dnuy/dp'][1:]-data['dnuy/dp'][0]])/0.1
    u,s,vh = np.linalg.svd(dCdS, full_matrices=True)
    null_knobs = vh.T[:,2:]
    return groups, null_knobs

def verify_null_knob(groups, knob):
    def make_df_many(sextupoles_list, delta_list):
        dfs = []
        assert len(sextupoles_list) == len(delta_list), f'{len(sextupoles_list)=} {len(delta_list)=}'
        for sextupoles,delta in zip(sextupoles_list, delta_list):
            df = pd.DataFrame({'ElementName':sextupoles,'ElementParameter':'K2',
                            'ParameterValue':delta,'ParameterMode':'differential'})
            dfs.append(df)
        return pd.concat(dfs,axis=0)

    #knob = null_knobs[:,-1]
    groups_list = list(groups.values())
    df = make_df_many(groups_list, knob*10)
    df_list = [make_df_many(groups_list, np.zeros(len(groups)))] + [df]

    sdds = pysdds.SDDSFile.from_df(df_list)
    sdds.set_mode('ascii')
    pysdds.write(sdds, param_file, overwrite=True)

    p = subprocess.run(['elegant','sext.ele'], check=True, capture_output=True)

    twi = pysdds.read('sext.twi')
    parameters = ['dnux/dp','dnuy/dp']
    data = {}
    for par in parameters:
        data[par] = twi[par].data

    return data

def make_knob_groups(groups, knob_matrix, postfix=''):
    def make_data(sextupoles_list, delta_list):
        data = {}
        assert len(sextupoles_list) == len(delta_list), f'{len(sextupoles_list)=} {len(delta_list)=}'
        for sextupoles,delta in zip(sextupoles_list, delta_list):
            for s in sextupoles:
                data[s+postfix] = delta
        return data
    
    groups_list = list(groups.values())
    knob_groups = {}
    for ki in range(knob_matrix.shape[1]):        
        d = make_data(groups_list, knob_matrix[:,ki])
        knob_groups[f'Family{ki+1}'] = d
    return knob_groups




