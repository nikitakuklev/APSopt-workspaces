import numpy as np
import sext_tools

VALID_KNOBS = (list(sext_tools.SUBGROUP_MAP_INV.keys()) + ['SVD0'] + sext_tools.HARMONIC_TOP_GROUPS  +
[x+'N' for x in sext_tools.HARMONIC_TOP_GROUPS])

def make_dummy_eval(gvocs):
   def master_eval_function_dummy(inputs_dict):
      """ Test function """
      assert len(inputs_dict) == len(gvocs.variables)
      assert all([k in VALID_KNOBS for k in inputs_dict.keys()])
      vals = np.array(list(inputs_dict.values()))
      lt = 5 - 10*np.sum((vals-0.4)**2)/(len(vals)) + np.random.randn()*0.2
      eff = 2000*np.sum(np.abs(vals))/(len(vals)) + np.random.randn()*0.05
      eff = min(eff, 100)
      extras = {k: np.random.randn() for k in ['_nux', '_nuy', '_eps_x_bxbOn', '_eps_y_bxbOn']}
      return {'LT': lt, 'EFF': eff, **extras}
   
   return master_eval_function_dummy


def make_dummy_eval_lifetime(gvocs):
   def master_eval_function_dummy(inputs_dict):
      """ Test function """
      assert len(inputs_dict) == len(gvocs.variables)
      vals = np.array(list(inputs_dict.values()))
      lt = 10 - 5*np.sum((vals-0.4)**2) + np.random.randn()*0.2
      extras = {k: np.random.randn() for k in ['_nux', '_nuy', '_eps_x_bxbOn', '_eps_y_bxbOn']}
      return {'LT': lt, **extras}
   
   return master_eval_function_dummy



def make_real_eval(gvocs):
   # Create a function that maps variables to objectives+observables

   def master_eval_function(inputs_dict):
      raise NotImplementedError
   
   return master_eval_function



