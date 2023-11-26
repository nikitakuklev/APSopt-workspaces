import numpy as np

def make_dummy_eval(gvocs):
   def master_eval_function_dummy(inputs_dict):
      """ Test function """
      assert len(inputs_dict) == len(gvocs.variables)
      vals = np.array(list(inputs_dict.values()))
      lt = 10 - 5*np.sum((vals-0.4)**2) + np.random.randn()*0.2
      eff = 100*np.sum((vals)) + np.random.randn()*0.05
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



