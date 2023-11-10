import numpy as np

def make_dummy_eval(gvocs):
   def master_eval_function_dummy(inputs_dict):
      """ Test function """
      assert len(inputs_dict) == len(gvocs.variables)
      vals = np.array(list(inputs_dict.values()))
      lt = 10 - 5*np.sum((vals-0.4)**2) + np.random.randn()*0.2
      eff = np.sum((vals)) + np.random.randn()*0.05
      return {'LT': lt, 'EFF': eff}
   
   return master_eval_function_dummy


def make_real_eval(gvocs):
   # Create a function that maps variables to objectives+observables

   def master_eval_function(inputs_dict):
      raise NotImplementedError
   
   return master_eval_function



