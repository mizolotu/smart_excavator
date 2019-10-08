import pickle
import numpy as np
from flask import Flask, request

# This function is called once, when script is first initialized
def initScript():
    
 # Load data from a file

 data_dir = 'C:\\Users\\mikha\\PycharmProjects\\SmartExcavator\\data'
 fname = data_dir + '\Dataset9_cycles.pkl'

 #model_dir = 'C:\Research\Mantsinen\SmartExcavator\models\'
 data_names = {
  'timestamps': ['Time'], 
  'controls': ['Input_Slew RealValue','Input_BoomLift RealValue', 'Input_DipperArm RealValue', 'Input_Bucket RealValue'], 
  'measurements': ['ForceR_Slew r', 'Cylinder_BoomLift_L x', 'Cylinder_DipperArm x', 'Cylinder_Bucket x']
 }
 #smart_excavator = SmartExcavator(data_dir,data_names,model_dir)
 #dataset_index = 5
 #fname = smart_excavator.data_files[dataset_index]

 
 print('Loading data from ' + fname)
 with open(fname, 'rb') as handle:
    data = pickle.load(handle)
 cycle_length = np.mean(np.array([len(x) for x in data] ))
 #print(cycle_length)
 measurements = np.vstack(data)

 #data = smart_excavator.get_data_from_file(fname)
 #timestamps = np.vstack(data['timestamps'])
 #measurements = np.vstack(data['measurements'])
 #time = timestamps.reshape(len(timestamps))

 # Loading parameters
 #params = smart_excavator.load_data('data_parameters.pkl')
 #timestep = params[0]
 #mu = params[1]
 #sigma = params[2]
 #x_min = params[3]
 #x_max = params[4]
 #print(timestep,mu,sigma,x_min,x_max)
 
 # Initiate vectors
 components = [p.split(' ')[0] for p in data_names['measurements']]
 parameters = [p.split(' ')[1] for p in data_names['measurements']]
 component_controls = [p.split(' ')[0] for p in data_names['controls']]
 #print(components)
 #print(parameters)
 #print(component_controls)

 # Saving parameters
 GObject.data['components'] = [GSolver.getParameter(components[index],parameters[index]) for index in range(len(components))]
 GObject.data['controls'] = component_controls
 GObject.data['error_prev'] = np.zeros(len(components))
 GObject.data['integ_prev'] = np.zeros(len(components))
 GObject.data['measurements'] = measurements
 GObject.data['cycle_length'] = cycle_length
 GObject.data['cycle_count'] = 0
 #GObject.data['time'] = time
 GObject.data['index'] = 0

 print('Starting cycle 0')
 
 return 0
		
# This function is called repeatedly during simulation
def callScript( deltaTime, simulationTime ):

 # Script parameters
 error_thr = [3,3]
 index_step = 1
 gain_P = [25, 10, 10, 10]
 gain_I = [0, 0, 0, 0]
 gain_D = [1, 0.1, 0.1, 0.1]
 scale_coeff = -1.

 # Loading data from the previous iteration
 #time_vals = GObject.data['time']
 errors_prev = GObject.data['error_prev']
 integs_prev = GObject.data['integ_prev']
 real_values = [x.value() for x in GObject.data['components']]
 #time = GObject.data['time']
 measurement_vals = GObject.data['measurements']
 component_controls = GObject.data['controls']
 index = GObject.data['index']
 cycle_length = GObject.data['cycle_length']
 cycle_count = GObject.data['cycle_count']

 #desired_values = np.array([np.interp(simulationTime - init_time + pid_delay, time_vals[np.arange(0,len(time_vals),100)], measurement_vals[np.arange(0,len(time_vals),100),index]) for index in range(len(real_values))])
 desired_values = measurement_vals[index,:]
 control_inputs = np.zeros(len(real_values))
 
 for i in range(len(real_values)):
  real_value = real_values[i]
  desired_value = desired_values[i]
  component_control = component_controls[i]
  error_prev = errors_prev[i]
  integ_prev = integs_prev[i]
  #print(real_value,desired_value)
  error = desired_value - real_value 
  diff = (error - error_prev)/deltaTime
  integ = integ_prev + (error + error_prev)/2 * deltaTime
  output = scale_coeff * (gain_P[i]*error + gain_I[i]*integ + gain_D[i]*diff)
  if output > 600:
   output = 600
  elif output < -600:
   output = -600        
  #print(gain_P*error, gain_I*integ, gain_D*diff, output)
  GDict[component_control].setInputValue(output)
  control_inputs[i] = output
  errors_prev[i] = error
  integs_prev[i] = integ

 #GObject.setInputValue(scale_coeff*output)
 GObject.data['error_prev'] = errors_prev
 GObject.data['integ_prev'] = integs_prev

 print(errors_prev)
 print(control_inputs)

 if np.abs(errors_prev[0]) < error_thr[0] and np.linalg.norm(errors_prev[1:]) < error_thr[1]:
  if index < len(measurement_vals) - 1:
   index += index_step
  else:
   print('Simulation is over!')
  #print('Index changed: ' + str(index))
  if index % cycle_length == 0:
   cycle_count += 1
   print('Starting cycle ' + str(cycle_count))
 else:
  #print('Errors: ' + str(errors_prev))
  #print('Inputs: ' + str(control_inputs))
  pass
  
 GObject.data['index'] = index
 GObject.data['cycle_count'] = cycle_count
 
 return 0
