import numpy as np

# Data names

data_names = {
    'timestamps': ['Time'],
    'controls': [
        'Input_Slew RealValue',
        'Input_BoomLift RealValue',
        'Input_DipperArm RealValue',
        'Input_Bucket RealValue'
    ],
    'measurements': [
        'ForceR_Slew r',
        'Cylinder_BoomLift_L x',
        'Cylinder_DipperArm x',
        'Cylinder_Bucket x'
    ]
}

# Component names

components = [p.split(' ')[0] for p in data_names['measurements']]
parameters = [p.split(' ')[1] for p in data_names['measurements']]
component_controls = [p.split(' ')[0] for p in data_names['controls']]


def initScript():
    GObject.data['component_objects'] = [GSolver.getParameter(components[index], parameters[index]) for index in range(len(components))]
    GObject.data['last_values'] = None
    GObject.data['max_delta'] = np.zeros(4)


def callScript(deltaTime, simulationTime):
    real_values = np.array([x.value() for x in GObject.data['component_objects']])
    if GObject.data['last_values'] is not None:
        delta = np.abs(real_values - GObject.data['last_values'])
        GObject.data['max_delta'] = np.max(np.vstack([GObject.data['max_delta'], delta]), axis=0)
    GObject.data['last_values'] = np.array(real_values)
    print(GObject.data['max_delta'])