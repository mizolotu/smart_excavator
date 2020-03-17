# SmartExcavator

## Requirements

1. Python 3.6.8 or higher
2. Mevea Simulation Software
3. Excavator Model.

## Installation

1. Clone the repo: 
```bash
git clone https://github.com/mizolotu/SmartExcavator
```

2. From "SmartExcavator" directory, install required python packages (it is obviously recommended to create a new virtualenv and install everything there):
```bash
pip install -r requirements.txt
```

3. Open Mevea Modeller and load Excavator Model. Go to Scripting -> Scripts, create a new script object, select "env_backend.py" from "SmartExcavator" directory as the file name in the object view. Go to I/O -> Inputs -> Input_Slew and select the object just created as the script name. 

4. In terminal, navigate to the Mevea Software resources directory (default: C:\Program Files (x86)\Mevea\Resources\Python\Bin) and install numpy and requests:
```bash
python -m pip install numpy requests".
```

## Demo

In terminal, navigate to "SmartExcavator" directory and run: 
```bash
python excavator_demo.py -m path_to_the_excavator_mvs
```

More options:
```bash
python excavator_demo.py --help
```

## Continue training

Residual policy has been learned for 4-5 days, while PPO - only for a couple og hours. To continue training, substitute policy_name with either "residual" or "ppo" (default: residual) and specify number of environments (default: 2), for example:
```bash
python excavator_demo.py -m path_to_the_excavator_mvs -t train -p residual -n 2
```


