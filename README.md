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

2. From "SmartExcavator" folder, install required python packages (it is obviously recommended to create a new virtualenv and install everything there):

```bash
pip install -r requirements.txt
```

3. Open Mevea Modeller and load Excavator Model. Go to Scripting -> Scripts, create a new script object, select "env_backend.py" as the file name in the object view. Go to I/O -> Inputs -> Input_Slew and select the object just created as the script name. 

4. In terminal, navigate to the Mevea Software resources folder (by default it is "C:\Program Files (x86)\Mevea\Resources\Python\Bin") and run 

```bash
python -m pip install numpy requests".
```

## Demo

1. In terminal, navigate to "SmartExcavator" folder and run: 

```bash
python excavator_demo.py -m path_to_the_excavator_mvs
```

2. 

