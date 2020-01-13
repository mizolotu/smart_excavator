# Simulation autorun script
# Sep 2018, VVH

import os
import sys
import subprocess
import winreg
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from time import sleep
from csv import reader as csvreader
from matplotlib.backends.backend_pdf import PdfPages

MTMDATA = 0
MODELXML = 1


def autoRunSimulation(modelfile, wsfile, paramfile, inputfile, mode=MTMDATA):
    # Omit file extension from model path
    if modelfile[-4:] in ['.xml', '.mvs']:
        modelfile = modelfile[:-4]

    # Get result path from model path
    resultpath = modelfile[:modelfile.rfind('\\')] + '\\Results\\'

    # Get solver path from Windows registry
    regkey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'Software\WOW6432Node\Mevea\Mevea Simulation Software')
    (solverpath, _) = winreg.QueryValueEx(regkey, 'InstallPath')
    solverpath += r'\Bin\MeveaSolver.exe'
    winreg.CloseKey(regkey)

    # Delete old data in result folder
    if os.path.exists(resultpath):
        resultfiles = os.listdir(resultpath)
        for r in resultfiles:
            if r[0:9] in ['Plotdata_', 'simulatio']:
                os.remove(resultpath + r)

    # Start reading input data
    print('Reading parameters...')

    # Read csv
    param_data = []
    with open(paramfile, 'r') as dataFile:

        # Check file delimiter (comma or semicolon)
        sample = dataFile.readline()
        dataFile.seek(0)

        if sample.find(';') != -1:
            delimiter = ';'
        else:
            delimiter = ','

        reader = csvreader(dataFile, delimiter=delimiter)

        # Headers from first row
        param_hdr = next(reader)

        while True:
            try:
                row = next(reader)
                param_data.append(row)
            except(StopIteration):
                break

        print(param_data)

    # param_data = [list(x) for x in zip(*param_data)] # Transpose parameter list
    casenum = len(param_data)  # Get case number

    # Start simulation
    print('Starting simulation...')

    # If mode == MODELXML, changes are made to model main xml file. The original is backed up first
    # This method is more versatile, but less "safe"
    # If script is interrupted, original model file may need to be manually restored from backup
    # CSV HEADER ROW FORMAT: (element path):(attribute name)
    if mode == MODELXML:

        # Solver argument list
        process_args = [solverpath, r'/loadmws', wsfile, r'/saveplots', r'/silent']
        if inputfile: process_args.extend([r'/playio', inputfile])

        # Backup original model file
        os.rename(modelfile + '.xml', modelfile + '_backup.xml')

        try:
            # Simulation loop
            for i in range(casenum):

                # Make a copy of the original model file with new parameters
                tree = ET.parse(modelfile + '_backup.xml')
                root = tree.getroot()

                for j in range(len(param_hdr)):

                    # Separate element path and key name
                    comp_path = param_hdr[j].split(r':', 1)

                    elem = root.findall(r'./' + comp_path[0])

                    if len(elem) < 1:
                        raise Exception('No component found with the path ' + comp_path[0])
                    elif len(elem) > 1:
                        raise Exception('Several components found with the path ' + comp_path[0])
                    else:
                        elem[0].set(comp_path[1], param_data[i][j])

                tree.write(modelfile + '.xml')

                # Launch solver
                subprocess.run(process_args)

                # Suspend for 2 seconds to make sure solver closes properly
                sleep(2)
                print('  Simulation ' + str(i + 1) + '/' + str(casenum) + ' finished.')

                # Remove xml file
                os.remove(modelfile + '.xml')

                # Find created result file and rename it
                resultfiles = os.listdir(resultpath)
                for r in resultfiles:
                    if r[0:9] == 'Plotdata_':
                        os.rename(resultpath + r, resultpath + 'simulation' + str(i + 1) + '_results.txt')
                        break
                else:
                    # If result file not found, raise an exception
                    raise FileNotFoundError('Result file not found!')

            else:
                # Restore original model file from backup
                os.rename(modelfile + '_backup.xml', modelfile + '.xml')

        except:
            # Remove xml file, if it exists
            if os.path.exists(modelfile + '.xml'):
                os.remove(modelfile + '.xml')

            # Restore original model file from backup
            os.rename(modelfile + '_backup.xml', modelfile + '.xml')

            raise

    # If mode == MTMDATA, no changes are made to the model file. Instead, mtmdata.xml is created and loaded at runtime
    # This method is safe, but limited by the number of implemented commands
    # CSV HEADER ROW FORMAT: (component name):(command name)
    elif mode == MTMDATA:

        mtmpath = os.getcwd() + r'\mtmdata.xml'

        # Solver argument list
        process_args = [solverpath, r'/loadmws', wsfile, r'/mtmxml', mtmpath, r'/saveplots', r'/silent']
        if inputfile: process_args.extend([r'/playio', inputfile])

        components = []
        commands = []

        # Separate element name and command name
        for j in range(len(param_hdr)):
            names = param_hdr[j].split(r':', 1)

            components.append(names[0])
            commands.append(names[1])

        try:
            # Simulation loop
            for i in range(casenum):

                root = ET.Element('MtmData')

                for j in range(len(components)):
                    ET.SubElement(root, 'KeyValue', Key=commands[j], Value=param_data[i][j], Address=components[j])

                tree = ET.ElementTree(root)
                tree.write(mtmpath, 'utf-8')

                # Launch solver
                subprocess.run(process_args)

                # Suspend for 2 seconds to make sure solver closes properly
                sleep(2)
                print('  Simulation ' + str(i + 1) + '/' + str(casenum) + ' finished.')

                # Remove mtmdata file
                os.remove(mtmpath)

                # Find created result file and rename it
                resultfiles = os.listdir(resultpath)
                for r in resultfiles:
                    if r[0:9] == 'Plotdata_':
                        os.rename(resultpath + r, resultpath + 'simulation' + str(i + 1) + '_results.txt')
                        break
                else:
                    # If result file not found, raise an exception
                    raise FileNotFoundError('Result file not found!')

        except:
            # Remove mtmdata file, if it was already created
            if os.path.exists(mtmpath):
                os.remove(mtmpath)

            raise

    # Start postprocessing
    print('Processing results...')

    # Read and plot output data (one plot per result set, all cases in the same plot)
    for i in range(casenum):
        plotdata = []

        with open(resultpath + 'simulation' + str(i + 1) + '_results.txt', 'r') as pdata:

            # Discard first line
            pdata.readline()

            # Get headings from second line
            headings = pdata.readline().split('\t')
            if headings[-1] in ['', '\n']: headings = headings[:-1]  # Get rid of empty element at the end

            while True:
                line = pdata.readline()
                if not line:
                    break
                else:
                    plotdata.append(line.split('\t'))

        plotdata = [list(x) for x in zip(*plotdata)]  # Transpose list
        plotdata = [[float(x) for x in y] for y in plotdata]  # Convert to floating point

        for j in range(1, len(headings)):
            x = plotdata[0]
            y = plotdata[j]
            plt.figure(j)
            plt.plot(x, y)

    # Define plot legend and heading
    for i in range(1, len(headings)):
        plt.figure(i)
        plt.title(headings[i])
        plt.grid(True)
        plt.legend(['simulation ' + str(n + 1) for n in range(casenum)])

    # Write plots into pdf
    with PdfPages(os.getcwd() + r'\simulation_results.pdf') as pdf:
        for i in range(1, len(headings)):
            plt.figure(i)
            plt.gcf().set_size_inches(11.69, 8.27)
            pdf.savefig(orientation='landscape', papertype='a4')
            plt.gcf().clear()

        # Write parameter table on last page
        plt.figure(0)
        plt.gcf().set_size_inches(11.69, 8.27)
        plt.axis('tight')
        plt.axis('off')
        plt.table(cellText=param_data, colLabels=param_hdr,
                  rowLabels=['simulation ' + str(n + 1) for n in range(casenum)], loc='center', cellLoc='center')
        pdf.savefig(orientation='landscape', papertype='a4')
        plt.gcf().clear()

    print('Done!')

if __name__ == '__main__':
    model_fname = 'C:\\Users\\iotli\\Desktop\\Excavator\\Model\\Excavator.mws'
    param_fname = 'C:\\Users\\iotli\\Desktop\\Excavator\\Model\\RunAutomatedSimulationScripts\\ParametersMTM.csv'
    ws_fname = 'C:\\Users\\iotli\\Desktop\\Mevea-Data\\test_workspace.mws'
    autoRunSimulation(model_fname, ws_fname, param_fname, [], 0)