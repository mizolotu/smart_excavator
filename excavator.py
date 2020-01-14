import winreg, subprocess

def autorun(mws):
    regkey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'Software\WOW6432Node\Mevea\Mevea Simulation Software')
    (solverpath, _) = winreg.QueryValueEx(regkey, 'InstallPath')
    solverpath += r'\Bin\MeveaSolver.exe'
    winreg.CloseKey(regkey)
    process_args = [solverpath, r'/loadmws', mws, r'/saveplots', r'/silent']
    print(process_args)
    while True:
        subprocess.run(process_args)