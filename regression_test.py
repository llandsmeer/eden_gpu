import pandas as pd
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def system(cmd):
    print(f'{bcolors.HEADER}Executing {cmd}{bcolors.ENDC}')
    status = os.system(cmd)
    if status != 0:
        print(f'{bcolors.FAIL}FAILED, exit code = {status}{bcolors.ENDC}')
        exit(status)

for Toolchain in ["gcc", "nvcc"]:
    system(f'rm -f results1.txt')
    system(f'make clean TOOLCHAIN={Toolchain}')
    system(f'make eden TOOLCHAIN={Toolchain}')
    system(f'bin/eden.debug.{Toolchain}.cpu.x nml examples/LEMS_NML2_Ex25_MultiComp.xml')

    ref = pd.read_csv('LEMS_NML2_Ex25_MultiComp.txt', sep=' +', header=None, engine='python')
    out = pd.read_csv('results1.txt', sep=' +', header=None, engine='python')

    fail = False
    for i in range(4):
        if not (ref[i].values == out[i].values).all():
            print(f'{bcolors.FAIL}REPRODUCTION ERROR!!{bcolors.ENDC}')
            fail = True

    if fail:
        import matplotlib.pyplot as plt
        for i in range(1, 4):
            plt.plot(out[0], out[i], color='black')
            plt.plot(ref[0], ref[i], '--', color='green')
        plt.show()
        exit(1)

    else:
        print(f'{bcolors.OKGREEN}VALIDATION PASS: {Toolchain}{bcolors.ENDC}')
