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


def system(cmd, gpu=False, submit=True):
    if os.path.exists('/opt/ibm/spectrum_mpi/jsm_pmix/bin/jsrun') and submit:
        if gpu:
            job = 'jsrun -g2 -c42 -n1'
        else:
            job = 'jsrun -c42 -n1'
        cmd = f'{job} {cmd}'
    print(f'{bcolors.HEADER}Executing {cmd}{bcolors.ENDC}')
    status = os.system(cmd)
    if status != 0:
        print(f'{bcolors.FAIL}FAILED, exit code = {status}{bcolors.ENDC}')
        exit(status)

final = []

def verify(toolchain, nmlfile, output, target):
    system(f'bin/eden.debug.{toolchain}.cpu.x nml {nmlfile}', gpu=toolchain=='nvcc', submit=True)

    ref = pd.read_csv(target, sep=' +', header=None, engine='python', na_values=['+nan', '-nan'])
    out = pd.read_csv(output, sep=' +', header=None, engine='python', na_values=['+nan', '-nan'])

    fail = False
    max_error = 0
    for i in range(4):
        target = ref[i].values
        pred = out[i].values
        error = (abs(target - pred) / target.ptp()).max()
        max_error = max(error, max_error)
        if not error < 0.02:
            msg = f'{bcolors.FAIL}{toolchain}|{nmlfile}: REPRODUCTION ERROR={error*100:.2f}%!!{bcolors.ENDC}'
            final.append(msg)
            print(msg)
            fail = True

    if fail:
        import matplotlib.pyplot as plt
        for i in range(1, 4):
            plt.plot(out[0], out[i], color='black')
            plt.plot(ref[0], ref[i], '--', color='green')
        plt.show()
        exit(1)

    else:
        msg = f'{bcolors.OKGREEN}{toolchain}|{nmlfile}: VALIDATION PASS (max error={max_error*100:.2f}%){bcolors.ENDC}'
        final.append(msg)
        print(msg)

toolchains = ["gcc", "nvcc"]

for toolchain in toolchains:
    system(f'rm -f results1.txt')
    system(f'make clean TOOLCHAIN={toolchain}')
    system(f'make eden TOOLCHAIN={toolchain}')
    verify(toolchain, 'examples/LEMS_NML2_Ex25_MultiComp.xml', 'results1.txt', 'LEMS_NML2_Ex25_MultiComp.txt')
    verify(toolchain, 'examples/LEMS_NML2_Ex25_MultiCelltypes_TEST.xml', 'results2.txt', 'LEMS_NML2_Ex25_MultiCelltypes_TEST.txt')

print('(-- logs repeated here --)')
for msg in final:
    print(msg)
