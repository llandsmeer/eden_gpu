import pandas as pd
import os

PLOT_ON_FAILURE = False
TEST_MPI = True

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

def verify(nmlfile, output, target, gpu=True, nmpi=None):
    if nmpi is not None:
        pre = f'mpirun -np {nmpi}'
    else:
        pre = ''
    system(f'{pre} build/eden {"mpi" if nmpi is not None else ""} {"gpu" if gpu else ""} nml {nmlfile}', gpu=gpu, submit=True)
    toolchain = "GPU" if gpu else "CPU"
    testcase = toolchain
    if nmpi is not None:
        testcase = testcase+f'/mpi={nmpi}'

    ref = pd.read_csv(target, sep=' +', header=None, engine='python', na_values=['+nan', '-nan'])
    out = pd.read_csv(output, sep=' +', header=None, engine='python', na_values=['+nan', '-nan'])

    fail = False
    max_error = 0
    for i in range(4):
        target = ref[i].values
        pred = out[i].values
        try:
            error = (abs(target - pred) / target.ptp()).max()
        except:
            fail = True
            msg = f'{bcolors.FAIL}{testcase}|{nmlfile}: REPRODUCTION ERROR=COULD NOT READ DATA!!{bcolors.ENDC}'
            final.append(msg)
            print(msg)
            fail = True
            break
        max_error = max(error, max_error)
        if not error < 0.02:
            msg = f'{bcolors.FAIL}{testcase}|{nmlfile}: REPRODUCTION ERROR={error*100:.2f}%!!{bcolors.ENDC}'
            final.append(msg)
            print(msg)
            fail = True
            break

    if fail and PLOT_ON_FAILURE:
        import matplotlib.pyplot as plt
        for i in range(1, 4):
            plt.plot(out[0], out[i], color='black')
            plt.plot(ref[0], ref[i], '--', color='green')
        plt.show()
        exit(1)

    elif not fail:
        msg = f'{bcolors.OKGREEN}{testcase}|{nmlfile}: VALIDATION PASS (max error={max_error*100:.2f}%){bcolors.ENDC}'
        final.append(msg)
        print(msg)

if TEST_MPI:
    system(f'sh -c "rm -f results1.txt; mkdir -p build; cd build; cmake -DUSE_MPI=ON  ..; make -j 4"')
else:
    system(f'sh -c "rm -f results1.txt; mkdir -p build; cd build; cmake -DUSE_MPI=OFF ..; make -j 4"')

# verify('examples/LEMS_NML2_Ex25_MultiComp.xml',           'results1.txt', 'LEMS_NML2_Ex25_MultiComp.txt', gpu=False)
# verify('examples/LEMS_NML2_Ex25_MultiCelltypes_TEST.xml', 'results2.txt', 'LEMS_NML2_Ex25_MultiCelltypes_TEST.txt', gpu=False)
verify('examples/LEMS_NML2_Ex25_MultiComp.xml',           'results1.txt', 'LEMS_NML2_Ex25_MultiComp.txt', gpu=True)
verify('examples/LEMS_NML2_Ex25_MultiCelltypes_TEST.xml', 'results2.txt', 'LEMS_NML2_Ex25_MultiCelltypes_TEST.txt', gpu=True)

# if TEST_MPI:
    # verify('examples/LEMS_NML2_Ex25_MultiComp.xml', 'results1.txt', 'LEMS_NML2_Ex25_MultiComp.txt', gpu=False, nmpi=1)
    # verify('examples/LEMS_NML2_Ex25_MultiComp.xml', 'results1.txt', 'LEMS_NML2_Ex25_MultiComp.txt', gpu=False, nmpi=2)
    # verify('examples/LEMS_NML2_Ex25_MultiComp.xml', 'results1.txt', 'LEMS_NML2_Ex25_MultiComp.txt', gpu=False, nmpi=4)
    # verify('examples/LEMS_NML2_Ex25_MultiComp.xml', 'results1.txt', 'LEMS_NML2_Ex25_MultiComp.txt', gpu=True, nmpi=1)
    # verify('examples/LEMS_NML2_Ex25_MultiComp.xml', 'results1.txt', 'LEMS_NML2_Ex25_MultiComp.txt', gpu=True, nmpi=2)
    # verify('examples/LEMS_NML2_Ex25_MultiComp.xml', 'results1.txt', 'LEMS_NML2_Ex25_MultiComp.txt', gpu=True, nmpi=4)


print('(-- logs repeated here --)')
for msg in final:
    print(msg)
