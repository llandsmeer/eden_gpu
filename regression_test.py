import pandas as pd
import os

for Toolchain in ["gcc","nvcc"]:

    if os.system('make clean') != 0:
        exit()

    if os.system('make eden TOOLCHAIN=' + Toolchain) != 0:
        exit()

    if os.system('bin/eden.debug.' + Toolchain + '.cpu.x nml examples/LEMS_NML2_Ex25_MultiComp.xml') != 0:
        exit()

    ref = pd.read_csv('LEMS_NML2_Ex25_MultiComp.txt', sep=' +', header=None, engine='python')
    out = pd.read_csv('results1.txt', sep=' +', header=None, engine='python')

    fail = False
    for i in range(4):
        if not (ref[i].values == out[i].values).all():
            print('REPRODUCTION ERROR!!')
            fail = True

    if fail:
        import matplotlib.pyplot as plt
        for i in range(1, 4):
            plt.plot(out[0], out[i], color='black')
            plt.plot(ref[0], ref[i], '--', color='green')
        plt.show()

    else:
        print('VALIDATION PASS: ' + Toolchain)