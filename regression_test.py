import pandas as pd
import os

if os.system('make eden') != 0:
    exit()

if os.system('bin/eden.debug.gcc.cpu.x nml examples/LEMS_NML2_Ex25_MultiComp.xml') != 0:
    exit()

ref = pd.read_csv('LEMS_NML2_Ex25_MultiComp.txt', sep=' +', header=None, engine='python')
out = pd.read_csv('results1.txt', sep=' +', header=None, engine='python')

fail = False
for i in range(4):
    if not (ref[0].values == out[0].values).all():
        print('REPRODUCTION ERROR!!')
        fail = True

if fail:
    import matplotlib.pyplot as plt
    for i in range(1, 4):
        plt.plot(out[0], out[i], color='black')
        plt.plot(ref[0], ref[i], '--', color='green')
    plt.show()

else:
    print('VALIDATION PASS')
