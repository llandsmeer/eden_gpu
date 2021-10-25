'''
HOW TO USE THIS SCRIPT (FOR MAX)

Run this bash script in the examples/io folder

for i in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192
do
    python3 write_C51A.net.nml.py $i > C51A.net.nml
    (echo -n "ascent_gpu $i " ; jsrun -n 1 -g 1 -c 1 ../../build/eden "$@" gpu threads_per_block 32 nml Run_C51A.nml | grep Run': ' ) | tee -a log
    (echo -n "ascent_cpu $i " ; jsrun -n 1 -g 1 -c 1 ../../build/eden "$@"                          nml Run_C51A.nml | grep Run': ' ) | tee -a log
done

for i in 16384 32768
do
    python3 write_C51A.net.nml.py $i > C51A.net.nml
    (echo -n "ascent_gpu $i " ; jsrun -n 1 -g 1 -c 1 ../../build/eden "$@" gpu threads_per_block 32 nml Run_C51A.nml | grep Run': ' ) | tee -a log
done

'''




import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(io.StringIO('''
device ncells _config config _setup setup _run run
gpu 1 Config: 0.014 Setup: 3.212 Run: 7.634
cpu 1 Config: 0.014 Setup: 2.349 Run: 0.091
gpu 2 Config: 0.014 Setup: 3.096 Run: 7.668
cpu 2 Config: 0.014 Setup: 2.438 Run: 0.178
gpu 4 Config: 0.013 Setup: 3.118 Run: 8.165
cpu 4 Config: 0.014 Setup: 2.572 Run: 0.374
gpu 8 Config: 0.013 Setup: 3.072 Run: 9.061
cpu 8 Config: 0.013 Setup: 2.401 Run: 0.694
gpu 16 Config: 0.013 Setup: 3.074 Run: 9.648
cpu 16 Config: 0.013 Setup: 2.400 Run: 1.405
gpu 32 Config: 0.013 Setup: 3.247 Run: 11.278
cpu 32 Config: 0.013 Setup: 2.572 Run: 2.963
gpu 64 Config: 0.014 Setup: 3.143 Run: 12.102
cpu 64 Config: 0.013 Setup: 2.652 Run: 5.953
gpu 128 Config: 0.014 Setup: 3.231 Run: 13.209
cpu 128 Config: 0.014 Setup: 2.718 Run: 12.042
gpu 265 Config: 0.014 Setup: 3.680 Run: 16.251
cpu 265 Config: 0.014 Setup: 3.070 Run: 25.775
gpu 512 Config: 0.014 Setup: 3.566 Run: 23.590
cpu 512 Config: 0.014 Setup: 2.970 Run: 46.880
gpu 1024 Config: 0.015 Setup: 4.835 Run: 51.401
cpu 1024 Config: 0.017 Setup: 6.799 Run: 108.438
gpu 2048 Config: 0.016 Setup: 5.251 Run: 120.707
cpu 2048 Config: 0.017 Setup: 4.702 Run: 189.194
gpu 4096 Config: 0.019 Setup: 7.465 Run: 290.554
cpu 4096 Config: 0.024 Setup: 8.455 Run: 393.237
ascent_gpu 1 Config: 0.029 Setup: 7.439 Run: 0.929
ascent_cpu 1 Config: 0.027 Setup: 5.420 Run: 0.024
ascent_gpu 2 Config: 0.029 Setup: 6.334 Run: 0.948
ascent_cpu 2 Config: 0.026 Setup: 4.860 Run: 0.047
ascent_gpu 4 Config: 0.028 Setup: 6.333 Run: 0.966
ascent_cpu 4 Config: 0.027 Setup: 4.863 Run: 0.093
ascent_gpu 8 Config: 0.028 Setup: 6.334 Run: 1.007
ascent_cpu 8 Config: 0.026 Setup: 4.866 Run: 0.185
ascent_gpu 16 Config: 0.028 Setup: 6.361 Run: 1.125
ascent_cpu 16 Config: 0.027 Setup: 4.884 Run: 0.368
ascent_gpu 32 Config: 0.029 Setup: 6.393 Run: 1.192
ascent_cpu 32 Config: 0.027 Setup: 4.915 Run: 0.734
ascent_gpu 64 Config: 0.030 Setup: 6.486 Run: 1.246
ascent_cpu 64 Config: 0.027 Setup: 4.980 Run: 1.467
ascent_gpu 128 Config: 0.031 Setup: 6.621 Run: 1.465
ascent_cpu 128 Config: 0.027 Setup: 5.099 Run: 2.938
ascent_gpu 256 Config: 0.032 Setup: 6.858 Run: 1.519
ascent_cpu 256 Config: 0.027 Setup: 5.328 Run: 5.899
ascent_gpu 512 Config: 0.033 Setup: 7.385 Run: 1.549
ascent_cpu 512 Config: 0.028 Setup: 5.802 Run: 11.801
ascent_gpu 1024 Config: 0.034 Setup: 8.483 Run: 1.765
ascent_cpu 1024 Config: 0.029 Setup: 6.727 Run: 23.737
ascent_gpu 2048 Config: 0.041 Setup: 10.580 Run: 2.078
ascent_cpu 2048 Config: 0.032 Setup: 8.610 Run: 47.829
ascent_gpu 4096 Config: 0.047 Setup: 14.889 Run: 2.870
ascent_cpu 4096 Config: 0.037 Setup: 12.359 Run: 95.346
ascent_gpu 8192 Config: 0.058 Setup: 25.841 Run: 8.824
ascent_cpu 8192 Config: 0.048 Setup: 20.547 Run: 190.637
ascent_gpu 16384 Config: 0.081 Setup: 40.694 Run: 16.719
ascent_gpu 32768 Config: 0.131 Setup: 75.462 Run: 41.855
'''.lstrip()), sep=' ')

gpu = df[df.device == 'gpu']
cpu = df[df.device == 'cpu']

ascent_gpu = df[df.device == 'ascent_gpu']
ascent_cpu = df[df.device == 'ascent_cpu']

def interp(df, at=15):
    X = np.array([df.ncells.max(), 2**15])
    x = np.log2(df.ncells.values)
    y = np.log2(df.run.values)
    x = x[-4:]
    y = y[-4:]
    slope, bias = np.polyfit(x, y, 1)
    Y = 2**(slope*np.log2(X) + bias)
    plt.plot(X, Y, 'o--', lw=1, color='grey')
    return 2**(slope*at + bias)

power15 = interp(ascent_cpu)
gtx15   = interp(gpu)
ascent15 = ascent_gpu.run.values[ascent_gpu.ncells.values == 2**15].item()
print('GTX1050/V100 speedup', gtx15 / ascent15)
print('ASCENT/V100 Speedup', power15 / ascent15)

plt.annotate(f'{power15 / ascent15:.1f}x', (2**15, 10**((np.log10(power15) + np.log10(ascent15))/2)))

plt.plot([2**15, 2**15], [ascent15, power15], '--', color='#dddddd', lw=3, zorder=-10)

plt.plot(
        gpu.ncells.values,
        gpu.run.values,
        'o--',
        label='GTX 1050 mobile',
        color='#00aaff'
        )

#plt.plot( cpu.ncells.values, cpu.run.values, 'o--', label='i7-8750H CPU @ 2.20GHz', color='#007fff')

plt.plot(
        ascent_gpu.ncells.values,
        ascent_gpu.run.values,
        'o--',
        label='V100',
        color='#007aaa'
        )

plt.plot(
        ascent_cpu.ncells.values,
        ascent_cpu.run.values,
        'o--',
        label='POWER9',
        color='black'
        )


plt.legend()
plt.xscale('log')
plt.yscale('log')

plt.title('CPU vs GPU comparison - 100ms IO cell simulation')
plt.xlabel('No. of C55A IO cells')
plt.ylabel('Simulation time (s)')
plt.xticks(ascent_gpu.ncells.values, [str(x) for x in ascent_gpu.ncells.values], rotation='vertical')
plt.xlim(1, 2**17)
plt.tight_layout()

plt.show()
