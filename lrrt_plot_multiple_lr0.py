from sys import argv
import matplotlib.pyplot as plt
from pandas import read_csv
from os.path import splitext

plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.monospace'] = ['Source Code Pro', 'Cousine', 'Courier New', 'DejaVu Sans Mono']
plt.style.use('tableau-colorblind10')

input_filename = argv[1]
output_pdf = splitext(input_filename)[0]+'.pdf'
print(input_filename)
data = read_csv(input_filename, sep='\s+', header=None)
x1 = data[2]
y11 = data[3]
y12 = data[4]
x2 = data[7]
y21 = data[8]
y22 = data[9]
x3 = data[12]
y31 = data[13]
y32 = data[14]


def set_common_par():
    plt.grid(True, which='major', axis='x', lw=1.1)
    plt.grid(True, which='minor', axis='x', lw=0.1)
    return()


plt.figure(figsize=[4.8*1920/1080, 4.8], dpi=225)
plt.suptitle('Learning rate range test')

plt.subplot(211)
plt.plot(x1, y11, '-', fillstyle='none', ms=4.5, mew=0.6, lw=1.5, label='lr0=1E-4')
plt.plot(x2, y21, '-', fillstyle='none', ms=4.5, mew=0.6, lw=1.5, label='lr0=1E-5')
plt.plot(x3, y31, '-', fillstyle='none', ms=4.5, mew=0.6, lw=1.5, label='lr0=1E-6')
plt.ylabel('Loss')
plt.xscale('log')
plt.yscale('linear')
plt.xlim(1E-6, 1)
plt.tick_params(axis='x', which='both', top=True, bottom=False, labelbottom=False, labeltop=True)
set_common_par()
plt.legend(loc='lower left')

plt.subplot(212, sharex=plt.subplot(211))
plt.ylabel('Accuracy')
plt.xlabel('Learning rate')
plt.plot(x1, y12, '-', fillstyle='none', ms=4.5, mew=0.6, lw=1.5, label='lr0=1E-4')
plt.plot(x2, y22, '-', fillstyle='none', ms=4.5, mew=0.6, lw=1.5, label='lr0=1E-5')
plt.plot(x3, y32, '-', fillstyle='none', ms=4.5, mew=0.6, lw=1.5, label='lr0=1E-6')
plt.tick_params(axis='x', which='both', top=False, bottom=True)
set_common_par()
plt.legend(loc='upper left')

plt.tight_layout(h_pad=0)
plt.savefig(output_pdf, bbox_inches='tight')
plt.show()
plt.close()