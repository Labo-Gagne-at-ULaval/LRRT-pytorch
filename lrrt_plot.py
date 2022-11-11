from sys import argv
import matplotlib.pyplot as plt
# import matplotlib.patches as mpl_patches
from pandas import read_csv
from os.path import splitext

plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.monospace'] = ['Source Code Pro', 'Cousine', 'Courier New', 'DejaVu Sans Mono']
plt.style.use('tableau-colorblind10')

input_filename = argv[1]
output_pdf = splitext(input_filename)[0]+'.pdf'
print(input_filename)
data = read_csv(input_filename, sep='\s+', header=None)
x = data[1]
y1 = data[2]    # Train Running Loss
y2 = data[3]    # Train Loss
y3 = data[4]    # Val Loss
y4 = data[5]    # Train Running Accuracy
y5 = data[6]    # Train Accuracy
y6 = data[7]    # Val Accuracy


def set_common_par():
    plt.grid(True, which='major', axis='x', lw=1.1)
    plt.grid(True, which='minor', axis='x', lw=0.1)
    plt.grid(True, which='major', axis='y', lw=1.1)
    return()


plt.figure(figsize=[4.8*1920/1080, 4.8], dpi=225)
plt.suptitle('Learning rate range test (fast). \nMNIST digit, linear 10-10 network.')

plt.subplot(211)
if y2.max() == 0:
    plt.plot(x, y1, '-', fillstyle='none', ms=4.5, mew=0.6, lw=1.5, label='Training dataset')
else:
    plt.plot(x, y2, '-', fillstyle='none', ms=4.5, mew=0.6, lw=1.5, label='Validation dataset')
if y3.max() > 0:
    plt.plot(x, y3, '-', fillstyle='none', ms=4.5, mew=0.6, lw=1.5, label='Validation dataset')
plt.ylabel('Loss')
plt.xscale('log')
plt.yscale('linear')
plt.xlim(1E-6, 1)
plt.ylim(0.01, 2.49)
plt.tick_params(axis='x', which='both', top=True, bottom=False, labelbottom=False, labeltop=True)
set_common_par()
plt.legend(loc='lower left')

plt.subplot(212, sharex=plt.subplot(211))
plt.ylabel('Accuracy')
plt.xlabel('Learning rate')
if y5.max() == 0:
    plt.plot(x, y4, '-', fillstyle='none', ms=4.5, mew=0.6, lw=1.5)
else:
    plt.plot(x, y5, '-', fillstyle='none', ms=4.5, mew=0.6, lw=1.5)
if y6.max() > 0:
    plt.plot(x, y6, '-', fillstyle='none', ms=4.5, mew=0.6, lw=1.5)
plt.ylim(0.01, 0.99)
plt.tick_params(axis='x', which='both', top=False, bottom=True)
set_common_par()
text_box = 'LR_min = 1E-6, LR_max = 1\n' \
           'batch size = 100\n' \
           'step_size = 400\n' \
           'epochs = 4\n' \
           'momentum = 0.90'
props = dict(boxstyle='round', ec='0.8', facecolor='white', alpha=0.8)
plt.text(1.2E-6, 0.92, text_box, size='x-small', ha='left', va='top', bbox=props)

plt.tight_layout(h_pad=0)
plt.savefig(output_pdf, bbox_inches='tight')
plt.show()
plt.close()
