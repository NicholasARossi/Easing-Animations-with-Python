
# Basic Packages
import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import numpy as np
plt.style.use('rossidata')
from tqdm import tqdm_notebook as tqdm
import time
import pickle as pkl
import glob
from collections import Counter
import sys

sys.path.append('/Users/nicholas.rossi/Documents/Code/2019/05/tools_notebooks/bioinformatics/tools/')
sys.path.append('/Users/nicholas.rossi/Documents/Code/2019/05/tools_notebooks/bioinformatics/tools/iceio')
from iceio import wrangle,gentools,ICE_scraper,dump_modifier

import matplotlib.pyplot as plt
plt.style.use('rossidata')
import numpy as np
import matplotlib.cm as cm
import pandas as pd

from matplotlib import animation, rc
rc('animation', html='html5')
import easing
from collections import Counter
from IPython.display import HTML, Image
import moviepy.editor as mpy



def bch_1(data,gene):
    #ata=pd.DataFrame(abs(np.random.random((3, 10))), index=['one', 'two', 'three'])
    easing.Eased(data).barchart_animation(destination='outputs/{}.mp4'.format(gene),plot_kws={'ylim':[0,100],'figsize':[12,4],'title':gene},smoothness=5,label=True)


if __name__ == '__main__':
    SOS_df = pd.read_csv('/Users/nicholas.rossi/Documents/Code/2019/07/rela_gif/2019-04-24T092041-sequencing-etl-dump.csv')
    destination = '/Users/nicholas.rossi/Documents/Code/2019/04/ICE_trails/output_files/ensemble_trial/singleplex/'
    ICE = wrangle.LoadICE(dropbad=True)

    ICE.load_data(destination)

    newdf = ICE.indels

    newdf['gene'] = [x.split('-')[0].split('+')[0].split('_')[0] for x in newdf.index]
    newdf = newdf.set_index('gene')
    #temp = newdf[newdf.index == 'BUB1B']

    genes=Counter([x.split('-')[0].split('+')[0].split('_')[0] for x in newdf.index]).most_common(10)

    for gene in set(newdf['gene']):

        temp = newdf[newdf.index==gene[0]]
        temp.index = ['Sample: {}'.format(r + 1) for r in range(len(temp))]





        bch_1(temp,gene[0])


