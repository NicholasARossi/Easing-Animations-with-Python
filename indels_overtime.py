
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

from IPython.display import HTML, Image
import moviepy.editor as mpy



def bch_1(data,gene):
    #ata=pd.DataFrame(abs(np.random.random((3, 10))), index=['one', 'two', 'three'])
    easing.Eased(data).barchart_animation(destination='outputs/{}.mp4'.format(gene),plot_kws={'ylim':[0,100],'figsize':[12,4],'title':gene},smoothness=5,label=True)


if __name__ == '__main__':
    SOS_df = pd.read_csv('/Users/nicholas.rossi/Documents/Code/2019/07/rela_gif/2019-04-24T092041-sequencing-etl-dump.csv')
    destination = '/Users/nicholas.rossi/Documents/Code/2019/04/ICE_trails/output_files/ensemble_trial/sos/'
    ICE = wrangle.LoadICE(dropbad=True)

    ICE.load_data(destination)

    newdf = pd.merge(SOS_df, ICE.indels, on='sample_name')
    newdf['gene'] = [x.split('-')[0].split('+')[0].split('_')[0] for x in newdf['sample_name']]

    #for gene in set(newdf['gene']):
    gene='Positive'

    temp = newdf[newdf['gene'] == gene]
    time_indel_df = temp[['created', '-30', '-29', '-28',
                          '-27', '-26', '-25', '-24', '-23', '-22', '-21', '-20', '-19', '-18',
                          '-17', '-16', '-15', '-14', '-13', '-12', '-11', '-10', '-9', '-8',
                          '-7', '-6', '-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5',
                          '6', '7', '8', '9', '10', '11', '12', '13', '14']]
    time_indel_df.loc[:,'created'] = pd.to_datetime(time_indel_df.created)
    time_indel_df = time_indel_df.sort_values(by='created')
    time_indel_df = time_indel_df.set_index('created')
    time_indel_df.columns = pd.to_numeric(time_indel_df.columns)





    bch_1(time_indel_df,gene)


