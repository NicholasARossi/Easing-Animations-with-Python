import matplotlib.pyplot as plt
plt.style.use('rossidata')
import numpy as np
import pandas as pd
from matplotlib import animation, rc
rc('animation', html='html5')
from IPython.display import HTML, Image



class Eased:
    """ This class takes the original time vector and raw data (as a m*n matrix or dataframe) along with an output vector and interpolation function
    For the input data, the rows are the different variables and the columns correspond to the time points"""

    def __init__(self, data, in_t=None):

        if isinstance(data, pd.DataFrame):
            self.labels=np.append(data.index.values,np.array([data.index.values[0],data.index.values[0]]))
            self.int_t = np.arange(len(self.labels)-1)


            self.data = np.vstack((data.values,data.values[0,:]))
            self.n_dims = data.shape[1]
            self.columns=data.columns
        elif isinstance(data, np.ndarray):
            if in_t is None:
                in_t=np.arange(np.shape(data)[0])
                print("No time vector included - defaulting to number of rows")

            self.int_t = in_t
            self.data = data
            self.n_dims = len(np.shape(data))
        else:
            print('\033[91m' + "Data is unrecognized type : must be either a numpy array or pandas dataframe")


    def No_interp(self,smoothness=10):
        out_t=np.linspace(min(self.int_t),max(self.int_t),len(self.int_t)*smoothness)
        self.n_steps = np.ceil(len(out_t) / len(self.int_t))
        self.out_t = out_t

        #This Function maps the input vecotor over the outuput time vector without interoplation
        if self.n_dims == 1: # if the input is only one row
            self.eased = np.zeros((len(self.out_t), 1))
            for i, t in enumerate(self.out_t):
                self.eased[i] = self.data[int(np.floor(i / self.n_steps))]
        else: #if the input is a multidimensional row
            self.eased = np.zeros((np.shape(self.data)[0], len(self.out_t)))
            for z in range(np.shape(self.data)[0]):
                for i, t in enumerate(self.out_t):
                    self.eased[z, i] = self.data[z, int(np.floor(i / self.n_steps))]

        return self.eased

    def power_ease(self, n,smoothness=10):
        out_t=np.linspace(min(self.int_t),max(self.int_t),len(self.int_t)*smoothness)
        self.n_steps = np.ceil(len(out_t) / len(self.int_t))
        self.out_t = out_t
        sign = n % 2 * 2
        if self.n_dims == 1:
            self.eased = np.zeros((len(self.out_t), 1))
            j = 0
            for i in range(len(self.int_t) - 1):

                start = self.data[i]
                end = self.data[i + 1]
                for t in np.linspace(0, 2, self.n_steps):
                    if (t < 1):
                        val = (end - start) / 2 * t ** n + start

                    else:
                        t -= 2
                        val = (1 - sign) * (-(end - start) / 2) * (t ** n - 2 * (1 - sign)) + start

                    self.eased[j] = val
                    j += 1
            self.eased[j:] = self.data[i + 1]

        else:
            self.eased = np.zeros(( len(self.out_t),np.shape(self.data)[1]))
            for z in range(np.shape(self.data)[1]):
                j = 0
                for i in range(len(self.int_t) - 1):

                    start = self.data[ i,z]
                    end = self.data[ i + 1,z]
                    for t in np.linspace(0, 2, self.n_steps):
                        if (t < 1):
                            val = (end - start) / 2 * t ** n + start

                        else:
                            t -= 2
                            val = (1 - sign) * (-(end - start) / 2) * (t ** n - 2 * (1 - sign)) + start

                        self.eased[ j,z] = val
                        j += 1
                self.eased[ j:,z] = self.data[ i + 1,z]

        return self.eased


    def scatter_animation2d(self,n=3,smoothness=30,speed=1.0,gif=False,destination=None,plot_kws=None,label=False):
        """
        Flexibly create a 2d scatter plot animation.

        This function creates a matplotlib animation from a pandas Dataframe or a MxN numpy array. The Columns are paired
        with x and y coordinates while the rows are the individual time points.

        This takes a number of parameters for the animation, as well as


        Parameters
        ----------
        n: Exponent of the power smoothing
        smoothness: how smooth the frames of the animation are
        speed: speed
        inline:
        gif:
        destination:
        :return:
        """



        #Running checks on data for mishappen arrays.
        if np.shape(self.data)[1]%2!=0:
            print('\033[91m' + "Failed: Data must have an even number of columns")
            exit()
        if np.shape(self.data)[0]<np.shape(self.data)[1]:
            print('\033[91m' + "Warning : Data has more columns (xys) than rows (time)")


        if plot_kws is None:
            plot_kws = dict()


        it_data=self.power_ease(n,smoothness)

        # filling out missing keys
        vanilla_params={'s':10,'color':'black','xlim':[np.min(it_data)-1, np.max(it_data)+1],'ylim':[np.min(it_data)-1,np.max(it_data)+1],'xlabel':'','ylabel':'','alpha':1.0,'figsize':(6,6)}
        for key in vanilla_params.keys():
            if key not in plot_kws.keys():
                plot_kws[key] = vanilla_params[key]



        fig, ax = plt.subplots(figsize=plot_kws['figsize'])
        ax.set_xlim(plot_kws['xlim'])
        ax.set_ylim(plot_kws['ylim'])
        ax.set_xlabel(plot_kws['xlabel'])
        ax.set_ylabel(plot_kws['ylabel'])

        if label==True:
            label_text = ax.text(plot_kws['xlim'][1]*0.75, plot_kws['ylim'][1]*.9, '',fontsize=18)

        n_dots=int(np.shape(self.data)[1]/2)
        dots=[]
        for i in range(n_dots):
            dots.append(ax.plot([], [], linestyle='none', marker='o', markersize=plot_kws['s'], color=plot_kws['color'], alpha=plot_kws['alpha']))



        def animate(z):
            for i in range(n_dots):
                dots[i][0].set_data(it_data[z,i*2],it_data[z,i*2+1])
            if label==True:
                label_text.set_text(self.labels[int(np.floor((z+smoothness/2)/smoothness))])
                return dots,label_text
            else:
                return dots

        anim = animation.FuncAnimation(fig, animate, frames=len(self.out_t),interval=400/smoothness/speed, blit=False)


        if destination is not None:
            if destination.split('.')[-1]=='mp4':
                writer = animation.writers['ffmpeg'](fps=60)
                anim.save(destination, writer=writer, dpi=100)
            if destination.split('.')[-1]=='gif':
                anim.save(destination, writer='imagemagick', fps=smoothness)

        if gif==True:
            return Image(url='animation.gif')
        else:
            return anim





    def barchart_animation(self,n=3,smoothness=30,speed=1.0,gif=False,destination=None,plot_kws=None,label=False,zero_edges=True,loop=True):
        '''
        This barchart animation create line barcharts that morph over time using the eased data class

        It takes the following additional arguments
        :param n: this is the power curve modifier to passed to power_ease
        :param smoothness: this is a rendering parameter that determines the relative framerate over the animation
        :param speed: How quickly does the animation unfold // a value of 1 indicates the default [R>0]

        :param destination: This is the output file (if none it will be displayed inline for jupyter notebooks) - extension determines filetype


        :param plot_kws: These are the matplotlib key work arghuments that can be passed in the event the defaults don't work great
        :param label: This is an optional paramter that will display labels of the pandas rows as the animation cycles through

        :return: rendered animation
        '''

        #
        # if loop==True:
        #     self.data=np.vstack((self.data,self.data[0,:]))
        #     # self.labels=list(self.labels)+[self.labels[0]]

        it_data = self.power_ease(n, smoothness)

        x_vect=np.arange(len(self.columns))


        ### running checks on the paramters

        #Runing checks on parameters
        assert speed>0, "Speed value must be greater than zero"


        # filling out missing keys
        vanilla_params = {'s': 10, 'color': 'black', 'xlim': [min(x_vect) - 1, max(x_vect) + 1],
                          'ylim': [np.min(it_data) - 1, np.max(it_data) + 1], 'xlabel': '', 'ylabel': '','title': '',
                          'alpha': 1.0, 'figsize': (6, 6)}
        for key in vanilla_params.keys():
            if key not in plot_kws.keys():
                plot_kws[key] = vanilla_params[key]

        fig, ax = plt.subplots(figsize=plot_kws['figsize'])
        ax.set_xlim(plot_kws['xlim'])
        ax.set_ylim(plot_kws['ylim'])
        ax.set_title(plot_kws['title'])
        ax.set_xlabel(plot_kws['xlabel'])
        ax.set_ylabel(plot_kws['ylabel'])
        ax.set_xticks(x_vect-np.mean(np.diff(x_vect))/2)
        ax.set_xticklabels(list(self.columns),rotation=90)

        plt.tight_layout()
        if label == True:
            label_text = ax.text(plot_kws['xlim'][1] * 0.25, plot_kws['ylim'][1] * .9, '', fontsize=18)

        lines=[]
        lines.append(ax.plot([], [], linewidth=3, drawstyle='steps-pre', color=plot_kws['color'], alpha=plot_kws['alpha']))


        # add zero padding to the data // makes for prettier histogram presentation
        if zero_edges==True:
            zero_pad=np.zeros((it_data.shape[0],1))
            it_data=np.hstack((zero_pad,it_data,zero_pad))
            x_vect=[min(x_vect)-1]+list(x_vect)+[max(x_vect)+1]

        def animate(z):
            lines[0][0].set_data(x_vect, it_data[z, :])

            if label==True:
                label_text.set_text(self.labels[int(np.floor((z+smoothness/2)/smoothness))])
                return lines,label_text
            else:
                return lines


        anim = animation.FuncAnimation(fig, animate, frames=it_data.shape[0],interval=400/smoothness/speed, blit=False)


        if destination is not None:
            if destination.split('.')[-1]=='mp4':
                writer = animation.writers['ffmpeg'](fps=60)
                anim.save(destination, writer=writer, dpi=100)
            if destination.split('.')[-1]=='gif':
                anim.save(destination, writer='imagemagick', fps=smoothness)

        if gif==True:
            return Image(url='animation.gif')
        else:
            return anim

    def timeseries_animation(self,n=1,speed=1.0,interp_freq=1,starting_pos = 25,gif=False,destination=None,plot_kws=None,norm_hist=True,n_bins=20):
        '''
        This method creates a timeseiers animation of ergodic processes
        :param smoothness:
        :param speed:
        :param starting_pos:
        :param gif:
        :param destination:
        :param plot_kws:
        :param label:
        :param zero_edges:
        :param loop:
        :return:
        '''


        data = self.power_ease(n, interp_freq)


        fig, ax = plt.subplots(1, 2, figsize=(12, 4),gridspec_kw={'width_ratios': [3, 1]},sharey=True)



        # filling out missing keys

        max_steps=len(data)


        vanilla_params = {'s': 10, 'color': 'black', 'xlim': [0, starting_pos],
                          'ylim': [np.min(data) - 1, np.max(data) + 1], 'xlabel': '', 'ylabel': '','title': '',
                          'alpha': 1.0, 'figsize': (12, 3),'linestyle':'o'}
        plot_kws={}
        x_vect=np.linspace(1,starting_pos,starting_pos*interp_freq)

        # Creating NaN padding at the end for time series plot
        data = np.append(data, x_vect * np.nan)

        # fill out parameters
        for key in vanilla_params.keys():
            if key not in plot_kws.keys():
                plot_kws[key] = vanilla_params[key]

        ax[0].set_ylim(plot_kws['ylim'])
        ax[1].set_ylim(plot_kws['ylim'])

        ax[0].set_xlim(plot_kws['xlim'])
        lines=[]
        lines.append(ax[0].plot([], [], linewidth=3, color=plot_kws['color'], alpha=plot_kws['alpha'],linestyle='none', marker='o'))

        set_bins=np.linspace(plot_kws['ylim'][0],plot_kws['ylim'][1],n_bins)
        bins, x = np.histogram(data,bins=set_bins,normed=norm_hist)
        ax[1].plot(bins, x[1:], linewidth=3, drawstyle='steps-pre', color='#d3d3d3', alpha=0.25)

        #TODO add t- xlabels for ts plot

        histlines=[]
        histlines.append(ax[1].plot([], [], linewidth=3,  drawstyle='steps-pre',color=plot_kws['color'], alpha=plot_kws['alpha']))

        # This function plots the distribution of flowing information // so we start at the beining and plot forward
        # reverse the orientation of data
        trace_data=data[::-1]


        def animate(z):
            lines[0][0].set_data(x_vect, trace_data[-(starting_pos*interp_freq+1)-z:-1-z])

            # compute the histogram of what what has passed
            if z>0:
                bins, x = np.histogram(trace_data[-(z):-1],bins=set_bins,normed=norm_hist)
                histlines[0][0].set_data(bins,x[1:])
                lines.append(ax[1].plot([], [], linewidth=3, color=plot_kws['color'], alpha=plot_kws['alpha']))


            return lines


        anim = animation.FuncAnimation(fig, animate, frames=max_steps,interval=400/speed, blit=False)


        if destination is not None:
            if destination.split('.')[-1]=='mp4':
                writer = animation.writers['ffmpeg'](fps=60)
                anim.save(destination, writer=writer, dpi=100)
            if destination.split('.')[-1]=='gif':
                anim.save(destination, writer='imagemagick', fps=30)

        if gif==True:
            return Image(url='animation.gif')
        else:
            return anim



if __name__ == "__main__":
    ### This class can create a number of plots // below are a few of the examples. Because this code is modular, these should be view as a jumping off point.

    # #barchart_example
    # data=pd.DataFrame(abs(np.random.random((3, 10))), index=['one', 'two', 'three'])
    # Eased(data).barchart_animation(destination='outputs/output.mp4',plot_kws={'ylim':[0,1]},smoothness=40,label=True)


    #Time Series Example

    data = np.random.rand(100, 1)
    Eased(data).timeseries_animation(starting_pos=25, speed=0.5, norm_hist=False)

    #

    ###plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})

    #




    # # Pep data for text based scatter change
    # data=pd.read_csv('examples/HDR_comparison.csv')
    # data=data[['ngs','iceV1','iceV2']]
    # d1=data.drop(columns='iceV2').stack().reset_index(drop=True)
    # d2=data.drop(columns='iceV1').stack().reset_index(drop=True)
    #
    # new_indexs=[]
    # for idx, row in d1.iteritems():
    #     if isinstance(row,str):
    #         d1.drop(idx, inplace=True)
    #         d2.drop(idx,inplace=True)
    #
    # # d1.index=new_indexs
    # # d2.index = new_indexs
    #
    # data=pd.DataFrame([d1, d2], index=['ice V1', 'ice V2'])
    # # stack a few times
    # data=data.append(data.loc['ice V1'])
    # Eased(data).scatter_animation2d(destination='output.gif',plot_kws={'xlim':[-5,70],'ylim':[-5,70],'xlabel':'NGS HDR %','ylabel':'ICE HDR %','figsize':(6,6)},smoothness=40,label=True)
    # # so we're going to organize this data so it's in the structure that works with this everyother





    # data = np.random.rand(12,100)
    #d1data=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).T
    #
    # time = np.arange(np.shape(data)[1])

    #df = pd.DataFrame(data
    #    {'num_legs': np.sin(np.linspace(0, 2 * np.pi, 10)), 'num_wings': np.cos(np.linspace(0, 2 * np.pi, 10))})

    #print(Eased(data).power_ease(n=3))

    # Eased(data).scatter_animation2d(n=5, smoothness=40, speed=2,destination="gifo.mp4", gif=False,plot_kws={"color": "k", "s": 20})
    # df = pd.DataFrame({'num_legs': [2, 4, 8, 0], 'num_wings': [2, 0, 0, 0], 'num_specimen_seen': [10, 2, 1, 8]},
    #                   index=['falcon', 'dog', 'spider', 'fish'])
    #
    # Eased(np.arange(10),np.arange(10))
    #
    # data = np.array(([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]))
    # time = np.arange(np.shape(data)[1])
    # Eased(data, time).scatter_animation2d()


    #
    # """ This main funciton runs an example animation comapring the different animation styles """
    # plt.close('all')
    # colors = ['#a8e6cf', '#dcedc1', '#ffd3b6', '#ffaaa5', '#ff8b94']
    # colors = cm.rainbow(np.linspace(0, 1, 10))
    #
    # fig_animate, ax = plt.subplots()
    # fig_traces, ax0 = plt.subplots(figsize=(12,4))
    #
    # ax.set_xlim([-0.1,1.1])
    # ax.set_ylim([-0.1,5])
    #
    # data=np.array([0,1,0,1,0,1,0,1,0,1])
    # input_time_vector = np.arange(0, 10, 1)
    # output_time_vector = np.linspace(0, 10, 2000)
    # ease = Eased(data, input_time_vector, output_time_vector)
    # labels=['No Interpolation']
    # data_list = [ease.No_interp()]
    # for r in range(9):
    #     data_list.append(ease.power_ease(r + 1))
    #     labels.append(str(r))
    # for r in  range(10):
    #     ax0.plot(output_time_vector[0:401],data_list[r][0:401],color=colors[r], linewidth=3, alpha=0.75,label=labels[r])
    # ax0.legend(title='exponent')
    # plt.axis('off')
    # fig_traces.savefig('media/traces.png',dpi=300)
    # dots = []
    # for i, data in enumerate(data_list):
    #     dots.append(ax.plot([], [], linestyle='none', marker='h', markersize=30, color=colors[i]))
    #
    #
    #
    # def animate(z):
    #     for i in range(len(dots)):
    #         dots[i][0].set_data(data_list[i][z],.25+.5*i)
    #
    #
    #     return dots
    #
    # anim = animation.FuncAnimation(fig_animate, animate, frames=len(output_time_vector), blit=False)
    #
    #
    #
    # writer = animation.writers['ffmpeg'](fps=60)
    # dpi=300
    # anim.save('media/interp.mp4', writer=writer,dpi=dpi)
