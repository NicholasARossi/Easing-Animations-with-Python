import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation, rc
rc('animation', html='html5')
from IPython.display import HTML, Image
from itertools import groupby


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

    def timeseries_animation(self,n=1,speed=1.0,interp_freq=0,starting_pos = 25,gif=False,destination=None,plot_kws=None,final_dist=False):
        '''
        This method creates a timeseiers animation of ergodic processes
        :param smoothness:
        :param speed:
        :param interp_freq: This is the number of steps between each given datapoint interp_freq=1 // no additional steps
        :param gif:
        :param destination:
        :param plot_kws:
        :param label:
        :param zero_edges:
        :param loop:
        :return:
        '''
        interp_freq+=1

        data = self.power_ease(n=n, smoothness=interp_freq)

        assert min(data.shape)==1, "timeseries animation only take 1 dimensional arrays"

        data=[k for k, g in groupby(list(data))]

        fig, ax = plt.subplots(1, 2, figsize=(12, 4),gridspec_kw={'width_ratios': [3, 1]},sharey=True)



        max_steps=len(data)


        vanilla_params = {'s': 10, 'color': 'black', 'xlim': [0, starting_pos],
                          'ylim': [np.min(data) - 1, np.max(data) + 1], 'xlabel': '', 'ylabel': '','title': '',
                          'alpha': 1.0, 'figsize': (12, 3),'linestyle':'none','marker':'o'}
        if plot_kws==None:
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
        lines.append(ax[0].plot([], [], linewidth=3, color=plot_kws['color'], alpha=plot_kws['alpha'],linestyle=plot_kws['linestyle'], marker=plot_kws['marker']))
        if 'bins' not in plot_kws.keys():
            plot_kws['bins']=np.linspace(plot_kws['ylim'][0],plot_kws['ylim'][1],20)


        #plotting light grey final dist:
        if final_dist==True:
            bins, x = np.histogram(data,bins=plot_kws['bins'])
            ax[1].plot(bins, x[1:], linewidth=3, drawstyle='steps-pre', color='#d3d3d3')

        else:
            bins, x = np.histogram(data,bins=plot_kws['bins'])
            ax[1].plot(bins, x[1:], linewidth=3, drawstyle='steps-pre', color='#d3d3d3',alpha=0)


        histlines=[]
        histlines.append(ax[1].plot([], [], linewidth=3,  drawstyle='steps-pre',color=plot_kws['color'], alpha=plot_kws['alpha']))


        # This function plots the distribution of flowing information // so we start at the beining and plot forward
        # reverse the orientation of data
        trace_data=data[::-1]


        def animate(z):
            lines[0][0].set_data(x_vect, trace_data[-(starting_pos*interp_freq+1)-z:-1-z])

            # compute the histogram of what what has passed
            if z>0:
                bins, x = np.histogram(trace_data[-(z):-1],bins=plot_kws['bins'])
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

    # simple example : one point moving over time
    data = np.random.random((10, 2))
    Eased(data).scatter_animation2d(n=3, speed=0.5, destination='media/singlepoint.gif')

