import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import pandas as pd

from matplotlib import animation, rc
rc('animation', html='html5')

from IPython.display import HTML, Image
import moviepy.editor as mpy



# ok so the objective here is to take a
# class GIF:
#
#     """ This class takes the original time vector and raw data (as a m*n matrix) along with an output vector and interpolation function
#     For the input data, the rows are the different variables and the columns correspond to the time points"""
#
#     def __init__(self, data, in_t, out_t):
#         self.int_t = in_t
#         self.out_t = out_t
#         self.n_steps = np.ceil(len(out_t) / len(in_t))
#         self.data = data
#         self.n_dims = len(np.shape(data))
#
#
#     def(video:
#
#
#     return self.eased






class Eased:
    """ This class takes the original time vector and raw data (as a m*n matrix or dataframe) along with an output vector and interpolation function
    For the input data, the rows are the different variables and the columns correspond to the time points"""

    def __init__(self, data, in_t=None):

        if isinstance(data, pd.DataFrame):
            self.int_t = np.arange(len(data.index))
            self.data = data.values
            self.n_dims = data.shape[1]
        elif isinstance(data, np.ndarray):
            if in_t is None:
                in_t=np.arange(np.shape(data)[0])
                print("No time vector included - defaulting to array length")

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


    def scatter_animation2d(self,n=3,smoothness=10,speed=1.0,color='#a8e6cf',inline=True,gif=False,destination=None,plot_kws=None):
        """
        Flexibly create a 2d scatter plot animation.

        This function creates a matplotlib animation from a pandas Dataframe or a MxN numpy array. The Columns are paired
        with x and y coordinates while the rows are the individual time points.

        This takes a number of parameters for the animation, as well as


        Parameters
        ----------
        :param n: Exponent of the power smoothing
        :param smoothness: how smooth the frames of the animation are
        :param speed: speed
        :param color:
        :param inline:
        :param gif:
        :param destination:
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

        fig, ax = plt.subplots()
        ax.set_xlim([min(it_data[0,:])-1,max(it_data[0,:])+1])
        ax.set_ylim([min(it_data[1,:])-1,max(it_data[1,:])+1])


        n_dots=int(np.shape(self.data)[1]/2)
        dots=[]

        for i in range(n_dots):
            dots.append(ax.plot([], [], linestyle='none', marker='o', markersize=20, color=color))


        def animate(z):
            for i in range(n_dots):
                dots[i][0].set_data(it_data[z,i*2],it_data[z,i*2+1])


            return dots

        anim = animation.FuncAnimation(fig, animate, frames=len(self.out_t),interval=400/smoothness/speed, blit=False)


        if destination is not None:
            writer = animation.writers['ffmpeg'](fps=60)
            anim.save(destination, writer=writer, dpi=100)


        if gif==True:
            if destination is not None:
                anim.save(destination,writer='imagemagick',fps=smoothness)
            return Image(url='animation.gif')
        else:
            return anim




if __name__ == "__main__":
    data = np.random.rand(12,100)
    #d1data=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).T
    #
    # time = np.arange(np.shape(data)[1])

    #df = pd.DataFrame(data
    #    {'num_legs': np.sin(np.linspace(0, 2 * np.pi, 10)), 'num_wings': np.cos(np.linspace(0, 2 * np.pi, 10))})

    #print(Eased(data).power_ease(n=3))

    Eased(data).scatter_animation2d(n=5, smoothness=40, speed=2,destination="gifo.mp4", gif=False)
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
