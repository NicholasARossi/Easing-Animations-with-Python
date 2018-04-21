import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import matplotlib.cm as cm



class Eased:
    """ This class takes the original time vector and raw data (in n-dimensions) along with an output vector and interpelation function to """

    def __init__(self, data, in_t, out_t):
        self.int_t = in_t
        self.out_t = out_t
        self.n_steps = int(len(out_t) / len(in_t))
        self.data = data
        self.n_dims = len(np.shape(data))


    def test(self):
        self.eased = np.zeros((np.shape(self.data)[0], len(self.out_t)))

        return self.n_steps, np.shape(self.eased)

    def No_interp(self):
        #if else determines if there are multiple dimensions
        if self.n_dims == 1:
            self.eased = np.zeros((len(self.out_t), 1))
            for i, t in enumerate(self.out_t):
                self.eased[i] = self.data[int(np.floor(i / self.n_steps))]
        else:
            self.eased = np.zeros((np.shape(self.data)[0], len(self.out_t)))
            for z in range(np.shape(self.data)[0]):
                for i, t in enumerate(self.out_t):
                    self.eased[z, i] = self.data[z, int(np.floor(i / self.n_steps))]

        return self.eased

    def power_ease(self, n):
        sign = n % 2 * 2
        #         # run over all dimensions
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
            self.eased = np.zeros((np.shape(self.data)[0], len(self.out_t)))
            for z in range(np.shape(self.data)[0]):
                j = 0
                for i in range(len(self.int_t) - 1):

                    start = self.data[z, i]
                    end = self.data[z, i + 1]
                    for t in np.linspace(0, 2, self.n_steps):
                        if (t < 1):
                            val = (end - start) / 2 * t ** n + start

                        else:
                            t -= 2
                            val = (1 - sign) * (-(end - start) / 2) * (t ** n - 2 * (1 - sign)) + start

                        self.eased[z, j] = val
                        j += 1
                self.eased[z, j:] = self.data[z, i + 1]

        return self.eased







if __name__ == "__main__":
    """ This main funciton runs an example animation comapring the different """
    plt.close('all')
    colors = ['#a8e6cf', '#dcedc1', '#ffd3b6', '#ffaaa5', '#ff8b94']
    colors = cm.rainbow(np.linspace(0, 1, 10))

    fig_animate, ax = plt.subplots()
    fig_traces, ax0 = plt.subplots(figsize=(12,4))

    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-0.1,5])

    # data = np.vstack((np.cos(np.linspace(0, 5*np.pi, 10)),np.sin(np.linspace(0, 5*np.pi, 10))))
    data=np.array([0,1,0,1,0,1,0,1,0,1])
    input_time_vector = np.arange(0, 10, 1)
    output_time_vector = np.linspace(0, 10, 2000)
    ease = Eased(data, input_time_vector, output_time_vector)

    data_list = [ease.No_interp()]
    for r in range(9):
        data_list.append(ease.power_ease(r + 1))
    for r in  range(10):
        ax0.plot(output_time_vector,data_list[r],color=colors[r], linewidth=3, alpha=0.75)
    fig_traces.savefig('traces.png',dpi=300)
    dots = []
    for i, data in enumerate(data_list):
        dots.append(ax.plot([], [], linestyle='none', marker='h', markersize=30, color=colors[i]))



    def animate(z):
        for i in range(len(dots)):
            dots[i][0].set_data(data_list[i][z],.25+.5*i)


        return dots

    anim = animation.FuncAnimation(fig_animate, animate, frames=len(output_time_vector), blit=False)



    writer = animation.writers['ffmpeg'](fps=60)
    dpi=300
    anim.save('interp.mp4', writer=writer,dpi=dpi)
