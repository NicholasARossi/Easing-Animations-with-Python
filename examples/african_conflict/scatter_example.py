import numpy as np
from matplotlib import animation, rc
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
import easing
import progressbar

def main():
    plt.close('all')
    fig_animate, ax = plt.subplots(figsize=(12, 12))
    fig_static, static =plt.subplots(figsize=(12, 12))
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    lines = []
    colors = ['#355C7D', '#6C5B7B', '#C06C84', '#F67280']
    line_links=[[1, 2],[3, 4],[5, 6],[7,  8]]

    for n, color in enumerate(colors):
        lines.append(ax.fill_between(line_links[n], [0, 0], color=colors[n]))


    num_frames=2000
    x_interpolations=[]
    y_interpolations=[]

    proxy_vect=np.linspace(0, 10*np.pi,10)
    xdata_vect=-np.cos(proxy_vect)
    ydata_vect=np.sin(proxy_vect)

    t_new=np.linspace(0, 2*np.pi, num_frames)
    t_int=proxy_vect
    x_interpolations.append(easing.Eased(xdata_vect,  t_int, t_new).No_interp())
    y_interpolations.append(easing.Eased(ydata_vect,  t_int, t_new).No_interp())

    ease_vals=[1,3,5]
    #
    for j in ease_vals:
        x_interpolations.append(easing.Eased(xdata_vect, t_int, t_new).power_ease(j))
        y_interpolations.append(easing.Eased(ydata_vect, t_int, t_new).power_ease(j))


    for o,interpolation in enumerate(x_interpolations):
        static.scatter(x_interpolations[o],y_interpolations[o],color=colors[o],alpha=0.9,linewidth=1)

    # static.set_xlim([0,5])
    fig_static.savefig('media/scatter_traces.png',dpi=300)
    points=[]
    for n, color in enumerate(colors):
        points.append(ax.plot([],[],linestyle='none',marker='o',color=color,markersize=50,alpha=0.9))



    bar = progressbar.ProgressBar(max_value=num_frames)

    def animate(z):
        for l,point in enumerate(x_interpolations):
            #slow way:
            points[l][0].set_data(x_interpolations[l][z], y_interpolations[l][z])

        bar.update(z)
        return lines

    anim = animation.FuncAnimation(fig_animate, animate,frames=num_frames, blit=False)
    plt.tight_layout()
    anim.save('media/fluid_scatter.mp4', writer='ffmpeg',fps=60, bitrate=1800)



if __name__ == "__main__":
    main()

