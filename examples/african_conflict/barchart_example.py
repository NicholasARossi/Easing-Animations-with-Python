import pandas as pd
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
    fig_static, static=plt.subplots(figsize=(12,6))
    ax.set_xlim([-1, 10])
    ax.set_ylim([-1, 6])
    lines = []
    colors = ['#355C7D', '#6C5B7B', '#C06C84', '#F67280']
    line_links=[[1, 2],[3, 4],[5, 6],[7,  8]]

    for n, color in enumerate(colors):
        lines.append(ax.fill_between(line_links[n], [0, 0], color=colors[n]))


    num_frames=2000
    interpolations=[]
    data_vect=[0,1,3,4,5,0,1,3,4,5,0,1,3,4,5,0,1,3,4,5]
    t_new=np.linspace(0, len(data_vect), num_frames)
    interpolations.append(easing.Eased(data_vect, data_vect, t_new).No_interp())
    ease_vals=[1,2,5]
    for j in range(3):
        interpolations.append(easing.Eased(data_vect, data_vect, t_new).power_ease(ease_vals[j]))
    for o,interpolation in enumerate(interpolations):
        static.plot(t_new,interpolation,color=colors[o],alpha=0.9,linewidth=5)
    static.set_xlim([0,5])
    fig_static.savefig('media/bar_traces.png',dpi=300)
    bar = progressbar.ProgressBar(max_value=num_frames)

    def animate(z):
        for l,interpolation in enumerate(interpolations):
            #slow way:
            path = lines[l].get_paths()[0]
            path.vertices[[0,3,4,5,6], 1] = interpolation[z]

        bar.update(z)
        return lines

    anim = animation.FuncAnimation(fig_animate, animate,frames=num_frames, blit=False)
    plt.tight_layout()
    anim.save('media/fluid_bar.mp4', writer='ffmpeg',fps=60, bitrate=1800)



if __name__ == "__main__":
    main()

