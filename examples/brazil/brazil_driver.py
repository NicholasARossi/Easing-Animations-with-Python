import pandas as pd
import numpy.random as rnd
import numpy as np
from matplotlib import animation, rc
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import progressbar
import geopy.distance
from matplotlib.colors import LinearSegmentedColormap
import sys
sys.path.insert(0, '../..')

import easing


plt.style.use('rossidata')


def main():

    ### The data is too large to be commited to github so you can see the rest of the flights here
    data = pd.read_csv('BrFlights2.csv', encoding="ISO-8859-1")



    fig_animate, ax = plt.subplots(figsize=(20, 12))

    colors = ["#009B3A", "#FEDF00", "#002776"]

    cmap = LinearSegmentedColormap.from_list('my_colormap',
                                             [(0, colors[0]),
                                              (0.5, colors[1]),
                                              (1, colors[2])], N=1000)

    m = Basemap(resolution='c', llcrnrlon=-180, llcrnrlat=-90, urcrnrlon=180, urcrnrlat=90)
    data2 = data.drop_duplicates(subset=['LongDest', 'LatDest', 'LongOrig', 'LatOrig'], keep="last")

    N_vals = len(data2)
    x_stack = np.array([0, 0, 0, 0])
    y_stack = np.array([0, 0, 0, 0])
    distances = []
    # bar = progressbar.ProgressBar(max_value=N_vals)

    for r in range(N_vals):
        x, y = m(data2['LongOrig'].iloc[r], data2['LatOrig'].iloc[r])

        x2, y2 = m(data2['LongDest'].iloc[r], data2['LatDest'].iloc[r])
        #     distances.append((x-x2)**2+(y-y2)**2)
        cords1 = (data2['LatOrig'].iloc[r], data2['LongOrig'].iloc[r])
        cords2 = (data2['LatDest'].iloc[r], data2['LongDest'].iloc[r])
        distances.append(geopy.distance.vincenty(cords1, cords2).km)

        x_stack = np.vstack((x_stack, np.array([x, x, x2, x2])))
        y_stack = np.vstack((y_stack, np.array([y, y, y2, y2])))
    # bar.update(r)

    x_stack = x_stack[~(x_stack == 0).any(1)]
    y_stack = y_stack[~(y_stack == 0).any(1)]

    indexz = np.unique(x_stack, return_index=True, axis=0)[1]
    distances = np.array(distances)[indexz]

    x_stack = x_stack[indexz, :]
    y_stack = y_stack[indexz, :]

    Xease = Eased(x_stack, np.array([0, 1, 2, 3]), np.linspace(0, 4, 600))
    Yease = Eased(y_stack, np.array([0, 1, 2, 3]), np.linspace(0, 4, 600))
    Xeased = Xease.power_ease(3)
    Yeased = Yease.power_ease(3)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(2, 2))
    # bins=np.logspace(2,4.5,30)
    bins = np.linspace(100, 10000, 25)
    n_bins = len(bins)
    n, bins, patches = ax.hist(distances, bins=bins)

    col = np.linspace(0, 1, n_bins)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cmap(c))
    ax.set_xlabel('Distance (kilometers)')

    ax.set_ylabel('Number of Flights')
    # ax.set_xscale('log')
    fig.savefig('distances.png', transparent=True, dpi=500)
    for j in range(np.shape(x_stack)[0]):
        val = np.random.randint(0, high=60)

        tempx = Xeased[j, val:]
        tempx = np.append(tempx, tempx[-1] * np.ones(val))
        Xeased[j, :] = tempx

        tempy = Yeased[j, val:]
        tempy = np.append(tempy, tempy[-1] * np.ones(val))
        Yeased[j, :] = tempy


    plt.close('all')


    fig_animate,ax=plt.subplots(figsize=(20,12))

    m = Basemap(resolution='c',llcrnrlon=-140,llcrnrlat=-90,urcrnrlon=90,urcrnrlat=90)

    shp_info = m.readshapefile('coastline/brazil_coastline','country',drawbounds=False,color='white', antialiased=1, ax=ax)

    m.drawcoastlines()
    m.drawmapboundary(fill_color='white')

    lines=[]
    max_dist=np.log10(max(distances))
    min_dist=np.log10(min(distances))
    for j in range(len(Xeased[:,0])):
        lines.append(ax.plot(Xeased[j,0],Yeased[j,0],color=cmap((np.log10(distances[j])-min_dist)/(max_dist-min_dist)),linestyle='none',marker='o'))

    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)


    def animate(z):
        for i in range(len(Xeased[:,0])):
            lines[i][0].set_data(Xeased[i,z], Yeased[i,z])
        bar.update(z)

        return lines
    plt.axis('off')
    anim3 = animation.FuncAnimation(fig_animate, animate,frames=600,blit=False)

    anim3.save('brazil_long.mp4', writer='ffmpeg',bitrate=1800)

if __name__ == "__main__":
    main()
