import pandas as pd
import numpy as np
from matplotlib import animation, rc
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
import easing
import time

import progressbar

# plt.style.use('rossidata')

### The functions below are used to make smooth colors
def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
  ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
  # Starting and ending colors in RGB form
  s = hex_to_RGB(start_hex)
  f = hex_to_RGB(finish_hex)
  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)

  return RGB_list

def hex_to_RGB(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def color_scale_index(fats,colors):
    return np.divide(colors[int((np.log10(fats)-2)/(4.4-2)*100)],255)


### the functions below are called to make certain dots **pop**

def traj_compute(n_points, hill):
    x_vect = np.linspace(0, 10, n_points)
    y = (x_vect / 5) ** hill / (1 + (x_vect / 5) ** hill)

    y2 = ((x_vect / 5) ** hill / (1 + (x_vect / 5) ** hill))[::-1] / 2 + 0.5

    return np.append(y, y2) * 1.3333333333333333


def size_compute(traj, tot_size, t_vect, year):
    temp = np.zeros(len(t_vect))
    idx = np.where(t_vect == float(year))[0][0]
    temp[idx:] = 0.5
    temp[idx - len(traj):idx] = traj
    return temp * tot_size



def size_matrix_compute(t_vect, years, sizes, traj):
    size_matrix = np.zeros((len(t_vect), len(years)))
    for i, year in enumerate(years):
        size_matrix[:, i] = size_compute(traj, sizes[i], t_vect, years[i])
    return size_matrix



def main():
    ### Loading Africa Data
    data = pd.read_csv('african_conflicts-1.csv', encoding="ISO-8859-1")


    ### plotting histograms:
    data = data.sort_values(by=['YEAR'])
    data['SUM_FATALITIES'] = np.cumsum(data['FATALITIES'])

    cum_sums = [0]
    maximums = pd.DataFrame()

    # sorting the data an caluculating the cumulative deaths over time
    for j, yar in enumerate(np.arange(1997, 2018)):
        cum_sums.append(data[data.YEAR == yar].iloc[-1].SUM_FATALITIES)
        # This is the largest death per year, for tracking the ping-poing
        maximums = maximums.append(data[data.YEAR == yar].sort_values(by=['FATALITIES']).iloc[-1])



    temp = maximums.iloc[0][:]
    temp['FATALITIES'] = 0
    temp['YEAR'] = 1996
    maximums = maximums.append(temp)

    maximums = maximums.sort_values(by=['YEAR'])
    plt.close('all')

    fig_animate, ax = plt.subplots(figsize=(12, 12))

    ax.set_xlim([-1, 10])
    ax.set_ylim([-1, 800000])

    timetext = ax.text(0, 800000, '')
    lines = []
    colors = ['#a8e6cf', '#dcedc1', '#ffd3b6', '#ffaaa5']
    line_links=[[1, 2],[3, 4],[5, 6],[7,  8]]


    for n, color in enumerate(colors):
        lines.append(ax.fill_between(line_links[n], [0, 0], color=colors[n]))


    num_frames=1000
    interpolations=[]
    interpolations.append(easing.Eased(cum_sums, np.arange(0, len(cum_sums), 1), np.linspace(0, len(cum_sums), num_frames)).No_interp())

    for j in range(3):
        interpolations.append(easing.Eased(cum_sums, np.arange(0, len(cum_sums), 1), np.linspace(0, len(cum_sums), num_frames)).power_ease(j+1))

    bar = progressbar.ProgressBar(max_value=num_frames)

    def animate(z):
        for l,interpolation in enumerate(interpolations):
            #slow way:
            path = lines[l].get_paths()[0]
            path.vertices[[0,3,4,5,6], 1] = interpolation[z]
        #     #fast way:
        #     # lines[l][0].set_data(line_links[l], [0, interpolation[z], interpolation[z], 0])



        timetext.set_text(str(z))
        bar.update(z)

        return lines

    anim3 = animation.FuncAnimation(fig_animate, animate,frames=1000, blit=False)
    plt.tight_layout()
    anim3.save('media/total_fataliites.mp4', writer='ffmpeg',fps=60, bitrate=1800)




    # ### Ploting
    #
    # sns.set_style('white')
    # plt.close('all')
    # fig,ax1=plt.subplots(figsize=(12,12))
    # # colors=linear_gradient("#00CED1","#FC89AC",n=100)
    # colors=linear_gradient("#fc00ff","#00dbde",n=100)
    #
    # # Lambert Conformal map of lower 48 states.
    # m = Basemap(resolution='c',llcrnrlon=-25,llcrnrlat=-40,urcrnrlon=60,urcrnrlat=40)
    #
    # shp_info = m.readshapefile('maps/Africa','states',drawbounds=True,color='slategrey', antialiased=1, ax=ax1)
    # m.drawcoastlines()
    # x_points=[]
    # y_points=[]
    # years=[]
    # sizes=[]
    # fatalities=[]
    # data=data[data['FATALITIES']>100]
    # for r in range(len(data)):
    #     if data['FATALITIES'].iloc[r]:
    #         if data['YEAR'].iloc[r]:
    #             x, y = m(data['LONGITUDE'].iloc[r], data['LATITUDE'].iloc[r])
    #             try:
    #                 x_points.append(float(x))
    #                 y_points.append(float(y))
    #                 years.append(data['YEAR'].iloc[r])
    #                 sizes.append(data['FATALITIES'].iloc[r])
    #
    #             except:
    #                 print(x,y)



if __name__ == "__main__":
    main()
