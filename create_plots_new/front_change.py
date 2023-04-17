import cv2

import numpy as np
import os
import plotly.express as px
import plotly.figure_factory as ff
import datetime
import plotly.io as pio
import plotly.graph_objs as go

pio.kaleido.scope.mathjax = None
import math
# import pylab
from matplotlib.colors import LinearSegmentedColormap
from PIL import ImageColor


def distribute_glacier(list_of_samples):
    list_of_glaciers = {}
    for glacier in ['JAC']:
    #for glacier in [ 'COL', 'Mapple', 'Crane', 'Jorum','DBE','SI', 'JAC']:
        list_of_glaciers[glacier] = [sample for sample in list_of_samples if glacier in sample]
    return list_of_glaciers


def create_dict(list_of_samples):
    list_dict = []
    for sample in list_of_samples:
        sample_split = sample.split('_')
        finish_date = datetime.datetime.fromisoformat(sample_split[1]) + datetime.timedelta(days=50)
        sample_dict = {
            'Glacier': sample_split[0],
            'Start': sample_split[1],
            'Finish': str(finish_date),
            'Satellite:': sample_split[2]
        }
        list_dict.append(sample_dict)
    return list_dict


if __name__ == '__main__':
    train_dir = '/home/ho11laqe/PycharmProjects/data_raw/fronts/train/'
    test_dir = '/home/ho11laqe/PycharmProjects/data_raw/fronts/test/'

    list_of_train_samples = os.listdir(train_dir)
    list_of_test_samples = os.listdir(test_dir)
    list_of_samples = list_of_train_samples + list_of_test_samples
    list_of_glaciers = distribute_glacier(list_of_samples)
    list_dict = create_dict(list_of_samples)

    # define color map
    colormap = px.colors.sequential.Reds[-1::-1]
    for glacier in list_of_glaciers:
        print(glacier)
        list_of_glaciers[glacier].sort()


        if glacier in ['COL', 'Mapple']:
            data_directory = test_dir
            image_directory = '/home/ho11laqe/PycharmProjects/data_raw/sar_images/test/'
        else:
            data_directory = train_dir
            image_directory = '/home/ho11laqe/PycharmProjects/data_raw/sar_images/train/'


        # define SAR blackground image
        if glacier == 'COL':
            canvas = cv2.imread(image_directory + 'COL_2011-11-13_TDX_7_1_092.png')
            shape = canvas.shape

        elif glacier == 'JAC':
            canvas = cv2.imread(image_directory + 'JAC_2009-06-21_TSX_6_1_005.png')
            shape = canvas.shape

        elif glacier == 'Jorum':
            canvas = cv2.imread(image_directory + 'Jorum_2011-09-04_TSX_7_4_034.png')
            shape = canvas.shape

        elif glacier == 'Mapple':
            canvas = cv2.imread(image_directory + 'Mapple_2008-10-13_TSX_7_2_034.png')
            shape = canvas.shape

        elif glacier == 'SI':
            canvas = cv2.imread(image_directory + 'SI_2013-08-14_TSX_7_1_125.png')

        elif glacier == 'Crane':
            canvas = cv2.imread(image_directory + 'Crane_2008-10-13_TSX_7_3_034.png')

        elif glacier == 'DBE':
            canvas = cv2.imread(image_directory + 'DBE_2008-03-30_TSX_7_3_049.png')

        else:
            print('No image for background')
            exit()

        number_images = len(list_of_glaciers[glacier])
        kernel = np.ones((3, 3), np.uint8)

        # iterate over all fronts of one glacier
        for i, image_name in enumerate(list_of_glaciers[glacier]):
            front = cv2.imread(data_directory + image_name)

            # if front label has to be resized to fit background image
            # the front is not dilated.
            if front.shape != canvas.shape:
                front = cv2.resize(front, (shape[1], shape[0]))

            else:
                front = cv2.dilate(front, kernel)

            # color interpolation based on position in dataset
            # TODO based on actual date
            index = (1 - i / number_images) * (len(colormap) - 1)
            up = math.ceil(index)
            down = up - 1
            color_up = np.array(ImageColor.getcolor(colormap[up], 'RGB'))
            color_down = np.array(ImageColor.getcolor(colormap[down], 'RGB'))
            dif = up - down
            color = color_up * (1 - dif) + color_down * dif

            # draw front on canvas
            non_zeros = np.nonzero(front)
            canvas[non_zeros[:2]] = np.uint([color for _ in non_zeros[0]])

        #scale reference for fontsize
        ref_x = 15000 / 7

        if glacier == 'COL':
            image = canvas[750:, 200:2800]
            new_shape = image.shape
            res = 7
            scale = new_shape[1] / ref_x
            fig = px.imshow(image, height=new_shape[0]- int(80 * scale), width=new_shape[1])
            legend = dict(thickness=int(50 * scale), tickvals=[-4.4, 4.4],
                          ticktext=['2011<br>(+0.8°C)', '2020<br>(+1.2°C)'],
                          outlinewidth=0)

        elif glacier == 'Mapple':
            image = canvas
            new_shape = image.shape
            res = 7
            scale = new_shape[1] / ref_x
            fig = px.imshow(image, height=new_shape[0] - int(150 * scale), width=new_shape[1])
            legend = dict(thickness=int(50 * scale), tickvals=[-4.8, 4.8], ticktext=['2006', '2020 '],
                          outlinewidth=0)

        elif glacier == 'Crane':
            image = canvas[:2500,:]
            new_shape = image.shape
            res = 7
            scale = new_shape[1] / ref_x
            fig = px.imshow(image, height=new_shape[0] - int(150 * scale), width=new_shape[1])
            legend = dict(thickness=int(50 * scale), tickvals=[-4.8, 4.8], ticktext=['2002', '2014'],
                          outlinewidth=0)

        elif glacier == 'Jorum':
            image = canvas#[200:1600, 1500:]
            new_shape = image.shape
            res = 7
            scale = new_shape[1] / ref_x
            fig = px.imshow(image, height=new_shape[0] - int(240 * scale), width=new_shape[1])
            legend = dict(thickness=int(50 * scale), tickvals=[-4.8, 4.8], ticktext=['2003', '2020'],
                          outlinewidth=0)

        elif glacier == 'DBE':
            image = canvas[700:, 750:]
            new_shape = image.shape
            res = 7
            scale = new_shape[1] / ref_x
            fig = px.imshow(image, height=new_shape[0] - int(150 * scale), width=new_shape[1])
            legend = dict(thickness=int(50 * scale), tickvals=[-4.7, 4.7], ticktext=['1995', '2014'],
                          outlinewidth=0)

        elif glacier == 'SI':
            image = canvas
            new_shape = image.shape
            res = 7
            scale = new_shape[0] / ref_x
            fig = px.imshow(image, height=new_shape[0] - int(240 * scale), width=new_shape[1])
            legend = dict(thickness=int(50 * scale), tickvals=[-4.8, 4.8], ticktext=['1995', '2014'],
                          outlinewidth=0)

        elif glacier == 'JAC':
            image = canvas[:, :]
            new_shape = image.shape
            res = 6
            scale = new_shape[1] / ref_x
            fig = px.imshow(image, height=new_shape[0] - int(340 * scale), width=new_shape[1])
            legend = dict(thickness=int(50 * scale), tickvals=[-4.6, 4.7],
                          ticktext=['2009<br>(+0.7°C)', '2015<br>(+0.9°C)'],
                          outlinewidth=0)
        else:
            fig = px.imshow(canvas)
            res = 7
            scale = 1

        colorbar_trace = go.Scatter(x=[None],
                                    y=[None],
                                    mode='markers',
                                    marker=dict(
                                        colorscale=colormap[::-1],
                                        showscale=True,
                                        cmin=-5,
                                        cmax=5,
                                        colorbar=legend
                                    ),
                                    hoverinfo='none'
                                    )
        fig.update_layout(yaxis=dict(tickmode='array',
                                     tickvals=[0, 5000 / res, 10000 / res, 15000 / res, 20000 / res, 25000 / res],
                                     ticktext=[0, 5, 10, 15, 20, 25],
                                     title='km'))
        fig.update_layout(xaxis=dict(tickmode='array',
                                     tickvals=[0, 5000 / res, 10000 / res, 15000 / res, 20000 / res, 25000 / res],
                                     ticktext=[0, 5, 10, 15, 20, 25],
                                     title='km'))

        fig.update_xaxes(tickfont=dict(size=int(40 * scale)))
        fig.update_yaxes(tickfont=dict(size=int(40 * scale)))
        fig.update_layout(font=dict(size=int(60 * scale), family="Computer Modern"))
        fig.update_coloraxes(colorbar_x=0)
        fig['layout']['xaxis']['title']['font']['size'] = int(60 * scale)
        fig['layout']['yaxis']['title']['font']['size'] = int(60 * scale)

        fig['layout']['showlegend'] = False
        fig.add_trace(colorbar_trace)
        fig.write_image('output/' + glacier + "_front_change.pdf", format='pdf')
        # fig.show()