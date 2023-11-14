import cv2
import numpy as np
import os
import plotly.express as px
import plotly.figure_factory as ff
import datetime
import plotly.io as pio
import plotly.graph_objs as go
import kaleido
pio.kaleido.scope.mathjax = None
import math
from PIL import ImageColor


def distribute_glacier(list_of_samples):
    list_of_glaciers = {}
    for glacier in ['COL']:
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

def get_interval(list_of_samples):
    dates = []
    for sample in list_of_samples:
        sample_split = sample.split('_')
        date = datetime.datetime.fromisoformat(sample_split[1])
        dates.append(date)
    return min(dates), max(dates)

if __name__ == '__main__':

    train_dir = '../../data_raw/sar_images/train/'
    test_dir = '../../data_raw/sar_images/test/'

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
            data_directory = '../../data_raw/fronts/test/'
            image_directory = '../../data_raw/sar_images/test/'
        else:
            data_directory = '../../data_raw/fronts/train/'
            image_directory = '../../data_raw/sar_images/train/'

        # define SAR blackground image
        if glacier == 'COL':
            canvas = cv2.imread(image_directory + 'COL_2011-11-13_TDX_7_1_092.png')
            shape = canvas.shape
            shape = np.divide(shape, 1).astype('int')

        elif glacier == 'JAC':
            canvas = cv2.imread(image_directory + 'JAC_2009-06-21_TSX_6_1_005.png')
            shape = canvas.shape
            shape = np.divide(shape, 2).astype('int')

        elif glacier == 'Jorum':
            canvas = cv2.imread(image_directory + 'Jorum_2011-09-04_TSX_7_4_034.png')
            shape = canvas.shape

        elif glacier == 'Mapple':
            canvas = cv2.imread(image_directory + 'Mapple_2008-10-13_TSX_7_2_034.png')
            shape = canvas.shape

        elif glacier == 'SI':
            canvas = cv2.imread(image_directory + 'SI_2013-08-14_TSX_7_1_125.png')
            shape = canvas.shape

        elif glacier == 'Crane':
            canvas = cv2.imread(image_directory + 'Crane_2008-10-13_TSX_7_3_034.png')
            shape = canvas.shape
            shape = np.divide(shape, 2).astype('int')

        elif glacier == 'DBE':
            canvas = cv2.imread(image_directory + 'DBE_2008-03-30_TSX_7_3_049.png')
            shape = canvas.shape

        else:
            print('No image for background')
            exit()

        start, end = get_interval(list_of_glaciers[glacier])



        number_images = len(list_of_glaciers[glacier])
        kernel = np.ones((3, 3), np.uint8)


        # iterate over all fronts of one glacier
        fronts = {"front":[], "color":[], "index":[], "date":[]}
        for i, image_name in enumerate(list_of_glaciers[glacier][45:46]):
            sample_split = image_name.split('_')
            date = datetime.datetime.fromisoformat(sample_split[1])
            image_name = image_name[:-len('.png')]
            print(image_name)
            front = cv2.imread(data_directory + image_name + '_front.png')
            canvas = cv2.imread(image_directory + image_name + '.png')


            # if front label has to be resized to fit background image
            # the front is not dilated.

            canvas = cv2.resize(canvas, (shape[1], shape[0]))
            front = cv2.resize(front, (shape[1], shape[0]))
            if int(sample_split[3])<=7:
                front = cv2.dilate(front, kernel,iterations=2)

            # color interpolation based on position in dataset
            interaval_len = end.timestamp() - start.timestamp()

            index = ((date.timestamp() - start.timestamp()) / interaval_len) * (len(colormap) - 1)
            down = math.floor(index)
            up = down + 1
            if up == len(colormap): up=down
            print(index)
            color_up = np.array(ImageColor.getcolor(colormap[len(colormap)-up-1], 'RGB'))
            color_down = np.array(ImageColor.getcolor(colormap[len(colormap)-down-1], 'RGB'))
            dif = index - down
            color_new = color_up * (1 - dif) + color_down * dif

            fronts['front'].append(front)
            fronts['color'].append(color_new)

            # draw front on canvas
            for j, front in enumerate(fronts['front']):
                non_zeros = np.nonzero(front)
                color = fronts['color'][j]
                canvas[non_zeros[:2]] = np.uint([color for _ in non_zeros[0]])


            #scale reference for fontsize
            ref_x = 15000 / 7
            index_legend = ((date.timestamp() - start.timestamp()) / interaval_len) * 9.8 - 4.9
            fronts['index'].append(index_legend)
            if glacier == 'COL':
                image = canvas[750:, 200:2800]
                new_shape = image.shape
                res = 7
                scale = new_shape[1] / ref_x
                fig = px.imshow(image, height=new_shape[0], width=new_shape[1])
                #fig.update_layout(title_text='Columbia Glacier (Alaska)', title_x=0.48, title_y=0.99,)
                tickvals = [-4.9] + fronts['index'] + [4.9]
                ticktext = ['2011'] +[""]*(len(fronts['index'])-1)+[str(date.year)] + ['2020']
                legend = dict(
                    thickness=int(50 * scale),
                    tickvals=tickvals,
                    ticktext=ticktext,
                    outlinewidth=0, ticks="inside", ticklen=50 * scale, tickwidth=2)

            elif glacier == 'Mapple':
                image = canvas
                new_shape = image.shape
                res = 7
                scale = new_shape[1] / ref_x
                fig = px.imshow(image, height=new_shape[0] - int(80 * scale), width=new_shape[1])
                legend = dict(thickness=int(50 * scale), tickvals=[-4.8, 4.8], ticktext=['2006', '2020 '],
                              outlinewidth=0)

            elif glacier == 'Crane':
                image = canvas[:2500,:]
                new_shape = image.shape
                res = 7
                scale = new_shape[1] / ref_x
                fig = px.imshow(image, height=new_shape[0] - int(150 * scale), width=new_shape[1])
                fig.update_layout(title_text=glacier+' Glacier', title_x=0.45, title_y=0.97)
                legend = dict(
                    thickness=int(50 * scale), tickvals=[-5, index_legend, 5], ticktext=['2002',str(date.year), '2014'],
                              outlinewidth=0, ticks="inside", ticklen=50*scale, tickwidth=5)

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
                res = 12
                scale = 0.7
                fig = px.imshow(image, height=new_shape[0] - int(340 * scale), width=new_shape[1])
                fig.update_layout(title_text='Jakobshavn Glacier', title_x=0.48, title_y=0.99, )
                tickvals = [-4.9] + fronts['index'] + [4.9]
                ticktext = ['2009-04'] + [""] * (len(fronts['index']) - 1) + [str(date.year) + "-%02i" % date.month] + [
                    '2015-03']
                legend = dict(
                    thickness=int(50 * scale),
                    tickvals=tickvals,
                    ticktext=ticktext,
                    outlinewidth=0, ticks="inside", ticklen=50 * scale, tickwidth=2)

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
            if not os.path.exists('output/'+glacier+'/'):
                os.makedirs('output/'+glacier+'/')
            fig.write_image('output/' + glacier + "/frame_%04i.png"%i, format='png')
            #fig.show()

        import os
        from PIL import Image

        # get file names in correct order
        dir = 'output/'+glacier+'/'
        files = os.listdir(dir)
        files.sort()

        # read all frames
        frames = []
        for image in files[::]:
            if image.endswith('.png'):
                frames.append(Image.open(dir + image))

        # append all frames to a gif
        if not os.path.exists('plots'):
            os.mkdir('plots')

        frame_one = frames[0]
        frame_one.save('output/'+glacier+'.gif', format="GIF", append_images=frames, save_all=True, duration=300, loop=1)