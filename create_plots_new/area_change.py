import cv2
import os
import plotly.express as px
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.io as pio
pio.kaleido.scope.mathjax = None


def distribute_glacier(list_of_samples):
    list_of_glaciers = {}
    for glacier in [ 'COL', 'Mapple', 'Crane', 'Jorum','DBE','SI', 'JAC']:
        list_of_glaciers[glacier] = [sample for sample in list_of_samples if glacier in sample]
    return list_of_glaciers


if __name__ == '__main__':
    generate_data = True
    if generate_data:
        # directories with zone label
        train_dir = '/home/ho11laqe/PycharmProjects/data_raw/zones/train'
        test_dir = '/home/ho11laqe/PycharmProjects/data_raw/zones/test'

        list_of_train_samples = []
        for sample in os.listdir(train_dir):
            list_of_train_samples.append(os.path.join(train_dir, sample))

        list_of_test_samples = []
        for sample in os.listdir(test_dir):
            list_of_test_samples.append(os.path.join(test_dir, sample))

        list_of_samples = list_of_train_samples + list_of_test_samples

        list_of_glacier = distribute_glacier(list_of_samples)

        fig = make_subplots(rows=len(list_of_glacier.keys()), cols=1)
        nan = []
        rock = []
        ice = []
        ocean = []
        date = []
        glacier_name = []
        for i, glacier in enumerate(list_of_glacier.keys()):

            for sample in list_of_glacier[glacier]:
                print(sample)
                seg_mask = cv2.imread(sample, cv2.IMREAD_GRAYSCALE)
                all_pixel = seg_mask.shape[0] * seg_mask.shape[1]
                nan.append(np.count_nonzero(seg_mask == 0) / all_pixel * 100)
                rock.append(np.count_nonzero(seg_mask == 64) / all_pixel * 100)
                ice.append(np.count_nonzero(seg_mask == 127) / all_pixel * 100)
                ocean.append(np.count_nonzero(seg_mask == 254) / all_pixel * 100)

                sample_split = sample.split('_')
                date.append(sample_split[-6])
                glacier_name.append(glacier)

        df = pd.DataFrame(dict(Shadow=nan, Rock=rock, Glacier=ice, Ocean=ocean, date=date, glacier_name=glacier_name))
        df.to_csv('output/area.csv')

    else:
        df = pd.read_csv('output/area.csv')

    df = df.drop_duplicates(subset=['date', 'glacier_name'])
    area_plot = px.area(df,
                        x="date",
                        y=["Rock", "Shadow", "Glacier", "Ocean"],
                        color_discrete_map={"Shadow": 'black', "Ocean": 'blue', "Glacier": "aliceblue", "Rock": "gray"},
                        template="plotly_white",
                        height=700,
                        width =600,
                        facet_row='glacier_name',
                        category_orders={'glacier': [ 'COL', 'Mapple', 'Crane', 'Jorum','DBE','SI', 'JAC']}
                        )
    area_plot.update_yaxes(type='linear', range=[0, 100], ticksuffix='%', title='area', side='right')
    area_plot.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1], textangle=0, x=0, xanchor='right'))
    area_plot.update_layout(legend=dict(title='Area:',
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1,
                                        font=dict(size=12)),
                            margin=dict(l=70, r=0, t=0, b=0)
                            )
    area_plot.for_each_yaxis(lambda a: a.update(title=''))
    area_plot.update_xaxes(title=' ',tickfont=dict(size=12))
    area_plot.update_layout(font=dict(family="Times New Roma", size=10, ))
    area_plot.update_annotations(font=dict(size=12))
    area_plot.write_image("output/area.pdf", format='pdf')
    # fig.show()
