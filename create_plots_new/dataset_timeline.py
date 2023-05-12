import numpy as cv
import os
import plotly.express as px
import plotly.figure_factory as ff
import datetime
import plotly.io as pio

pio.kaleido.scope.mathjax = None


def distribute_glacier(list_of_samples):
    list_of_glaciers = {}
    for glacier in ['COL', 'Mapple', 'Crane', 'Jorum', 'DBE', 'SI', 'JAC']:
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
    list_of_train_samples = os.listdir('../../data_raw/fronts/train')
    list_of_test_samples = os.listdir('../../data_raw/fronts/test')
    list_of_samples = list_of_train_samples + list_of_test_samples
    list_of_glaciers = distribute_glacier(list_of_samples)
    list_dict = create_dict(list_of_samples)

    fig = px.timeline(list_dict, x_start='Start', x_end='Finish', color="Satellite:", y='Glacier',
                      color_discrete_sequence=px.colors.qualitative.G10, template="plotly_white",
                      height=300, category_orders={'Glacier': ['COL', 'Mapple', 'Crane', 'Jorum', 'DBE', 'SI', 'JAC'],
                                                   'Satellite:': ['ERS', 'RSAT', 'ENVISAT', 'PALSAR', 'TSX', 'TDX',
                                                                  'S1']})
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
        margin=dict(l=0, r=0, t=0, b=0), )
    fig.update_layout(
        font=dict(family="Computer Modern", size=14))
    fig.write_image("output/dataset_timeline.pdf", format='pdf')
    # fig.show()
