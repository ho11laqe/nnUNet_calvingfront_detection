import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import os
pio.kaleido.scope.mathjax = None
import json


if __name__ == '__main__':
    experiments =['Task501_Glacier_front',
                  'Task502_Glacier_zone',
                  'Task503_Glacier_mtl_early',
                  'Task503_Glacier_mtl_late',
                  'Task505_Glacier_mtl_boundary',
                  'Task500_Glacier_zonefronts']
    data_dir = '/home/ho11laqe/Desktop/nnUNet_results/Final_Eval/'

    nofront = {}
    nozone = {}
    for experiment in experiments:
        no_front_exp_front = []
        no_front_exp_zone = []
        #nofront[experiment] = {'Front': [], 'Zone': []}
        for fold in range(5):
            results_json_path = os.path.join(data_dir, experiment, 'fold_'+str(fold), 'pngs', 'eval_results.json')
            if not os.path.exists(results_json_path):
                results_json_path = os.path.join(data_dir, experiment, 'fold_' + str(fold), 'eval_results.json')

            with open(results_json_path, 'r') as f:
                result = json.load(f)
            if 'Front_Delineation' in result.keys():
                #no_front_exp_front.append(result['Front_Delineation']['Result_all']['Number_no_front'])
                no_front_exp_front.append(result['Front_Delineation']['Result_all']['mean'])
            else:
                no_front_exp_front.append(0)
            if 'Zone_Delineation' in result.keys():
                no_front_exp_zone.append(result['Zone_Delineation']['Result_all']['mean'])
            else:
                no_front_exp_zone.append(0)

        #nofront[experiment]['Front'] = no_front_exp_front
        #nofront[experiment]['Zone'] = no_front_exp_zone
        nofront[experiment] = no_front_exp_front
        nozone[experiment] = no_front_exp_zone

    box_width = 0.8
    fig = px.box(None, points="all", template="plotly_white", width=1200, height=500)

    fig.add_trace(go.Box(y=nofront['Task501_Glacier_front'], name='Front<br>STL', width=box_width,
                            marker_color='CadetBlue',  pointpos=0, boxpoints='all', boxmean=True))

    fig.add_trace(go.Box(y=nofront['Task503_Glacier_mtl_early'], name='Early Front <br>MTL', width=box_width,
                            marker_color='YellowGreen',  pointpos=0, boxpoints='all', boxmean=True))
    fig.add_trace(go.Box(y=nofront['Task503_Glacier_mtl_late'], name='Late Front <br>MTL', width=box_width,
                            marker_color='#e1e400 ',  pointpos=0, boxpoints='all', boxmean=True))
    fig.add_trace(go.Box(y=nofront['Task505_Glacier_mtl_boundary'], name='Boundary<br> Front MTL', width=box_width,
                            marker_color='gold', pointpos=0, boxpoints='all', boxmean=True))
    fig.add_trace(go.Box(y=nofront['Task500_Glacier_zonefronts'], name='Fused Labels <br> Front', width=box_width,
               marker_color='orange', pointpos=0, boxpoints='all', boxmean=True))

    fig.add_trace(go.Box(y=nozone['Task502_Glacier_zone'], name='Zone<br> STL', width=box_width,
                            marker_color='LightBlue ',  pointpos=0, boxpoints='all', boxmean=True))
    fig.add_trace(go.Box(y=nozone['Task503_Glacier_mtl_early'], name='Early Zone <br>MTL', width=box_width,
                            marker_color='YellowGreen', pointpos=0, boxpoints='all', boxmean=True,))
    fig.add_trace(go.Box(y=nozone['Task503_Glacier_mtl_late'], name='Late Zone<br> MTL', width=box_width,
                            marker_color='#e1e400',  pointpos=0, boxpoints='all', boxmean=True))
    fig.add_trace(go.Box(y=nozone['Task505_Glacier_mtl_boundary'], name='Boundary <br>Zone MTL', width=box_width,
                            marker_color='gold',  pointpos=0, boxpoints='all', boxmean=True))
    fig.add_trace(go.Box(y=nozone['Task500_Glacier_zonefronts'], name='Fused Labels <br> Zone', width=box_width,
               marker_color='orange', pointpos=0, boxpoints='all', boxmean=True))

    fig.update_layout(showlegend=False, font=dict(family="Times New Roma", size=18))
    fig.update_yaxes(title='Front delineation error (m)')
    # fig.show()
    fig.write_image("output/results.pdf", format='pdf')