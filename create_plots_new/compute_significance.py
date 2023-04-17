import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.kaleido.scope.mathjax = None
import os
import json


if __name__ == '__main__':

    experiments = ['Task501_Glacier_front',
                   'Task502_Glacier_zone',
                   'Task503_Glacier_mtl_early',
                   'Task503_Glacier_mtl_late',
                   'Task505_Glacier_mtl_boundary',
                   'Task500_Glacier_zonefronts']
    data_dir = '/home/ho11laqe/Desktop/nnUNet_results/Final_Eval/'

    zone_mean = {}
    front_mean = {}
    for experiment in experiments:
        print(experiment)
        zone_mean_exp = []
        front_mean_exp = []
        # nofront[experiment] = {'Front': [], 'Zone': []}
        for fold in range(5):
            # load json file with results
            results_json_path = os.path.join(data_dir, experiment, 'fold_' + str(fold), 'pngs',
                                             'eval_results.json')
            if not os.path.exists(results_json_path):
                results_json_path = os.path.join(data_dir, experiment, 'fold_' + str(fold), 'eval_results.json')

            with open(results_json_path, 'r') as f:
                result = json.load(f)

            if 'Front_Delineation' in result.keys():

                front_mean_exp.append(result['Front_Delineation']['Result_all']['mean'])
            else:
                front_mean_exp.append(0)

            if 'Zone_Delineation' in result.keys():
                zone_mean_exp.append(result['Zone_Delineation']['Result_all']['mean'])
            else:
                zone_mean_exp.append(0)

        print(np.mean(zone_mean_exp), np.std(zone_mean_exp))
        print(np.mean(front_mean_exp), np.std(front_mean_exp))
        zone_mean[experiment] = zone_mean_exp
        front_mean[experiment] = front_mean_exp

    for exp1 in experiments:
        for exp2 in experiments:
            # FRONT
            mean1 = np.mean(front_mean[exp1])
            var1 = np.var (front_mean[exp1])
            mean2 = np.mean(front_mean[exp2])
            var2 = np.var(front_mean[exp2])

            T_front = abs(mean1 - mean2) / np.sqrt((var1 / 5) + (var2 / 5))
            print(exp1 + '<>' +exp2)
            print('Tfront:'+ str(T_front))

            # Zone
            mean1 = np.mean(zone_mean[exp1])
            var1 = np.var(zone_mean[exp1])
            mean2 = np.mean(zone_mean[exp2])
            var2 = np.var(zone_mean[exp2])

            T_zone = abs(mean1 - mean2) / np.sqrt((var1 / 5) + (var2 / 5))
            print('Tzone:' + str(T_zone))
            print('')
        """
        box_width = 0.8
        fig = px.box(None, points="all", template="plotly_white", width=600, height=500)

        fig.add_trace(go.Box(y=zone_mean['Task502_Glacier_zone'], name='Zone<br> STL', width=box_width,
                             line_color='black', fillcolor='LightBlue ', pointpos=0, boxpoints='all', boxmean=True))
        fig.add_trace(go.Box(y=zone_mean['Task503_Glacier_mtl_early'], name='Early Zone <br>MTL', width=box_width,
                             line_color='black', fillcolor='YellowGreen', pointpos=0, boxpoints='all',
                             boxmean=True, ))
        fig.add_trace(go.Box(y=zone_mean['Task503_Glacier_mtl_late'], name='Late Zone<br> MTL', width=box_width,
                             line_color='black', fillcolor='#e1e400', pointpos=0, boxpoints='all', boxmean=True))
        fig.add_trace(
            go.Box(y=zone_mean['Task505_Glacier_mtl_boundary'], name='Boundary <br>Zone MTL', width=box_width,
                   line_color='black', fillcolor='gold', pointpos=0, boxpoints='all', boxmean=True))

        fig.update_layout(showlegend=False, font=dict(family="Times New Roman", size=18))
        fig.update_yaxes(title='Front mean')
        # fig.show()
        fig.write_image('Front mean' + ".pdf", format='pdf')
        """