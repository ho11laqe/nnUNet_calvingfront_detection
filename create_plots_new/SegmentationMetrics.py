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
                  'Task500_Glacier_zonefronts_nodeep',
                  'Task500_Glacier_zonefronts'
                  ]
    data_dir = '/home/ho11laqe/Desktop/nnUNet_results/Final_Eval/'

    for metric in ['Precision', 'Recall', 'F1', 'IoU']:

        zone_metric = {}
        for experiment in experiments:

            zone_metric_exp = []
            #nofront[experiment] = {'Front': [], 'Zone': []}
            for fold in range(5):
                # load json file with results
                results_json_path = os.path.join(data_dir, experiment, 'fold_'+str(fold), 'pngs', 'eval_results.json')
                if not os.path.exists(results_json_path):
                    results_json_path = os.path.join(data_dir, experiment, 'fold_' + str(fold), 'eval_results.json')

                with open(results_json_path, 'r') as f:
                    result = json.load(f)

                if 'Zone_Segmentation' in result.keys():
                    avg_metric = 'Average_'+metric
                    if metric == 'F1':
                        avg_metric = 'Average_' + metric + ' Score'
                    zone_metric_exp.append(result['Zone_Segmentation']['Zone_'+metric][avg_metric])
                else:
                    zone_metric_exp.append(0)

            zone_metric[experiment] = zone_metric_exp

        box_width = 0.8
        fig = px.box(None, points="all", template="plotly_white", width=700, height=500)

        fig.add_trace(go.Box(y=zone_metric['Task502_Glacier_zone'], name='Zone<br> STL', width=box_width,
                                line_color='LightBlue ',  pointpos=0, boxpoints='all', boxmean=True))
        fig.add_trace(go.Box(y=zone_metric['Task503_Glacier_mtl_early'], name='Early Zone <br>MTL', width=box_width,
                                line_color='YellowGreen', pointpos=0, boxpoints='all', boxmean=True,))
        fig.add_trace(go.Box(y=zone_metric['Task503_Glacier_mtl_late'], name='Late Zone<br> MTL', width=box_width,
                                line_color='#e1e400',  pointpos=0, boxpoints='all', boxmean=True))
        fig.add_trace(go.Box(y=zone_metric['Task505_Glacier_mtl_boundary'], name='Boundary<br>Zone MTL', width=box_width,
                                line_color='gold',  pointpos=0, boxpoints='all', boxmean=True))
        fig.add_trace(go.Box(y=zone_metric['Task500_Glacier_zonefronts'], name='Fused Labels<br>Front', width=box_width,
                                line_color='orange', pointpos=0, boxpoints='all', boxmean=True))

        fig.update_layout(showlegend=False, font=dict(family="Times New Roman", size=18))
        fig.update_yaxes(title=metric)
        # fig.show()
        fig.write_image('output/'+metric+".pdf", format='pdf')