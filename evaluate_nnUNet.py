import math
from argparse import ArgumentParser
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from data_processing.data_postprocessing import postprocess_zone_segmenation, postprocess_front_segmenation, \
    extract_front_from_zones
import torch.nn as nn
# from segmentation_models_pytorch.losses.dice import DiceLoss
# from PIL import Image
# from models.front_segmentation_model import DistanceMapBCE
import re
from pathlib import Path
import cv2
import scipy.stats as st
from scipy.spatial import distance
import skimage
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import json
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import os

pio.kaleido.scope.mathjax = None


def front_error(prediction, label):
    """
    prediction: mask of the front prediction (WxH)
    label: mask of the front label (WxH)

    returns the mean distance of the two fronts
    """
    front_is_present_flag = True
    polyline_pred = np.nonzero(prediction)
    polyline_label = np.nonzero(label)

    # Generate Nx2 matrix of pixels that represent the front
    pred_coords = np.array(list(zip(polyline_pred[0], polyline_pred[1])))
    mask_coords = np.array(list(zip(polyline_label[0], polyline_label[1])))

    # Return NaN if front is not detected in either pred or mask
    if pred_coords.shape[0] == 0 or mask_coords.shape[0] == 0:
        front_is_present_flag = False
        return front_is_present_flag, np.nan, np.nan, np.nan

    # Generate the pairwise distances between each point and the closest point in the other array

    distances1 = distance.cdist(pred_coords, mask_coords).min(axis=1)

    distances2 = distance.cdist(mask_coords, pred_coords).min(axis=1)
    distances = np.concatenate((distances1, distances2))

    # Calculate the average distance between each point and the closest point in the other array
    mean_distance = np.mean(distances)
    median_distance = np.median(distances)
    return front_is_present_flag, mean_distance, median_distance, distances


def multi_class_metric(metric_function, complete_predicted_mask, complete_target):
    metrics = []
    metric_na, metric_stone, metric_glacier, metric_ocean = metric_function(np.ndarray.flatten(complete_target),
                                                                            np.ndarray.flatten(complete_predicted_mask),
                                                                            average=None)
    metric_macro_average = (metric_na + metric_stone + metric_glacier + metric_ocean) / 4
    metrics.append(metric_macro_average)
    metrics.append(metric_na)
    metrics.append(metric_stone)
    metrics.append(metric_glacier)
    metrics.append(metric_ocean)
    return metrics


def get_matching_out_of_folder(file_name, folder):
    files = os.listdir(folder)
    matching_files = [a for a in files if
                      re.match(pattern=os.path.split(file_name)[1][:-4], string=os.path.split(a)[1])]
    if len(matching_files) > 1:
        print("Something went wrong!")
        print(f"targets_matching: {matching_files}")
    if len(matching_files) < 1:
        print("Something went wrong! No matches found")
    return matching_files[0]


def turn_colors_to_class_labels_zones(mask):
    mask_class_labels = np.copy(mask)
    mask_class_labels[mask == 0] = 0
    mask_class_labels[mask == 64] = 1
    mask_class_labels[mask == 127] = 2
    mask_class_labels[mask == 254] = 3
    return mask_class_labels


def turn_colors_to_class_labels_front(mask):
    mask_class_labels = np.copy(mask)
    mask_class_labels[mask == 0] = 0
    mask_class_labels[mask == 255] = 1
    return mask_class_labels


def print_zone_metrics(metric_name, list_of_metrics):
    metrics = [metric for [metric, _, _, _, _] in list_of_metrics if not np.isnan(metric)]
    metrics_na = [metric_na for [_, metric_na, _, _, _] in list_of_metrics if not np.isnan(metric_na)]
    metrics_stone = [metric_stone for [_, _, metric_stone, _, _] in list_of_metrics if not np.isnan(metric_stone)]
    metrics_glacier = [metric_glacier for [_, _, _, metric_glacier, _] in list_of_metrics if
                       not np.isnan(metric_glacier)]
    metrics_ocean = [metric_ocean for [_, _, _, _, metric_ocean] in list_of_metrics if not np.isnan(metric_ocean)]
    result = {}
    print(f"Average {metric_name}: {sum(metrics) / len(metrics)}")
    result[f'Average_{metric_name}'] = sum(metrics) / len(metrics)
    print(f"Average {metric_name} NA Area: {sum(metrics_na) / len(metrics_na)}")
    result[f'Average_{metric_name}_NA_Area'] = sum(metrics_na) / len(metrics_na)
    print(f"Average {metric_name} Stone: {sum(metrics_stone) / len(metrics_stone)}")
    result[f"Average_{metric_name}_Stone"] = sum(metrics_stone) / len(metrics_stone)
    print(f"Average {metric_name} Glacier: {sum(metrics_glacier) / len(metrics_glacier)}")
    result[f"Average_{metric_name}_Glacier"] = sum(metrics_glacier) / len(metrics_glacier)
    print(f"Average {metric_name} Ocean and Ice Melange: {sum(metrics_ocean) / len(metrics_ocean)}")
    result[f"Average_{metric_name}_Ocean_and_Ice_Melange"] = sum(metrics_ocean) / len(metrics_ocean)

    return result


def print_front_metric(name, metric):
    result = {}
    print(f"Average {name}: {sum(metric) / len(metric)}")
    result[f"Average {name}"] = sum(metric) / len(metric)
    return result


def mask_prediction_with_bounding_box(post_complete_predicted_mask, file_name, bounding_boxes_directory):
    matching_bounding_box_file = get_matching_out_of_folder(file_name, bounding_boxes_directory)
    with open(os.path.join(bounding_boxes_directory, matching_bounding_box_file)) as f:
        coord_file_lines = f.readlines()
    left_upper_corner_x, left_upper_corner_y = [round(float(coord)) for coord in coord_file_lines[1].split(",")]
    left_lower_corner_x, left_lower_corner_y = [round(float(coord)) for coord in coord_file_lines[2].split(",")]
    right_lower_corner_x, right_lower_corner_y = [round(float(coord)) for coord in coord_file_lines[3].split(",")]
    right_upper_corner_x, right_upper_corner_y = [round(float(coord)) for coord in coord_file_lines[4].split(",")]

    # Make sure the Bounding Box coordinates are within the image
    if left_upper_corner_x < 0: left_upper_corner_x = 0
    if left_lower_corner_x < 0: left_lower_corner_x = 0
    if right_upper_corner_x > len(post_complete_predicted_mask[0]): right_upper_corner_x = len(
        post_complete_predicted_mask[0]) - 1
    if right_lower_corner_x > len(post_complete_predicted_mask[0]): right_lower_corner_x = len(
        post_complete_predicted_mask[0]) - 1
    if left_upper_corner_y > len(post_complete_predicted_mask): left_upper_corner_y = len(
        post_complete_predicted_mask) - 1
    if left_lower_corner_y < 0: left_lower_corner_y = 0
    if right_upper_corner_y > len(post_complete_predicted_mask): right_upper_corner_y = len(
        post_complete_predicted_mask) - 1
    if right_lower_corner_y < 0: right_lower_corner_y = 0

    # remember cv2 images have the shape (height, width)
    post_complete_predicted_mask[:right_lower_corner_y, :] = 0.0
    post_complete_predicted_mask[left_upper_corner_y:, :] = 0.0
    post_complete_predicted_mask[:, :left_upper_corner_x] = 0.0
    post_complete_predicted_mask[:, right_lower_corner_x:] = 0.0

    return post_complete_predicted_mask


def post_processing(target_masks, complete_predicted_masks, bounding_boxes_directory, complete_test_directory):
    meter_threshold = 750  # in meter
    print("Post-processing ...\n\n")
    for file_name in complete_predicted_masks:
        prediction_name = file_name
        if file_name.endswith('_zone.png'):
            file_name = file_name[:-len("_zone.png")] + ".png"
        if file_name.endswith('_front.png'):
            file_name = file_name[:-len("front.png")] + ".png"

        print(f"File: {file_name}")
        resolution = int(os.path.split(file_name)[1][:-4].split('_')[-3])
        # pixel_threshold (pixel) * resolution (m/pixel) = meter_threshold (m)
        pixel_threshold = meter_threshold / resolution
        complete_predicted_mask = cv2.imread(os.path.join(complete_test_directory, prediction_name).__str__(),
                                             cv2.IMREAD_GRAYSCALE)

        if target_masks == "zones":
            post_complete_predicted_mask = postprocess_zone_segmenation(complete_predicted_mask)
            post_complete_predicted_mask = extract_front_from_zones(post_complete_predicted_mask, pixel_threshold)
        else:
            complete_predicted_mask_class_labels = turn_colors_to_class_labels_front(complete_predicted_mask)
            post_complete_predicted_mask = postprocess_front_segmenation(complete_predicted_mask_class_labels,
                                                                         pixel_threshold)
            post_complete_predicted_mask = post_complete_predicted_mask * 255

        post_complete_predicted_mask = mask_prediction_with_bounding_box(post_complete_predicted_mask, file_name,
                                                                         bounding_boxes_directory)
        cv2.imwrite(os.path.join(complete_postprocessed_test_directory, file_name), post_complete_predicted_mask)


def calculate_front_delineation_metric(complete_postprocessed_test_directory, post_processed_predicted_masks,
                                       directory_of_target_fronts, bounding_boxes_directory):
    list_of_mean_front_errors = []
    list_of_median_front_errors = []
    list_of_all_front_errors = []
    number_of_images_with_no_predicted_front = 0
    results = {}
    for file_name in post_processed_predicted_masks[:]:

        post_processed_predicted_mask = cv2.imread(
            os.path.join(complete_postprocessed_test_directory, file_name).__str__(), cv2.IMREAD_GRAYSCALE)
        matching_target_file = get_matching_out_of_folder(file_name, directory_of_target_fronts)
        target_front = cv2.imread(os.path.join(directory_of_target_fronts, matching_target_file).__str__(),
                                  cv2.IMREAD_GRAYSCALE)
        if file_name.endswith("_front.png"):
            resolution = int(os.path.split(file_name)[1][:-4].split('_')[-4])
        else:
            resolution = int(os.path.split(file_name)[1][:-4].split('_')[-3])

        # images need to be turned into a Tensor [0, ..., n_classes-1]
        post_processed_predicted_mask_class_labels = turn_colors_to_class_labels_front(post_processed_predicted_mask)
        target_front_class_labels = turn_colors_to_class_labels_front(target_front)

        if file_name.endswith('_front.png'):
            post_processed_predicted_mask_class_labels = mask_prediction_with_bounding_box(
                post_processed_predicted_mask_class_labels, file_name[:-len('_front.png')] + '.png',
                bounding_boxes_directory)
            post_processed_predicted_mask_class_labels = skeletonize(post_processed_predicted_mask_class_labels)
        front_is_present_flag, mean_error, median_error, errors = front_error(
            post_processed_predicted_mask_class_labels, target_front_class_labels)

        if not front_is_present_flag:
            number_of_images_with_no_predicted_front += 1
        else:
            list_of_mean_front_errors.append(resolution * mean_error)
            list_of_median_front_errors.append(resolution * median_error)
            list_of_all_front_errors = np.concatenate((list_of_all_front_errors, resolution * errors))
    print(f"Number of images with no predicted front: {number_of_images_with_no_predicted_front}")
    results["Number_no_front"] = number_of_images_with_no_predicted_front
    if number_of_images_with_no_predicted_front >= len(post_processed_predicted_masks):
        print(
            f"Number of images with no predicted front is equal to complete set of images. No metrics can be calculated.")
        return [], {}
    list_of_mean_front_errors_without_nan = [front_error for front_error in list_of_mean_front_errors if
                                             not np.isnan(front_error)]
    list_of_median_front_errors_without_nan = [front_error for front_error in list_of_median_front_errors if
                                               not np.isnan(front_error)]
    print(
        f"Mean-mean distance error (in meters): {sum(list_of_mean_front_errors_without_nan) / len(list_of_mean_front_errors_without_nan)}")
    results["Mean_mean_distance"] = sum(list_of_mean_front_errors_without_nan) / len(
        list_of_mean_front_errors_without_nan)
    print(
        f"Mean-median distance error (in meters): {sum(list_of_median_front_errors_without_nan) / len(list_of_median_front_errors_without_nan)}")
    results["Mean_median_distance"] = sum(list_of_median_front_errors_without_nan) / len(
        list_of_median_front_errors_without_nan)

    list_of_mean_front_errors_without_nan = np.array(list_of_mean_front_errors_without_nan)
    list_of_median_front_errors_without_nan = np.array(list_of_median_front_errors_without_nan)
    print(f"Median-mean distance error (in meters): {np.median(list_of_mean_front_errors_without_nan)}")
    results["Median_mean_distance"] = np.median(list_of_mean_front_errors_without_nan)
    print(f"Median-median distance error (in meters): {np.median(list_of_median_front_errors_without_nan)}")
    results["Median_median_distance"] = np.median(list_of_median_front_errors_without_nan)

    list_of_all_front_errors_without_nan = [front_error for front_error in list_of_all_front_errors if
                                            not np.isnan(front_error)]
    list_of_all_front_errors_without_nan = np.array(list_of_all_front_errors_without_nan)
    confidence_interval = st.norm.interval(alpha=0.95,
                                           loc=np.mean(list_of_all_front_errors_without_nan),
                                           scale=st.sem(list_of_all_front_errors_without_nan))
    mean = np.mean(list_of_all_front_errors_without_nan)
    std = np.std(list_of_all_front_errors_without_nan)
    print(f"Confidence interval: {confidence_interval}, mean: {mean}, standard deviation: {std}")
    results["Confidence_interval"] = confidence_interval
    results['mean'] = mean
    results['standard_deviation'] = std
    return list_of_mean_front_errors_without_nan, results


def calculate_segmentation_metrics(target_mask_modality, complete_predicted_masks, complete_test_directory,
                                   directory_of_complete_targets):
    print("Calculate segmentation metrics ...\n\n")
    list_of_ious = []
    list_of_precisions = []
    list_of_recalls = []
    list_of_f1_scores = []
    result = {}
    for file_name in complete_predicted_masks:
        print(f"File: {file_name}")
        complete_predicted_mask = cv2.imread(os.path.join(complete_test_directory, file_name).__str__(),
                                             cv2.IMREAD_GRAYSCALE)
        matching_target_file = get_matching_out_of_folder(file_name, directory_of_complete_targets)
        complete_target = cv2.imread(os.path.join(directory_of_complete_targets, matching_target_file).__str__(),
                                     cv2.IMREAD_GRAYSCALE)

        if target_mask_modality == "zones":
            # images need to be turned into a Tensor [0, ..., n_classes-1]
            complete_predicted_mask_class_labels = turn_colors_to_class_labels_zones(complete_predicted_mask)
            complete_target_class_labels = turn_colors_to_class_labels_zones(complete_target)
            # Segmentation evaluation metrics
            list_of_ious.append(
                multi_class_metric(jaccard_score, complete_predicted_mask_class_labels, complete_target_class_labels))
            list_of_precisions.append(
                multi_class_metric(precision_score, complete_predicted_mask_class_labels, complete_target_class_labels))
            list_of_recalls.append(
                multi_class_metric(recall_score, complete_predicted_mask_class_labels, complete_target_class_labels))
            list_of_f1_scores.append(
                multi_class_metric(f1_score, complete_predicted_mask_class_labels, complete_target_class_labels))
        else:
            # images need to be turned into a Tensor [0, ..., n_classes-1]
            complete_predicted_mask_class_labels = turn_colors_to_class_labels_front(complete_predicted_mask)
            complete_target_class_labels = turn_colors_to_class_labels_front(complete_target)
            # Segmentation evaluation metrics
            flattened_complete_target_class_labels = np.ndarray.flatten(complete_target_class_labels)
            flattened_complete_predicted_mask_class_labels = np.ndarray.flatten(complete_predicted_mask_class_labels)
            list_of_ious.append(
                jaccard_score(flattened_complete_target_class_labels, flattened_complete_predicted_mask_class_labels))
            list_of_precisions.append(
                precision_score(flattened_complete_target_class_labels, flattened_complete_predicted_mask_class_labels))
            list_of_recalls.append(
                recall_score(flattened_complete_target_class_labels, flattened_complete_predicted_mask_class_labels))
            list_of_f1_scores.append(
                f1_score(flattened_complete_target_class_labels, flattened_complete_predicted_mask_class_labels))

    if target_mask_modality == "zones":
        result_precision = print_zone_metrics("Precision", list_of_precisions)
        result["Zone_Precision"] = result_precision
        result_recal = print_zone_metrics("Recall", list_of_recalls)
        result["Zone_Recall"] = result_recal
        result_f1 = print_zone_metrics("F1 Score", list_of_f1_scores)
        result["Zone_F1"] = result_f1
        result_iou = print_zone_metrics("IoU", list_of_ious)
        result["Zone_IoU"] = result_iou
    else:
        if len(list_of_precisions) > 0:
            result_precsions = print_front_metric("Precision", list_of_precisions)
            result["Front_Precsion"] = result_precsions
        if len(list_of_recalls) > 0:
            result_recall = print_front_metric("Recall", list_of_recalls)
            result["Front_Recall"] = result_recall
        if len(list_of_f1_scores):
            result_f1 = print_front_metric("F1 Score", list_of_f1_scores)
            result["Front_F1"] = result_f1
        if len(list_of_ious) > 0:
            result_iou = print_front_metric("IoU", list_of_ious)
            result["Front_IoU"] = result_iou

    return result


def check_whether_winter_half_year(name):
    split_name = name[:-4].split('_')
    if split_name[0] == "COL" or split_name[0] == "JAC":
        nord_halbkugel = True
    else:  # Jorum, Maple, Crane, SI, DBE
        nord_halbkugel = False
    month = int(split_name[1].split('-')[1])
    if nord_halbkugel:
        if month < 4 or month > 8:
            winter = True
        else:
            winter = False
    else:
        if month < 4 or month > 8:
            winter = False
        else:
            winter = True
    return winter


def front_delineation_metric(modality, complete_postprocessed_test_directory, directory_of_target_fronts,
                             bounding_boxes_directory):
    print("Calculating distance errors ...\n\n")
    if modality == 'front':
        post_processed_predicted_masks = list(
            file for file in os.listdir(complete_postprocessed_test_directory) if file.endswith('_front.png'))

    elif modality == 'zone':
        post_processed_predicted_masks = list(file for file in os.listdir(complete_postprocessed_test_directory))

    print("")
    print("####################################################################")
    print(f"# Results for all images")
    print("####################################################################")
    fig = px.box(None, points="all", template="none", log_x=True, height=300, )
    G10 = px.colors.qualitative.Safe
    width = 0.5
    list_of_mean_front_errors_without_nan, result_all = calculate_front_delineation_metric(
        complete_postprocessed_test_directory, post_processed_predicted_masks, directory_of_target_fronts,
        bounding_boxes_directory)
    np.savetxt(os.path.join(complete_postprocessed_test_directory, os.pardir, "distance_errors.txt"),
               list_of_mean_front_errors_without_nan)
    fig.add_trace(go.Box(x=list_of_mean_front_errors_without_nan, marker_color='orange', boxmean=True, boxpoints='all',
                         name='all', width=width))

    results = {}
    results['Result_all'] = result_all

    # Season subsetting
    for season in ["winter", "summer"]:
        print("")
        print("####################################################################")
        print(f"# Results for only images in {season}")
        print("####################################################################")
        subset_of_predictions = []
        for file_name in post_processed_predicted_masks:
            winter = check_whether_winter_half_year(file_name)
            if (winter and season == "summer") or (not winter and season == "winter"):
                continue
            subset_of_predictions.append(file_name)

        if len(subset_of_predictions) == 0: continue
        all_errors, result_season = calculate_front_delineation_metric(complete_postprocessed_test_directory,
                                                                       subset_of_predictions,
                                                                       directory_of_target_fronts,
                                                                       bounding_boxes_directory)
        if season == 'winter':
            color = G10[0]
        else:
            color = G10[9]
        print(season, np.mean(all_errors), np.std(all_errors))
        fig.add_trace(go.Box(x=all_errors, marker_color=color, boxmean=True, boxpoints='all', name=season, width=width,
                             legendrank=0))

        results[season] = result_season
    fig.update_layout(showlegend=False, font=dict(family="Times New Roma", size=12))
    fig.update_xaxes(title='Mean Distance Error [m]')
    fig.update_layout(yaxis={'categoryorder': 'array', 'categoryarray': ['summer', 'winter', 'all']})
    fig.update_traces(orientation='h')  # horizontal box plots

    # Front Gourmelon
    fig.add_shape(type='line', x0=738, y0=-0.5, x1=738, y1=0.5, line=dict(color='lightgray', ), xref='x', yref='y')

    fig.add_shape(type='line', x0=1054, y0=0.5, x1=1054, y1=1.5, line=dict(color='lightgray', ), xref='x', yref='y')

    fig.add_shape(type='line', x0=887, y0=1.5, x1=887, y1=2.5, line=dict(color='lightgray', ), xref='x', yref='y')

    # Zone Gourmelon
    fig.add_shape(type='line', x0=732, y0=-0.5, x1=732, y1=0.5, line=dict(color='gray', ), xref='x', yref='y')

    fig.add_shape(type='line', x0=776, y0=0.5, x1=776, y1=1.5, line=dict(color='gray', ), xref='x', yref='y')

    fig.add_shape(type='line', x0=753, y0=1.5, x1=753, y1=2.5, line=dict(color='gray', ), xref='x', yref='y')

    fig.add_annotation(
        dict(font=dict(color='gray', size=10), x=2.88, y=2.37, showarrow=False, text="Zone (Gourmelon et al.)  ",
             textangle=0, xanchor='left', xref="x", yref="y"))
    fig.add_annotation(
        dict(font=dict(color='lightgray', size=10), x=2.96, y=2, showarrow=False, text="Front (Gourmelon et al.) ",
             textangle=0, xanchor='left', xref="x", yref="y"))
    fig.write_image("create_plots_new/output/error_season_%s.pdf"%modality, format='pdf')

    # Glacier subsetting
    fig = px.box(None, points="all", template="plotly_white", log_x=True, height=400)
    fig.add_trace(go.Box(x=list_of_mean_front_errors_without_nan, marker_color='orange', boxmean=True, boxpoints='all',
                         name='all', width=width, legendrank=7))
    color = {'Columbia': G10[7], 'Mapple': G10[8]}
    for glacier in ["Mapple", "COL", "Crane", "DBE", "JAC", "Jorum", "SI"]:
        subset_of_predictions = []
        for file_name in post_processed_predicted_masks:

            if not file_name[:-4].split('_')[0] == glacier:
                continue
            subset_of_predictions.append(file_name)
        if len(subset_of_predictions) == 0: continue
        all_errors, result_glacier = calculate_front_delineation_metric(complete_postprocessed_test_directory,
                                                                        subset_of_predictions,
                                                                        directory_of_target_fronts,
                                                                        bounding_boxes_directory)
        print(glacier, np.mean(all_errors), np.std(all_errors))
        if glacier == "COL":
            glacier = "Columbia"
        fig.add_trace(
            go.Box(x=all_errors, marker_color=color[glacier], boxmean=True, boxpoints='all', name=glacier,
                   width=width, ))
        for season in ['winter', 'summer']:
            print("")
            print("####################################################################")
            print(f"# Results for only images from {glacier}")
            print("####################################################################")
            subset_of_predictions = []
            for file_name in post_processed_predicted_masks:
                if glacier == "Columbia":
                    glacier = "COL"
                if not file_name[:-4].split('_')[0] == glacier:
                    continue
                winter = check_whether_winter_half_year(file_name)
                if (winter and season == "summer") or (not winter and season == "winter"):
                    continue
                subset_of_predictions.append(file_name)
            if len(subset_of_predictions) == 0: continue
            all_errors, result_glacier = calculate_front_delineation_metric(complete_postprocessed_test_directory,
                                                                            subset_of_predictions,
                                                                            directory_of_target_fronts,
                                                                            bounding_boxes_directory)
            print(glacier, np.mean(all_errors), np.std(all_errors))
            if glacier == "COL":
                season = " " + season
                glacier = "Columbia"
            fig.add_trace(
                go.Box(x=all_errors, marker_color=color[glacier], boxmean=True, boxpoints='all',
                       name=season + "_" + glacier,
                       width=width, ))
            results[glacier] = {}
            results[glacier]['all'] = result_glacier
    # Front Gourmelon
    offset = 0

    fig.add_shape(type='line', x0=140, y0=-0.5 + offset, x1=140, y1=0.5 + offset, line=dict(color='lightgray', ),
                  xref='x', yref='y')

    fig.add_shape(type='line', x0=173, y0=0.5 + offset, x1=173, y1=1.5 + offset, line=dict(color='lightgray', ),
                  xref='x', yref='y')

    fig.add_shape(type='line', x0=150, y0=1.5 + offset, x1=150, y1=2.5 + offset, line=dict(color='lightgray', ),
                  xref='x', yref='y')

    fig.add_shape(type='line', x0=907, y0=2.5 + offset, x1=907, y1=3.5 + offset, line=dict(color='lightgray', ),
                  xref='x', yref='y')

    fig.add_shape(type='line', x0=1157, y0=3.5 + offset, x1=1157, y1=4.5 + offset, line=dict(color='lightgray', ),
                  xref='x', yref='y')

    fig.add_shape(type='line', x0=1032, y0=4.5 + offset, x1=1032, y1=5.5 + offset, line=dict(color='lightgray', ),
                  xref='x', yref='y')

    fig.add_shape(type='line', x0=887, y0=5.5 + offset, x1=887, y1=6.5 + offset, line=dict(color='lightgray', ),
                  xref='x', yref='y')

    fig.add_shape(type='line', x0=262, y0=-0.5 + offset, x1=262, y1=0.5 + offset, line=dict(color='gray', ), xref='x',
                  yref='y')

    fig.add_shape(type='line', x0=340, y0=0.5 + offset, x1=340, y1=1.5 + offset, line=dict(color='gray', ), xref='x',
                  yref='y')

    fig.add_shape(type='line', x0=287, y0=1.5 + offset, x1=287, y1=2.5 + offset, line=dict(color='gray', ), xref='x',
                  yref='y')

    fig.add_shape(type='line', x0=854, y0=2.5 + offset, x1=854, y1=3.5 + offset, line=dict(color='gray', ), xref='x',
                  yref='y')

    fig.add_shape(type='line', x0=826, y0=3.5 + offset, x1=826, y1=4.5 + offset, line=dict(color='gray', ), xref='x',
                  yref='y')

    fig.add_shape(type='line', x0=840, y0=4.5 + offset, x1=840, y1=5.5 + offset, line=dict(color='gray', ), xref='x',
                  yref='y')

    fig.add_shape(type='line', x0=753, y0=5.5 + offset, x1=753, y1=6.5 + offset, line=dict(color='gray', ), xref='x',
                  yref='y')

    fig.add_annotation(dict(font=dict(color='gray', size=10), x=2.88, y=6.37 + offset, showarrow=False,
                            text="Zone (Gourmelon et al.)  ",
                            textangle=0, xanchor='left', xref="x", yref="y"))
    fig.add_annotation(dict(font=dict(color='lightgray', size=10), x=2.96, y=6 + offset, showarrow=False,
                            text="Front (Gourmelon et al.) ",
                            textangle=0, xanchor='left', xref="x", yref="y"))

    fig.update_layout(showlegend=False, font=dict(family="Times New Roma", size=12))
    fig.update_xaxes(title='Mean Distance Error [m]')
    fig.update_layout(yaxis={'categoryorder': 'array',
                             'categoryarray': ['summer_Mapple', 'winter_Mapple', 'Mapple', ' summer_Columbia', ' winter_Columbia', 'Columbia', 'all'],
                             'range': [-.5, 6.5]},

                      xaxis={'showgrid': True, "showline": True,
                             "tickvals": [10,20,50,100,200,500,1000,2000,5000],
                             })

    fig.update_traces(orientation='h')  # horizontal box plots
    fig.write_image("create_plots_new/output/error_glacier_%s.pdf"%modality, format='pdf')

    # Sensor subsetting
    color = {'ERS': G10[0], 'RSAT': G10[1], 'ENVISAT': G10[2], 'PALSAR': G10[3],
             'TDX': G10[5], 'Columbia_TDX': G10[5], 'Mapple_TDX': G10[5], 'Columbia_S1': G10[5], 'Mapple_S1': G10[5],
             'S1': G10[6]}
    fig = px.box(None, points="all", template="plotly_white", log_x=True, height=550)
    fig.add_trace(go.Box(x=list_of_mean_front_errors_without_nan, marker_color='orange', boxmean=True, boxpoints='all',
                         name='all', width=width))
    all_errors_TDXTSX = np.array([])
    for sensor in ["RSAT", "S1", "ENVISAT", "ERS", "PALSAR", "TDX", "TSX"]:

        print("")
        print("####################################################################")
        print(f"# Results for only images from {sensor}")
        print("####################################################################")
        subset_of_predictions = []
        for file_name in post_processed_predicted_masks:
            if not file_name[:-4].split('_')[2] == sensor:
                continue
            subset_of_predictions.append(file_name)

        if len(subset_of_predictions) == 0: continue

        all_errors, result_sensor = calculate_front_delineation_metric(complete_postprocessed_test_directory,
                                                                       subset_of_predictions,
                                                                       directory_of_target_fronts,
                                                                       bounding_boxes_directory)
        print(sensor, np.mean(all_errors), np.std(all_errors))
        if sensor == "TSX":
            sensor = "Mapple_TDX"
            all_errors_TDXTSX = np.append(all_errors_TDXTSX, all_errors)

        if sensor == "TDX":
            sensor = 'Columbia_TDX'
            all_errors_TDXTSX = np.append(all_errors_TDXTSX, all_errors)

        fig.add_trace(go.Box(x=all_errors, marker_color=color[sensor], boxmean=True, boxpoints='all', name=sensor,
                             width=width))

        results[sensor] = result_sensor

        if sensor == "S1":
            subset_of_predictions_Mapple = []
            subset_of_predictions_COL = []
            for file_name in subset_of_predictions:
                if file_name[:-4].split('_')[0] == "Mapple":
                    subset_of_predictions_Mapple.append(file_name)
                if file_name[:-4].split('_')[0] == "COL":
                    subset_of_predictions_COL.append(file_name)
            all_errors_mapple, result_sensor_mapple = calculate_front_delineation_metric(
                complete_postprocessed_test_directory,
                subset_of_predictions_Mapple,
                directory_of_target_fronts,
                bounding_boxes_directory)
            all_errors_COL, result_sensor_COL = calculate_front_delineation_metric(
                complete_postprocessed_test_directory,
                subset_of_predictions_COL,
                directory_of_target_fronts,
                bounding_boxes_directory)
            fig.add_trace(
                go.Box(x=all_errors_COL, marker_color=color['S1'], boxmean=True, boxpoints='all', name='Columbia_S1',
                       width=width))
            fig.add_trace(
                go.Box(x=all_errors_mapple, marker_color=color['S1'], boxmean=True, boxpoints='all', name='Mapple_S1',
                       width=width))
    fig.add_trace(
        go.Box(x=all_errors_TDXTSX, marker_color=color['TDX'], boxmean=True, boxpoints='all', name='TDX', width=width))

    MDE_all = 753
    fig.add_shape(type='line', x0=MDE_all, y0=8.5, x1=MDE_all, y1=9.5, line=dict(color='gray', ), xref='x', yref='y')
    fig.add_annotation(dict(font=dict(color='gray', size=10), x=2.88, y=9.37 + offset, showarrow=False,
                            text="Zone (Gourmelon et al.)  ",
                            textangle=0, xanchor='left', xref="x", yref="y"))
    fig.add_annotation(dict(font=dict(color='lightgray', size=10), x=2.96, y=9 + offset, showarrow=False,
                            text="Front (Gourmelon et al.) ",
                            textangle=0, xanchor='left', xref="x", yref="y"))
    fig.add_shape(type='line', x0=887, y0=8.5, x1=887, y1=9.5, line=dict(color='lightgray', ), xref='x', yref='y')

    MDE_s1_mapple = 141
    fig.add_shape(type='line', x0=MDE_s1_mapple, y0=-0.5, x1=MDE_s1_mapple, y1=0.5, line=dict(color='gray', ), xref='x',
                  yref='y')
    fig.add_shape(type='line', x0=206, y0=-0.5, x1=206, y1=0.5, line=dict(color='lightgray', ), xref='x', yref='y')

    MDE_s1_col = 2587
    fig.add_shape(type='line', x0=MDE_s1_col, y0=0.5, x1=MDE_s1_col, y1=1.5, line=dict(color='gray', ), xref='x',
                  yref='y')
    fig.add_shape(type='line', x0=3537, y0=0.5, x1=3537, y1=1.5, line=dict(color='lightgray', ), xref='x', yref='y')

    MDE_s1 = 2201
    fig.add_shape(type='line', x0=MDE_s1, y0=1.5, x1=MDE_s1, y1=2.5, line=dict(color='gray', ), xref='x',
                  yref='y')
    fig.add_shape(type='line', x0=2806, y0=1.5, x1=2806, y1=2.5, line=dict(color='lightgray', ), xref='x', yref='y')

    MDE_TDX_mapple = 246
    fig.add_shape(type='line', x0=MDE_TDX_mapple, y0=2.5, x1=MDE_TDX_mapple, y1=3.5, line=dict(color='gray', ),
                  xref='x', yref='y')
    fig.add_shape(type='line', x0=129, y0=2.5, x1=129, y1=3.5, line=dict(color='lightgray', ), xref='x', yref='y')

    MDE_TDX_col = 587
    fig.add_shape(type='line', x0=MDE_TDX_col, y0=3.5, x1=MDE_TDX_col, y1=4.5, line=dict(color='gray', ),
                  xref='x', yref='y')
    fig.add_shape(type='line', x0=744, y0=3.5, x1=744, y1=4.5, line=dict(color='lightgray', ), xref='x', yref='y')

    MDE_TDX = 547
    fig.add_shape(type='line', x0=MDE_TDX, y0=4.5, x1=MDE_TDX, y1=5.5, line=dict(color='gray', ),
                  xref='x', yref='y')
    fig.add_shape(type='line', x0=663, y0=4.5, x1=663, y1=5.5, line=dict(color='lightgray', ), xref='x', yref='y')

    MDE_palsar = 437
    fig.add_shape(type='line', x0=MDE_palsar, y0=5.5, x1=MDE_palsar, y1=6.5, line=dict(color='gray', ), xref='x',
                  yref='y')
    fig.add_shape(type='line', x0=197, y0=5.5, x1=197, y1=6.5, line=dict(color='lightgray', ), xref='x', yref='y')

    MDE_envisat = 493
    fig.add_shape(type='line', x0=MDE_envisat, y0=6.5, x1=MDE_envisat, y1=7.5, line=dict(color='gray', ), xref='x',
                  yref='y')
    fig.add_shape(type='line', x0=191, y0=6.5, x1=191, y1=7.5, line=dict(color='lightgray', ), xref='x', yref='y')

    MDE_ers = 437
    fig.add_shape(type='line', x0=MDE_ers, y0=7.5, x1=MDE_ers, y1=8.5, line=dict(color='gray', ), xref='x', yref='y')
    fig.add_shape(type='line', x0=127, y0=7.5, x1=127, y1=8.5, line=dict(color='lightgray', ), xref='x', yref='y')

    fig.update_layout(showlegend=False, font=dict(family="Times New Roma", size=12, color="black"))
    fig.update_xaxes(title='Mean Distance Error [m]')
    fig.update_layout(
        yaxis={'categoryorder': 'array',
               'categoryarray': ['Mapple_S1', 'Columbia_S1', 'S1', 'Mapple_TDX', 'Columbia_TDX', 'TDX', 'PALSAR',
                                 'ENVISAT', 'ERS', 'all'], 'range': [-.5, 10.5]},
        xaxis={'showgrid': True, "showline": True,
                "tickvals": [10,20,50,100,200,500,1000,2000,5000]})

    fig.update_traces(orientation='h')  # horizontal box plots
    fig.write_image("create_plots_new/output/error_satellite_%s.pdf"%modality, format='pdf')

    # Resolution subsetting
    fig = px.box(None, points="all", template="plotly_white", log_x=True,height=300 )
    fig.add_trace(go.Box(x=list_of_mean_front_errors_without_nan, marker_color='orange', boxmean=True, boxpoints='all',
                         name='all', width=width))
    color = {20: G10[9], 17: G10[8], 7: G10[3]}
    for res in [20, 17, 7]:
        print("")
        print("####################################################################")
        print(f"# Results for only images with a resolution of {res}")
        print("####################################################################")
        subset_of_predictions = []
        for file_name in post_processed_predicted_masks:
            if not int(file_name[:-4].split('_')[3]) == res:
                continue
            subset_of_predictions.append(file_name)
        if len(subset_of_predictions) == 0: continue
        all_errors, result_res = calculate_front_delineation_metric(complete_postprocessed_test_directory,
                                                                    subset_of_predictions, directory_of_target_fronts,
                                                                    bounding_boxes_directory)
        fig.add_trace(
            go.Box(x=all_errors, marker_color=color[res], boxmean=True, boxpoints='all', name=str(res)+" m per pixel", width=width))
        results[res] = result_res
    fig.update_layout(showlegend=False, font=dict(family="Times New Roma", size=12))
    fig.update_xaxes(title='Mean Distance Error [m]', )
    fig.update_yaxes(title='Resolution')
    fig.update_layout(yaxis={'categoryorder': 'array', 'categoryarray': ['7 m per pixel', '17 m per pixel', '20 m per pixel', 'all']},
                      xaxis={'showgrid': True, "showline": True,
                             "tickvals": [10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
                             } )
    fig.update_traces(orientation='h')  # horizontal box plots
    fig.write_image("create_plots_new/output/error_resolution_%s.pdf"%modality, format='pdf')

    # Season and glacier subsetting
    for glacier in ["Mapple", "Columbia", "Crane", "DBE", "JAC", "Jorum", "SI"]:
        for season in ["winter", "summer"]:
            print("")
            print("####################################################################")
            print(f"# Results for only images in {season} and from {glacier}")
            print("####################################################################")
            subset_of_predictions = []
            for file_name in post_processed_predicted_masks:
                winter = check_whether_winter_half_year(file_name)
                if not file_name[:-4].split('_')[0] == glacier:
                    continue
                if (winter and season == "summer") or (not winter and season == "winter"):
                    continue
                subset_of_predictions.append(file_name)
            if len(subset_of_predictions) == 0: continue
            _, results_gla_season = calculate_front_delineation_metric(complete_postprocessed_test_directory,
                                                                       subset_of_predictions,
                                                                       directory_of_target_fronts,
                                                                       bounding_boxes_directory)
            results[glacier][season] = results_gla_season

    return results


def visualizations(complete_postprocessed_test_directory, directory_of_target_fronts, directory_of_sar_images,
                   bounding_boxes_directory, visualizations_dir):
    print("Creating visualizations ...\n\n")
    post_processed_predicted_masks = os.listdir(os.path.join(complete_postprocessed_test_directory))
    for file_name in post_processed_predicted_masks:
        if not file_name.endswith('.png'):
            continue
        resolution = int(os.path.split(file_name)[1][:-4].split('_')[-3])
        if resolution < 10:
            dilation = 5
        else:
            dilation = 3

        if file_name.endswith('_front.png'):
            post_processed_predicted_mask = cv2.imread(
                os.path.join(complete_postprocessed_test_directory, file_name).__str__(), cv2.IMREAD_GRAYSCALE)
            post_processed_predicted_mask = mask_prediction_with_bounding_box(post_processed_predicted_mask,
                                                                              file_name[:-len('_front.png')] + '.png',
                                                                              bounding_boxes_directory)
            post_processed_predicted_mask[post_processed_predicted_mask > 1] = 1
            post_processed_predicted_mask_skeletonized = skeletonize(post_processed_predicted_mask)
            post_processed_predicted_mask = np.zeros(post_processed_predicted_mask_skeletonized.shape)
            post_processed_predicted_mask[post_processed_predicted_mask_skeletonized] = 255
            matching_target_file = get_matching_out_of_folder(file_name[:-len('_front.png')] + '.png',
                                                              directory_of_target_fronts)
            target_front = cv2.imread(os.path.join(directory_of_target_fronts, matching_target_file).__str__(),
                                      cv2.IMREAD_GRAYSCALE)
            matching_sar_file = get_matching_out_of_folder(file_name[:-len('_front.png')] + '.png',
                                                           directory_of_sar_images)
            sar_image = cv2.imread(os.path.join(directory_of_sar_images, matching_sar_file).__str__(),
                                   cv2.IMREAD_GRAYSCALE)
        elif file_name.endswith('_zone.png'):
            continue
        elif file_name.endswith('_recon.png'):
            continue
        else:
            post_processed_predicted_mask = cv2.imread(
                os.path.join(complete_postprocessed_test_directory, file_name).__str__(), cv2.IMREAD_GRAYSCALE)
            matching_target_file = get_matching_out_of_folder(file_name, directory_of_target_fronts)
            target_front = cv2.imread(os.path.join(directory_of_target_fronts, matching_target_file).__str__(),
                                      cv2.IMREAD_GRAYSCALE)
            matching_sar_file = get_matching_out_of_folder(file_name, directory_of_sar_images)
            sar_image = cv2.imread(os.path.join(directory_of_sar_images, matching_sar_file).__str__(),
                                   cv2.IMREAD_GRAYSCALE)

        predicted_front = np.array(post_processed_predicted_mask)
        ground_truth_front = np.array(target_front)
        kernel = np.ones((dilation, dilation), np.uint8)
        predicted_front = cv2.dilate(predicted_front, kernel, iterations=1)
        ground_truth_front = cv2.dilate(ground_truth_front, kernel, iterations=1)

        sar_image = np.array(sar_image)
        sar_image_rgb = skimage.color.gray2rgb(sar_image)
        sar_image_rgb = np.uint8(sar_image_rgb)

        sar_image_rgb[predicted_front > 0] = [0, 255, 255]  # b, g, r
        sar_image_rgb[ground_truth_front > 0] = [255, 51, 51]
        correct_prediction = np.logical_and(predicted_front, ground_truth_front)
        sar_image_rgb[correct_prediction > 0] = [255, 0, 255]  # [51, 255, 51]   # [0, 153, 0]

        # Insert Bounding Box
        matching_bounding_box_file = get_matching_out_of_folder(file_name, bounding_boxes_directory)
        with open(os.path.join(bounding_boxes_directory, matching_bounding_box_file)) as f:
            coord_file_lines = f.readlines()
        left_upper_corner_x, left_upper_corner_y = [round(float(coord)) for coord in coord_file_lines[1].split(",")]
        left_lower_corner_x, left_lower_corner_y = [round(float(coord)) for coord in coord_file_lines[2].split(",")]
        right_lower_corner_x, right_lower_corner_y = [round(float(coord)) for coord in coord_file_lines[3].split(",")]
        right_upper_corner_x, right_upper_corner_y = [round(float(coord)) for coord in coord_file_lines[4].split(",")]

        bounding_box = np.zeros((len(sar_image_rgb), len(sar_image_rgb[0])))
        if left_upper_corner_x < 0: left_upper_corner_x = 0
        if left_lower_corner_x < 0: left_lower_corner_x = 0
        if right_upper_corner_x > len(sar_image_rgb[0]): right_upper_corner_x = len(sar_image_rgb[0]) - 1
        if right_lower_corner_x > len(sar_image_rgb[0]): right_lower_corner_x = len(sar_image_rgb[0]) - 1
        if left_upper_corner_y > len(sar_image_rgb): left_upper_corner_y = len(sar_image_rgb) - 1
        if left_lower_corner_y < 0: left_lower_corner_y = 0
        if right_upper_corner_y > len(sar_image_rgb): right_upper_corner_y = len(sar_image_rgb) - 1
        if right_lower_corner_y < 0: right_lower_corner_y = 0

        bounding_box[left_upper_corner_y, left_upper_corner_x:right_upper_corner_x] = 1
        bounding_box[left_lower_corner_y, left_lower_corner_x:right_lower_corner_x] = 1
        bounding_box[left_lower_corner_y:left_upper_corner_y, left_upper_corner_x] = 1
        bounding_box[right_lower_corner_y:right_upper_corner_y, right_lower_corner_x] = 1
        bounding_box = cv2.dilate(bounding_box, kernel, iterations=1)
        sar_image_rgb[bounding_box > 0] = [255, 255, 0]

        cv2.imwrite(os.path.join(visualizations_dir, file_name), sar_image_rgb)


def main(complete_test_directory, directory_of_complete_targets_zones, directory_of_complete_targets_fronts,
         directory_of_sar_images):
    # ###############################################################################################
    # CALCULATE SEGMENTATION METRICS (IoU & Hausdorff Distance)
    # ###############################################################################################
    complete_predicted_masks_zones = list(
        file for file in os.listdir(complete_test_directory) if file.endswith('_zone.png'))
    complete_predicted_masks_fronts = list(
        file for file in os.listdir(complete_test_directory) if file.endswith('_front.png'))
    src = Path(directory_of_sar_images).parent.parent.parent
    bounding_boxes_directory = os.path.join(src, "data_raw", "bounding_boxes")
    results = {}
    # only on zone
    
    if len(complete_predicted_masks_zones) > 0:
        results_seg = calculate_segmentation_metrics('zones', complete_predicted_masks_zones, complete_test_directory,
                                      directory_of_complete_targets_zones,)
        results['Zone_Segmentation'] = results_seg
    
    if len(complete_predicted_masks_fronts) >0:
        results_seg = calculate_segmentation_metrics('fronts', complete_predicted_masks_fronts,
                                                     complete_test_directory,
                                                     directory_of_complete_targets_fronts, )
        results['Front_Segmentation'] = results_seg
    
    # ###############################################################################################
    # POST-PROCESSING
    # ###############################################################################################
    src = Path(directory_of_sar_images).parent.parent.parent
    print(src)
    
    
    if len(complete_predicted_masks_zones) > 0:
        post_processing('zones', complete_predicted_masks_zones, bounding_boxes_directory, complete_test_directory)
    
    # ###############################################################################################
    # CALCULATE FRONT DELINEATION METRIC (Mean distance error)
    # ###############################################################################################

    if len(complete_predicted_masks_zones) > 0:
        print("Front delineation from ZONE post processed")
        results_zone = front_delineation_metric('zone', complete_postprocessed_test_directory,
                                                directory_of_complete_targets_fronts, bounding_boxes_directory)
        results['Zone_Delineation'] = results_zone

    if len(complete_predicted_masks_fronts) > 0:
        print("Front delineation from FRONT directly")
        results_front = front_delineation_metric('front', complete_test_directory, directory_of_complete_targets_fronts,
                                                 bounding_boxes_directory)
        results['Front_Delineation'] = results_front

    results_file = open(complete_test_directory + '/eval_results.json', "w")
    json.dump(results, results_file)

    # ###############################################################################################
    # MAKE VISUALIZATIONS
    # ###############################################################################################
    if len(complete_predicted_masks_zones) > 0:
        visualizations(complete_postprocessed_test_directory, directory_of_complete_targets_fronts,
                       directory_of_sar_images,
                       bounding_boxes_directory, visualizations_dir)

    if len(complete_predicted_masks_fronts) > 0:
        front_prediction_dir = complete_test_directory

        visualizations(front_prediction_dir, directory_of_complete_targets_fronts, directory_of_sar_images,
                       bounding_boxes_directory, visualizations_dir)


if __name__ == "__main__":
    print("Start Evaluation")
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--predictions', help="Directory with predictions as png")
    parser.add_argument('--labels_fronts', help="Directory with labels as png")
    parser.add_argument('--labels_zones', help="Directory with labels as png")
    parser.add_argument('--sar_images', help="Directory with sar images")
    hparams = parser.parse_args()

    complete_test_directory = hparams.predictions
    complete_postprocessed_test_directory = os.path.join(complete_test_directory, "postprocessed")

    os.makedirs(complete_postprocessed_test_directory, exist_ok=True)

    visualizations_dir = os.path.join(complete_test_directory, "visualization")
    os.makedirs(visualizations_dir, exist_ok=True)

    main(hparams.predictions, hparams.labels_zones, hparams.labels_fronts, hparams.sar_images)
