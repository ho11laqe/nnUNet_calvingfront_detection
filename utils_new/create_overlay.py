import argparse
import os
from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.image as pltimage
import cv2

def main(predictions, labels, sar_images):
    prediction_files = os.listdir(predictions)
    output_folder = join(predictions, 'eval')
    maybe_mkdir_p(output_folder)
    for file in prediction_files:
        if file.endswith('.png'):
            file = file[:-len('.png')]

            prediction_path = join(predictions, file + '.png')
            label_path = join(labels, file + '_front.png')
            image_path = join(sar_images, file + '.png')

            prediction = io.imread(prediction_path, as_gray=True)
            label = io.imread(label_path)
            image = io.imread(image_path)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            image_rgb[label > 0] = [0, 255, 0]
            image_rgb[prediction > 0] = [255, 0, 0]

            output_path = join(output_folder, file+'.png')
            pltimage.imsave(output_path, image_rgb)
            print(file)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", '--predictions', help="Folder with png files of prediction")
    parser.add_argument("-l", '--labels', help="Folder with png files of labels")
    parser.add_argument("-i", '--sar_images', help="Folder with sar_images of glacier")
    args = parser.parse_args()
    predictions = args.predictions
    labels = args.labels
    sar_images = args.sar_images

    main(predictions, labels, sar_images)