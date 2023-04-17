import imageio
from skimage import io
import skimage

import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
import copy

from datetime import date
import numpy as np
from argparse import ArgumentParser
from skimage.transform import resize
# import matplotlib.pyplot as plt
import cv2


def color_map(m):
    return m[0] * np.array([1, 1, 1]) + (255 - m[0]) * np.array([0, 0, 1])


def createOverlay(image, front, zone, boundary):
    """
    creates an image with the front label overlaying the glacier image

    :param image: Image of the glacier
    :param front: Image of the label of the front
    :return: an rgb image with the black and white image and red front line
    """

    # value for NA area=0, stone=64, glacier=127, ocean with ice melange=254

    image_rgb = np.array(image * 0.5, dtype=np.uint8)

    try:
        image_rgb[zone == 0] += np.array(np.array([0, 0, 0]) / 2, dtype=np.uint8)
        image_rgb[zone == 64] += np.array(np.array([52, 46, 55]) / 2, dtype=np.uint8)
        image_rgb[zone == 127] += np.array(np.array([254, 254, 254]) / 2, dtype=np.uint8)
        image_rgb[zone == 254] += np.array(np.array([60, 145, 230]) / 2, dtype=np.uint8)

    finally:
        #try:
        #    image_rgb[boundary > 0] = np.array(np.array([241, 143, 1]), dtype=np.uint8)
        #finally:
        image_rgb[front == 255] = np.array(np.array([255, 0, 0]), dtype=np.uint8)

    return image_rgb


def create_target(sar_image_path):
    sample_name = sar_image_path.split('/')[-1]
    sar_image = cv2.imread(sar_image_path)
    front_image_path = '/home/ho11laqe/PycharmProjects/data_raw/fronts_dilated_5/train/' + sample_name[
                                                                                           :-len('.png')] + '_front.png'
    zone_image_path = '/home/ho11laqe/PycharmProjects/data_raw/zones/train/' + sample_name[
                                                                               :-len('.png')] + '_zones.png'

    boundary_image_path = '/home/ho11laqe/PycharmProjects/data_raw/boundaries_dilated_5/train/' + sample_name[
                                                                                                  :-len(
                                                                                                      '.png')] + '_boundary.png'
    front = cv2.imread(front_image_path, cv2.IMREAD_GRAYSCALE)
    zone = cv2.imread(zone_image_path, cv2.IMREAD_GRAYSCALE)
    boundary = cv2.imread(boundary_image_path, cv2.IMREAD_GRAYSCALE)
    overlay = createOverlay(sar_image, front, zone, boundary)
    cv2.imwrite('output/target.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--image_dir', help="Directory with predictions as png")
    args = parser.parse_args()

    image_dir = args.image_dir

    front_gif = []
    fronts = []
    zone_gif = []
    zones = []
    boundary_gif = []
    boundaries = []

    sar_image_path = '/home/ho11laqe/PycharmProjects/data_raw/sar_images/train/DBE_2008-03-30_TSX_7_3_049.png'
    sar_image = cv2.imread(sar_image_path)
    shape = sar_image.shape
    new_shape = (int(shape[1] / 4), int(shape[0] / 4))
    sar_image = cv2.resize(sar_image, new_shape)

    create_target(sar_image_path)

    list_images = os.listdir(image_dir)
    list_images.sort(key=lambda y: int(y.split('_')[6]))

    for i, image_file in enumerate(list_images[:300]):
        epoch = image_file.split('_')[6]
        if image_file.endswith('_front.png'):
            print(image_file)
            front = cv2.imread(image_dir + '/' + image_file, cv2.IMREAD_GRAYSCALE)
            front = cv2.resize(front, new_shape, interpolation=cv2.INTER_NEAREST)
            # image = Image.fromarray(front)
            # image_draw = ImageDraw.Draw(image)
            # image_draw.text((1,1), 'Epoch: '+str(epoch))
            # front_gif.append(image)
            fronts.append(front)
        elif image_file.endswith('_zone.png'):
            print(image_file)
            zone = cv2.imread(image_dir + '/' + image_file, cv2.IMREAD_GRAYSCALE)
            zone = cv2.resize(zone, new_shape, interpolation=cv2.INTER_NEAREST)
            # image = Image.fromarray(zone)
            # image_draw = ImageDraw.Draw(image)
            # image_draw.text((1, 1), 'Epoch: ' + str(epoch))
            # zone_gif.append(image)
            zones.append(zone)
        elif image_file.endswith('_boundary.png'):
            print(image_file)
            boundary = cv2.imread(image_dir + '/' + image_file, cv2.IMREAD_GRAYSCALE)
            boundary = cv2.resize(boundary, new_shape, interpolation=cv2.INTER_NEAREST)
            # image = Image.fromarray(boundary)
            # image_draw = ImageDraw.Draw(image)
            # image_draw.text((1, 1), 'Epoch: ' + str(epoch))
            # boundary_gif.append(image)
            boundaries.append(boundary)

    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 40)
    font_legend = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 20)
    overlay_gif = []
    for epoch, (front, zone, boundary) in enumerate(zip(fronts, zones, boundaries)):
        overlay = createOverlay(sar_image, front, zone, boundary)
        image = Image.fromarray(overlay)
        image_draw = ImageDraw.Draw(image)

        image_draw.rectangle((0, 40, 195, 210), fill='gray')

        image_draw.rectangle((10, 60, 30, 80), fill=(60, 145, 230, 120))
        image_draw.text((35, 60), 'Ocean', font=font_legend)

        image_draw.rectangle((10, 90, 30, 110), fill=(255, 255, 255))
        image_draw.text((35, 90), 'Glacier', font=font_legend)

        image_draw.rectangle((10, 120, 30, 140), fill=(255, 0, 0))
        image_draw.text((35, 120), 'Glacier Front', font=font_legend)

        image_draw.rectangle((10, 150, 30, 170), fill=(92, 76, 85))
        image_draw.text((35, 150), 'Rock', font=font_legend)

        image_draw.rectangle((10, 180, 30, 200), fill=(0, 0, 0))
        image_draw.text((35, 180), 'Shadow', font=font_legend)

        image_draw.rectangle((0, 0, 330, 45), fill='gray')
        image_draw.text((8, 1), 'Epoch:%03i' % epoch + '/' + str(len(fronts)), font=font, )
        if epoch < 10:
            for i in range(10 - epoch):
                print(i)
                overlay_gif.append(image)
        else:
            overlay_gif.append(image)

    frame_one = overlay_gif[0]
    frame_one.save("output/overlay.gif", format="GIF", append_images=overlay_gif,
                   save_all=True, duration=200, loop=0)
