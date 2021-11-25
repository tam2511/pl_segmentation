import cv2
import numpy as np
import os


def check_image_load(path: str) -> bool:
    '''
    Checks that the picture on the specified path is correct
    :param path: path to image
    :return: is the path correct
    '''
    try:
        image = cv2.imread(path)
        _ = image.shape
        return True
    except Exception:
        return False


def read_image(image_path: str):
    '''
    Read image from file
    :param image_path: path of file
    :return: opencv image or None
    '''
    with open(image_path, 'rb') as f:
        nparr = np.fromstring(f.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image


def get_label(image_path: str) -> str:
    '''
    Parsing label from path of file
    :param image_path: path of file
    :return: label name
    '''
    while os.path.dirname(image_path) != '':
        image_path = os.path.dirname(image_path)
    return image_path
