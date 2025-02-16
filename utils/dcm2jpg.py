import os
import cv2
import numpy as np
from PIL import Image
import sys
import shutil
import time
import pydicom
from pydicom import dcmread
import SimpleITK as sitk

import pydicom
from glob import glob
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut


# get dcm pixelspacing
def get_pixelspacing(path):
    dcm_data = pydicom.dcmread(path)
    row, col = dcm_data.get('PixelSpacing', 'None')
    return row, col


def Dcm2jpg(file_path):
    # Get all picture names
    pruuid_list = []

    file_name_list = []
    ps_x_list = []
    ps_y_list = []

    names = os.listdir(file_path)
    name = os.listdir(file_path)[0][:-4]
    directory = file_path + name + '/'

    # Separate the file names in the folder from the. DCM following them
    # for name in names:
    #  if name.split('.')[-1] == 'dcm':
    #      index = name.rfind('.')
    #      name = name[:index]
    #      file_name_list.append(name)

    for files in names:
        if files[-4:] != '.dcm':
            continue
        files = files[:-4]
        directory = file_path + 'jpg'

        if not os.path.exists(directory):
            os.makedirs(directory)

        i_path = file_path + files + ".dcm"
        row, col = get_pixelspacing(i_path)
        ps_x_list.append(row)
        ps_y_list.append(col)
        out_path = file_path + 'jpg' + '/' + files + ".jpg"
        print('pic path------------', i_path)
        print('out path------------', out_path)
        img_jpg = convert_image_og(i_path, out_path)
        row, col = get_pixelspacing(i_path)
    #  return img_jpg, ps_x_list, ps_y_list
    return img_jpg, ps_x_list, ps_y_list, row, col


def convert_image_og(input_file_name, output_file_name, new_width=None):
    try:
        image_file_reader = sitk.ImageFileReader()
        # only read DICOM images
        image_file_reader.SetImageIO('GDCMImageIO')
        image_file_reader.SetFileName(input_file_name)
        image_file_reader.ReadImageInformation()
        image_size = list(image_file_reader.GetSize())
        if len(image_size) == 3 and image_size[2] == 1:
            image_size[2] = 0
        image_file_reader.SetExtractSize(image_size)
        image = image_file_reader.Execute()
        if new_width:
            print('hre')
            original_size = image.GetSize()
            original_spacing = image.GetSpacing()
            new_spacing = [(original_size[0] - 1) * original_spacing[0]
                           / (new_width - 1)] * 2
            new_size = [new_width, int((original_size[1] - 1)
                                       * original_spacing[1] / new_spacing[1])]
            image = sitk.Resample(image1=image, size=new_size,
                                  transform=sitk.Transform(),
                                  interpolator=sitk.sitkLinear,
                                  outputOrigin=image.GetOrigin(),
                                  outputSpacing=new_spacing,
                                  outputDirection=image.GetDirection(),
                                  defaultPixelValue=0,
                                  outputPixelType=image.GetPixelID())
        # If a single channel image, rescale to [0,255]. Also modify the
        # intensity values based on the photometric interpretation. If
        # MONOCHROME2 (minimum should be displayed as black) we don't need to
        # do anything, if image has MONOCRHOME1 (minimum should be displayed as
        # white) we flip # the intensities. This is a constraint imposed by ITK
        # which always assumes MONOCHROME2.
        if image.GetNumberOfComponentsPerPixel() == 1:
            image = sitk.RescaleIntensity(image, 0, 255)
            if image_file_reader.GetMetaData('0028|0004').strip() \
                    == 'MONOCHROME1':
                image = sitk.InvertIntensity(image, maximum=255)
            image = sitk.Cast(image, sitk.sitkUInt8)
        sitk.WriteImage(image, output_file_name)
        return True
    except BaseException:
        return False

