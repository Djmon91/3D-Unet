#coding:utf-8
import os, sys, time
import argparse
import SimpleITK as sitk
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Calc jaccard index')
    parser.add_argument('--inputImageFile', '-i', help='Input image file')
    parser.add_argument('--maskImageFile', 	'-m', help='Mask Image File')
    parser.add_argument('--outputImageFile','-o', help='Output image file')

    args = parser.parse_args()

    # Read image(input, mask)
    sitkInput = sitk.ReadImage(args.inputImageFile)
    input = sitk.GetArrayFromImage(sitkInput)
    _shape = input.shape
    input = input.flatten()
    sitkMask = sitk.ReadImage(args.maskImageFile)
    mask = sitk.GetArrayFromImage(sitkMask).flatten()
    
    # Extract mask region from input image
    extracted_input = np.array([i if m else 0 for i, m in zip(input, mask)], dtype=input.dtype)
    assert(input.shape == extracted_input.shape)
    extracted_input = extracted_input.reshape(_shape)

    # Save extracted input image
    saveInput = sitk.GetImageFromArray(extracted_input)
    saveInput.SetSpacing(sitkInput.GetSpacing())
    saveInput.SetOrigin(sitkInput.GetOrigin())
    result_dir = os.path.dirname(args.outputImageFile)
    if not os.path.exists(result_dir):
    	os.makedirs(result_dir)
    sitk.WriteImage(saveInput, args.outputImageFile)



if __name__ == '__main__':
    main()