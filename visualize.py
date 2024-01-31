import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import nibabel as nib
import cv2 as cv


def load_and_plot(dir, protocol, time, sbref):
    img = nib.load(dir)

    print(img.shape)
    if sbref:
        plt.figure(1); plt.clf()
        plt.imshow(img.dataobj[:,:, 32], cmap = "gray")
        plt.show()
        return
    if time == 1:
        max = 490 if protocol == 'rs' else 332
        step = 10
    else:
        max = 64
        step = 4

    plot_img = img.dataobj[:, :, 32, :] if time else img.dataobj[:, :, :, 1]

    for i in range(0, max, step):
        plt.figure(1); plt.clf()
        plt.title(f"Slice {i} of {max}")
        plt.imshow(plot_img[:, :, i], cmap="gray")
        plt.pause(.5)
        
    plt.close()

rs_processed = "imaging_data_eid1236578/rsfMRI/NIFTI/1236578_20227_2_0/rfMRI.ica/filtered_func_data_clean.nii"
ts_processed = "imaging_data_eid1236578/tfMRI/NIFTI/1236578_20249_2_0/tfMRI.feat/filtered_func_data.nii.gz"

rs_og = "imaging_data_eid1236578/rsfMRI/NIFTI/1236578_20227_2_0/rfMRI.nii.gz"
ts_og = "imaging_data_eid1236578/tfMRI/NIFTI/1236578_20249_2_0/tfMRI.nii.gz"

rs_sbref = "imaging_data_eid1236578/rsfMRI/NIFTI/1236578_20227_2_0/rfMRI_SBREF.nii.gz"
ts_og = "imaging_data_eid1236578/tfMRI/NIFTI/1236578_20249_2_0/tfMRI_SBREF.nii.gz"

# load_and_plot(rs_og, 'rs', 1, 0)

img = nib.load(rs_processed)
plt.figure()
t1_img=img.dataobj[:,:, :, 1]

plt.imshow(t1_img[:,:,22], cmap = "gray")
plt.show()

