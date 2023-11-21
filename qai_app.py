import streamlit as st 
import pandas as pd 
import cv2 
from PIL import Image
import numpy as np 
from utils import *
import os 
from stqdm import stqdm
import matplotlib.gridspec as gridspec

def compare_image_quality(image_ref, image_compare):

    print('--')
    print(image_ref.shape , image_compare.shape)
    # Resize the comparison image to the size of the reference image
    if image_ref.shape != image_compare.shape:
        image_compare = cv2.resize(image_compare, (image_ref.shape[1], image_ref.shape[0]))


    # Compute MSE
    mse_value = np.sum((image_ref.astype("float") - image_compare.astype("float")) ** 2)
    mse_value /= float(image_ref.shape[0] * image_ref.shape[1])

    pixel_max = float(np.max(image_ref))
    # Compute PSNR
    if mse_value == 0:
        psnr_value = float('inf')
    else:
        psnr_value = 20 * np.log10(pixel_max / np.sqrt(mse_value))

    # Compute SSIM
    ssim_value, _ = ssim(image_ref, image_compare, full=True)

    return mse_value, psnr_value, ssim_value

def read_image(image):
    pil_image = Image.open(image) 
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    return gray_image  

image_reference_path = st.sidebar.file_uploader('Upload image reference') 

if image_reference_path:
    image_reference = read_image(image_reference_path)
    st.image(image_reference)
    mselist=[]
    psnr=[]
    ssimlist=[]
    for frame_name in stqdm(os.listdir('frames')):
        frame_path= 'frames/'+frame_name
        image_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        mse_value, psnr_value, ssim_value=compare_image_quality(image_reference, image_frame) 
        print(mse_value, psnr_value, ssim_value)
        mselist.append(mse_value)
        psnr.append(psnr_value)
        ssimlist.append(ssim_value)

    data={}
    data['mse']=mselist
    data['psnr']=psnr
    data['ssim']=ssimlist

    n = len(data['mse'])
    sec = 5 
    x = np.linspace(0, int(n*5), n)
    y1 = data['mse']
    y2 = data['psnr']
    y3 = data['ssim']
    fig = plt.figure(figsize=(10,8),tight_layout=True)

    gs = gridspec.GridSpec(3, 2)

    ax0 = fig.add_subplot(gs[:,0])
    ax0.imshow(image_reference)
    ax0.set_title('Reference image')

    ax1 = fig.add_subplot(gs[0,1])
    ax1.plot(x, y1, color='blue')
    ax1.set_title('mse behavior')
    ax1.set_ylabel('mean squared error')
    ax1.set_xlabel('frames')

    ax2 = fig.add_subplot(gs[1,1])
    ax2.plot(x, y2, color='green')
    ax2.set_title('psnr behavior')
    ax2.set_ylabel('Peak Signal-to-Noise Ratio')
    ax2.set_xlabel('frames')

    ax3 = fig.add_subplot(gs[2,1])
    ax3.plot(x, y3, color='red')
    ax3.set_title('ssim behavior')
    ax3.set_ylabel('Structural Similarity Index')
    ax3.set_xlabel('frames')

    # Adjust layout and show plot
    plt.tight_layout()
    st.pyplot(fig)