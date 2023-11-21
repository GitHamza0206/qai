import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt 

def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse, diff


def compare_image_quality(image_path_ref, image_path_compare):
    # Load the images
    image_ref = cv2.imread(image_path_ref, cv2.IMREAD_GRAYSCALE)
    image_compare = cv2.imread(image_path_compare, cv2.IMREAD_GRAYSCALE)
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

def validate_image(image_path_ref, 
                   image_path_compare,
                   mse_threshold=3000,
                   psnr_threshold=30,
                   ssim_threshold=0.5
                   ):
    
    mse, psnr, ssim = compare_image_quality(image_path_ref, image_path_compare)
    # Check for threshold values
    result = ""
    if mse > mse_threshold:
        result += "Alert: MSE %.2f exceeds threshold %.2f \n" % (mse, mse_threshold)
        print("Alert: MSE %.2f exceeds threshold %.2f \n" % (mse, mse_threshold))
    if psnr < psnr_threshold:
        print(
            "Alert: PSNR %.2f is below threshold %.2f" % (psnr, psnr_threshold) ) 
        
        result += "Alert: PSNR %.2f is below threshold %.2f\n"% (psnr, psnr_threshold)
    if ssim < ssim_threshold:
        print(
            "Alert: SSIM %.2f is below threshold %.2f " % (ssim, ssim_threshold) )
        result += "Alert: SSIM %.2f is below threshold %.2f \n " % (ssim, ssim_threshold)
    
    return result


ref_image = "ref_images/img-ref.jpg"
compare_image = "Dirty Lens/image_14-07-2021_22-20-41_329152.jpg"
#compare_image = "Dirty Lens/image_18-07-2021_20-54-41_033344.jpg"
compare_image = "Dirty Lens/image_29-06-2021_10-55-04_514607.jpg"
compare_image = "Dirty Lens/Lens Cleaning/Lens wash (2).jpg"


#mse, psnr, ssim_val= compare_image_quality(ref_image, compare_image)


def show_result(image_path_ref, image_path_compare):
    image_ref = cv2.imread(image_path_ref)
    image_compare = cv2.imread(image_path_compare)
    
    result = validate_image(image_path_ref=ref_image,
                    image_path_compare=compare_image,
                    mse_threshold=10,
                    psnr_threshold=30,
                    ssim_threshold=0.5)
    # initialize the figure
    fig = plt.figure("Images")
    plt.suptitle("MSE: %.2f,PSNR:%.2f,SSIM: %.2f" % (mse, psnr, ssim_val))
    plt.suptitle(result)
    # loop over the images
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(image_ref, cmap = plt.cm.gray)
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(image_compare, cmap = plt.cm.gray)
    plt.axis("off")
    # show the figure
    plt.show()


if __name__ == '__main__':
    validate_image(image_path_ref=ref_image,
                    image_path_compare=compare_image,
                    mse_threshold=10,
                    psnr_threshold=30,
                    ssim_threshold=0.5)
    
    show_result(ref_image, compare_image)
    pass