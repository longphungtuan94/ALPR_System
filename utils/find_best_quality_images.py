import cv2


def get_best_images(plate_images, num_img_return):
    """
    Get the top num_img_return quality images (with the least blur).
    Laplacian function returns a value which indicates how blur the image is.
    The lower the value, the more blur the image have
    """

    # first, pick the image with the largest area because the bigger the image, the bigger the characters on the plate
    if len(plate_images) > (num_img_return + 2):
        plate_images = sorted(plate_images, key=lambda x : x[0].shape[0]*x[0].shape[1], reverse=True)[:(num_img_return+2)]

    # secondly, pick the images with the least blur
    if len(plate_images) > num_img_return:
        plate_images = sorted(plate_images, key=lambda img : cv2.Laplacian(img[0], cv2.CV_64F).var(), reverse=True)[:num_img_return]
        # img[0] because plate_images = [plate image, char on plate]
    return plate_images