import numpy as np
# import cv2
from PIL import ImageEnhance
from PIL import Image
from module.logger import logger
event_arg_dict={
    'event_20210819_cn_notclr':[3.3, 2.3, 1.05, 1],
    'event_20210819_cn_ff':[1.3, 1, 1.1, 1]
}

def enh_unfog(imageIN, 
            sharp_factor=1, color_factor=1, contrast_factor=1, brightness_factor=1, 
            event_name=''):
    if event_name:
        logger.info(event_name)
        sharp_factor = event_arg_dict.get(event_name)[0]
        color_factor = event_arg_dict.get(event_name)[1]
        contrast_factor = event_arg_dict.get(event_name)[2]
        brightness_factor = event_arg_dict.get(event_name)[3]

    imageIN = Image.fromarray(imageIN.astype('uint8')).convert('RGB')

    enh_shrp = ImageEnhance.Sharpness(imageIN)
    image_enh_1 = enh_shrp.enhance(sharp_factor)
    enh_col = ImageEnhance.Color(image_enh_1)
    image_enh_2 = enh_col.enhance(color_factor)
    enh_ctra = ImageEnhance.Contrast(image_enh_2)
    image_enh_3 = enh_ctra.enhance(contrast_factor)
    enh_brit = ImageEnhance.Brightness(image_enh_3)
    image_enh_4 = enh_brit.enhance(brightness_factor)


    image_enh_done = np.array(image_enh_4)
    #
    # cv2.imshow('img', image_enh_done)
    # cv2.imwrite('E:/AzurLaneAutoScript/screenshots/unfog3.png', image_enh_done)
    # cv2.waitKey()
    #
    return image_enh_done
