import numpy as np
import cv2
def specify_region(img, mode='cam1'):
    region_dict = {'cam1': np.array([[[0, 350], [921, 106], [1417, 156], [1440, 325],
                                 [1918, 441], [1920, 1080], [0, 1080]]], dtype = np.int32),
                   'cam2': np.array([[[895, 307], [1685, 211], [1920, 320], [1920, 1080], [1200, 1080]]], dtype = np.int32)}
    region = region_dict[mode]
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, region, 1)
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return img*mask

img = cv2.imread('/mnt/ActionRecog/dataset/car_count/shangqi_dataset/0_cam2.jpg')
img = specify_region(img, mode='cam2')
cv2.imwrite('/mnt/ActionRecog/dataset/car_count/shangqi_dataset/0_cam2_re.jpg', img)

