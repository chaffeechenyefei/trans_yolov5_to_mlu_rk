import cv2
import numpy as np


class Image_Stitching():
    def __init__(self):
        self.ratio = 0.5
        self.min_match = 10
        self.sift = cv2.SIFT_create()
        self.smoothing_window_size = 10
        self.first = True
        self.H = np.array([[ 9.61801270e-01, -5.69017395e-01,  1.32273162e+03],
                           [ 4.96410389e-01,  8.99407063e-01, -2.39338235e+02],
                           [ 2.93268792e-05, -3.20408643e-05,  1.00000000e+00]])

    def registration(self, img1, img2):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches = []
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])

        if len(good_points) > self.min_match:
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
        return H

    def create_mask(self, img1, img2, version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1 + height_img1
        width_panorama = width_img1 + width_img2
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if version == 'left_image':
            # pass
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    def blending(self, img1, img2):
        # if self.first:
        #     self.H = self.registration(img1, img2)
        #     self.first = False
        # print(self.H)19041967=63
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1 + height_img1
        width_panorama = width_img1 + width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self.create_mask(img1, img2, version='left_image')
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1
        mask2 = self.create_mask(img1, img2, version='right_image')
        panorama2 = cv2.warpPerspective(img2, self.H, (width_panorama, height_panorama)) * mask2
        print(panorama1.shape, panorama2.shape)
        td_off = 30
        panorama2[:2160-td_off, :] = panorama2[td_off:2160, :]
        # lr_off = 63
        # panorama2[:, :3480-lr_off] = panorama2[:, lr_off:3480]
        # panorama2[:, 1920-lr_off:1920] = 0
        result = panorama1 + panorama2
        # result[:1080, :1920] = panorama1[:1080, :1920]

        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]

        b_channel, g_channel, r_channel = cv2.split(final_result)
        rows, cols = np.where(np.sum(final_result, axis=-1) != 0)
        alpha_channel = np.zeros([final_result.shape[0], final_result.shape[1]], dtype=b_channel.dtype)
        alpha_channel[rows, cols] = 255

        png_result = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        return png_result


def main():
    img1 = cv2.imread('/mnt/ActionRecog/dataset/car_count/shangqi_dataset/Camera_1.jpg')
    img2 = cv2.imread('/mnt/ActionRecog/dataset/car_count/shangqi_dataset/Camera_2.jpg')
    print(img1.shape, img2.shape)
    final = Image_Stitching().blending(img1, img2)
    cv2.imwrite('/mnt/ActionRecog/dataset/car_count/shangqi_dataset/0_merge.png', final)


if __name__ == '__main__':
    main()