import numpy as np
import os
import cv2
import random


class Pipeline(object):
    def __init__(self,):
        super(Pipeline, self).__init__()
        self.flip_mirror = RandomFlip(flip_ratio=0.5,direction='horizontal')
        self.add_noise = AddNoise()
        self.random_effect = RandomEffect()
        self.random_effect2 = RandomEffect2()
        self.rotate_func = Rotate(angle_range=[-30, 30])
        self.sharpen_func = Sharpness()
        self.random_crop = RandomCrop(crop_ratio=[0.95,1.0])

    def __call__(self, img_cv2):
        img_cv2,_ = self.flip_mirror(img_cv2)
        if np.random.rand() < 0.3:
            img_cv2 = self.add_noise(img_cv2)
        if np.random.rand() < 0.3:
            img_cv2 = self.random_effect2(img_cv2)
        if np.random.rand() < 0.3:
            img_cv2 = self.random_effect(img_cv2)
        if np.random.rand() < 0.3:
            img_cv2, _ = self.rotate_func(img_cv2)
        if np.random.rand() < 0.2:
            img_cv2 = self.sharpen_func(img_cv2)

        return img_cv2



class RandomFlip(object):
    def __init__(self, flip_ratio=None, direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.direction = direction
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, img_cv2, mask=None):
        flip = True if np.random.rand() < self.flip_ratio else False
        if flip:
            if self.direction == 'horizontal':
                img_cv2 = np.flip(img_cv2, axis=1)#[left,right]
                if mask is not None:
                    mask = np.flip(mask, axis=1)
            else:
                img_cv2 = np.flip(img_cv2, axis=0)#[up down]
                if mask is not None:
                    mask = np.flip(mask, axis=0)

        return img_cv2, mask

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(self.flip_ratio)



class RandomCrop(object):
    def __init__(self, crop_ratio=[0.85, 0.95] ):
        self.crop_ratio = crop_ratio
        if crop_ratio is not None:
            assert min(crop_ratio) > 0 and max(crop_ratio) <= 1

    def __call__(self, img_cv2, mask=None, dst_sz=None):
        h, w = img_cv2.shape[0:2]
        ratio = np.random.uniform(low=min(self.crop_ratio), high=max(self.crop_ratio))
        new_h, new_w = int(h * ratio), int(w * ratio)
        x0 = np.random.randint(low=0, high=w - new_w)
        y0 = np.random.randint(low=0, high=h - new_h)
        img_cv2 = img_cv2[y0: y0 + new_h, x0:x0 + new_w]
        if dst_sz is not None:
            img_cv2 = cv2.resize(img_cv2,dst_sz)
        else:
            img_cv2 = cv2.resize(img_cv2, (w, h))
        if mask is not None:
            mask = mask[y0: y0 + new_h, x0:x0 + new_w]
            if dst_sz is not None:
                mask = cv2.resize(mask,dst_sz)
            else:
                mask = cv2.resize(mask, (w, h))
        return img_cv2, mask

    def __repr__(self):
        return self.__class__.__name__ + '(crop_ratio={})'.format(
            self.crop_ratio)


class AddNoise(object):
    def __init__(self):
        super(AddNoise, self).__init__()

    def _addSaltAndPepperNoise(self, src, percentage):
        SP_NoiseImg = src
        SP_NoiseNum = int(percentage * src.shape[0] * src.shape[1])
        for i in range(SP_NoiseNum):
            randX = random.randint(0, src.shape[0] - 1)
            randY = random.randint(0, src.shape[1] - 1)
            if random.randint(0, 1) == 0:
                SP_NoiseImg[randX, randY] = 0
            else:
                SP_NoiseImg[randX, randY] = 255
        return SP_NoiseImg

    def _addGaussianNoise(self, image, percentage):
        G_Noiseimg = image
        G_NoiseNum = int(percentage * image.shape[0] * image.shape[1])
        for i in range(G_NoiseNum):
            temp_x = np.random.randint(0, image.shape[0] - 1)
            temp_y = np.random.randint(0, image.shape[1] - 1)
            G_Noiseimg[temp_x][temp_y] = 255
        return G_Noiseimg

    def __call__(self, origin_img):
        if random.randint(0, 1):
            return self._addSaltAndPepperNoise(origin_img, percentage=random.uniform(0, 0.05))
        else:
            return self._addGaussianNoise(origin_img, percentage=random.uniform(0.01, 0.1))



class RandomEffect(object):
    def __init__(self):
        super(RandomEffect, self).__init__()

    def _change_contrast(self, origin_img, gamma=[0.8, 1.2]):
        img_data = origin_img.astype(np.float32) / 255.0
        assert len(img_data.shape) == 3
        for i in range(img_data.shape[-1]):
            img_data[:, :, i] = np.power(img_data[:, :, i], random.uniform(a=min(gamma), b=max(gamma))) * 255

        img_data = np.clip(img_data, a_min=0, a_max=255)
        output_img = img_data.astype(np.uint8)
        return output_img

    def _decrease_brightness(self, origin_img):
        weight = [0.1, 0.2, 0.3]
        background = [10, 30]
        blank = np.ones(origin_img.shape, origin_img.dtype) * random.randint(a=min(background), b=max(background))
        weight = random.choice(weight)
        output_img = cv2.addWeighted(origin_img, 1 - weight, blank, weight, 1.0)
        return output_img

    def _increase_brightness(self, origin_img):
        weight = [0.1, 0.2, 0.3]
        background = [200, 250]
        blank = np.ones(origin_img.shape, origin_img.dtype) * random.randint(a=min(background), b=max(background))
        weight = random.choice(weight)
        output_img = cv2.addWeighted(origin_img, 1 - weight, blank, weight, 1.0)
        return output_img

    def _add_warm_light(self, origin_img, intensity=0.1):
        # origin_img: GBR
        blank = np.zeros(origin_img.shape, origin_img.dtype)
        blank[:, :, :] = [100, 238, 247]
        output_img = cv2.addWeighted(origin_img, 1 - intensity, blank, intensity, 1.0)
        return output_img

    def _add_random_tone(self, origin_img, ratio=0.1):
        # origin_img: GBR
        warm_light = np.zeros(origin_img.shape, origin_img.dtype)
        warm_light[:, :, :] = [100, 238, 247]
        dark_blue = np.zeros(origin_img.shape, origin_img.dtype)
        dark_blue[:, :, :] = [102, 51, 0]
        orange = np.zeros(origin_img.shape, origin_img.dtype)
        orange[:, :, :] = [0, 102, 204]
        light_coffe = np.zeros(origin_img.shape, origin_img.dtype)
        light_coffe[:, :, :] = [102, 153, 204]
        dark_pink = np.zeros(origin_img.shape, origin_img.dtype)
        dark_pink[:, :, :] = [204, 153, 204]
        tone = random.choice([warm_light, dark_blue, orange, light_coffe, dark_pink])
        output_img = cv2.addWeighted(origin_img, 1 - ratio, tone, ratio, 1.0)
        return output_img

    def __call__(self, img_cv2):
        if np.random.uniform(low=0, high=1.0) <= 0.2:
            img_cv2 = self._change_contrast(img_cv2)
        if np.random.uniform(low=0, high=1.0) <= 0.1:
            img_cv2 = self._add_warm_light(img_cv2)
        if np.random.uniform(low=0, high=1.0) <= 0.1:
            img_cv2 = self._add_random_tone(img_cv2)
        if np.random.uniform(low=0, high=1.0) <= 0.1:
            if np.random.uniform(low=0, high=1.0) <= 0.5:
                img_cv2 = self._decrease_brightness(img_cv2)
            else:
                img_cv2 = self._increase_brightness(img_cv2)
        return img_cv2


class RandomEffect2(object):
    def __init__(self):
        super(RandomEffect2, self).__init__()

    def _erode_process(self, origin_img, ksize=(3, 3)):
        dist = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
        output_img = cv2.erode(origin_img, dist)
        return output_img

    def _dilate_process(self, origin_img, ksize=(3, 3)):
        dist = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
        dilation = cv2.dilate(origin_img, dist)
        return dilation

    def _blur_process(self, origin_img, ksize=(3, 3)):
        blur_img = cv2.blur(origin_img, ksize)
        return blur_img

    def __call__(self, img_cv2):
        if np.random.rand() < 0.1:
            img_cv2 = self._erode_process(img_cv2)
        if np.random.rand() < 0.1:
            img_cv2 = self._dilate_process(img_cv2)
        if np.random.rand() < 0.1:
            img_cv2 = self._blur_process(img_cv2)
        return img_cv2


class Rotate(object):
    def __init__(self, angle_range=[-10, 10]):
        self.angle_range = angle_range

    def _rotate(self, img_cv2, angle):
        if img_cv2 is not None:
            h, w = img_cv2.shape[0:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle=angle, scale=1.0)
            img_res = cv2.warpAffine(img_cv2, M, (w, h))
            return img_res
        else:
            return None

    def __call__(self, img_cv2, mask=None):
        angle = random.uniform(a=min(self.angle_range), b=max(self.angle_range))
        return self._rotate(img_cv2, angle), self._rotate(mask, angle)


class Sharpness(object):
    def __init__(self):
        super(Sharpness, self).__init__()

    def __call__(self, img_cv2):
        sharpen_op = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpen_img = cv2.filter2D(src=img_cv2, ddepth=cv2.CV_32F, kernel=sharpen_op)
        sharpen_img = cv2.convertScaleAbs(sharpen_img)
        return sharpen_img