import os
import cv2
import shutil
import xml.etree.ElementTree as ET
names2id = {'car': 0, 'van': 1, 'others': 2, 'bus': 3}

def convert_detrac2coco(imgs_path, xml_path, save_path, task):
    dir_list = os.listdir(imgs_path)
    txt_save_path = os.path.join(save_path, task + '.txt')
    with open(txt_save_path, 'w') as txt:
        for dir_idx, each_dir in enumerate(dir_list):
            print(dir_idx, each_dir)
            each_xml = os.path.join(xml_path, each_dir + '.xml')
            xml_tree = ET.parse(each_xml)
            xml_root = xml_tree.getroot()
            for child in xml_root:
                if child.tag == 'frame':
                    pic_id = child.attrib['num'].zfill(5)
                    pic_path = os.path.join(imgs_path, each_dir, 'img' + pic_id + '.jpg')
                    img = cv2.imread(pic_path)
                    img_shape = img.shape  # height, width, channel
                    target_list = list(child)[0]

                    pic_save_path = os.path.join(save_path, 'images', task, each_dir + '_' + 'img' + pic_id + '.jpg')
                    if not os.path.exists(os.path.dirname(pic_save_path)):
                        os.makedirs(os.path.dirname(pic_save_path))
                    label_save_path = os.path.join(save_path, 'labels', task, each_dir + '_' + 'img' + pic_id + '.txt')
                    if not os.path.exists(os.path.dirname(label_save_path)):
                        os.makedirs(os.path.dirname(label_save_path))
                    shutil.copy(pic_path, pic_save_path)
                    txt.writelines(pic_save_path + '\n')

                    with open(label_save_path, 'w') as f:
                        for target in target_list:
                            box, attribute = target[0], target[1]
                            x = round(float(box.attrib['left'])/img_shape[1], 6)
                            y = round(float(box.attrib['top'])/img_shape[0], 6)
                            w = round(float(box.attrib['width'])/img_shape[1], 6)
                            h = round(float(box.attrib['height'])/img_shape[0], 6)
                            vehicle = attribute.attrib['vehicle_type']
                            log = '{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(names2id[vehicle], x, y, w, h)
                            f.writelines(log + '\n')

if __name__ == '__main__':

    detrac_path = '/mnt/ActionRecog/dataset/DETRAC/Insight-MVT_Annotation_Train/'
    detrac_xml_path = '/mnt/ActionRecog/dataset/DETRAC/DETRAC-Train-Annotations-XML/'
    coco_path = '/mnt/ActionRecog/dataset/DETRAC_coco/'

    convert_detrac2coco(imgs_path=detrac_path, xml_path = detrac_xml_path, save_path=coco_path, task='train')