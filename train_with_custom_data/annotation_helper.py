import json
from dicttoxml import dicttoxml
from collections import OrderedDict
import os
from xml.dom.minidom import parseString
import numpy as np
import cPickle
from random import shuffle


def bbox_json_to_xml(taskname, input_dir, output_dir):
    """
    Change labelme object bounding box in json format to affordancenet required xml format.
    Currently only support single object in an image.
    """

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            with open(os.path.join(input_dir, filename), "r") as fh:
                data = json.load(fh)
            x_min, y_min = data['shapes'][0]['points'][0]
            x_max, y_max = data['shapes'][0]['points'][1]
            # print(x_min, y_min, x_max, y_max)
            object_name = data['shapes'][0]['label']
            # print(object_name)
            width = data['imageWidth']
            height = data['imageHeight']
            depth = 3
            # print(width, height, depth)

            file_id = filename.split(".")[0]

            dict = OrderedDict()
            dict["folder"] = taskname
            dict["filename"] = "{}.jpg".format(file_id)
            dict["size"] = OrderedDict()
            dict["size"]["width"] = width
            dict["size"]["height"] = height
            dict["size"]["depth"] = depth
            dict["object"] = OrderedDict()
            dict["object"]["name"] = object_name
            dict["object"]["pose"] = "Upspecified"
            dict["object"]["truncated"] = 0
            dict["object"]["difficult"] = 0
            dict["object"]["bndbox"] = OrderedDict()
            dict["object"]["bndbox"]["xmin"] = int(x_min)
            dict["object"]["bndbox"]["ymin"] = int(y_min)
            dict["object"]["bndbox"]["xmax"] = int(x_max)
            dict["object"]["bndbox"]["ymax"] = int(y_max)

            xml = dicttoxml(dict, custom_root='annotation', attr_type=False)
            xml_filename = "{}.xml".format(file_id)
            xml_filename = os.path.join(output_dir, xml_filename)
            with open(xml_filename, "w") as fout:
                # use dom to beautify xml
                dom = parseString(xml)
                xml = dom.toprettyxml()
                # remove first line
                xml = xml.split("?>\n")[1]
                fout.write(xml)


def mask_npy_to_sm(input_dir, output_dir):
    """
    Change labelme affordance mask in npy format to affordancenet required sm format.
    Currently only support single object in an image.
    """

    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            data = np.load(os.path.join(input_dir, filename))
            # change to uint8 for affordance-net
            data = data.astype(np.uint8)
            img_name = filename.split(".")[0]
            # only support 1 object per image for now
            sm_filename = "{}_{}_segmask.sm".format(img_name, 1)

            print(sm_filename)
            ## write each mask to sm file
            with open(os.path.join(output_dir, sm_filename), 'wb') as fout:
                cPickle.dump(data, fout, cPickle.HIGHEST_PROTOCOL)


def write_train_test_files(output_dir):
    with open(os.path.join(output_dir, "train.txt"), "w") as fh:
        ids = list(range(1, 37)) + list(range(55, 91))
        shuffle(ids)
        img_names = ["{}".format(id) for id in ids]
        fh.write("\n".join(img_names))

    with open(os.path.join(output_dir, "test.txt"), "w") as fh:
        ids = list(range(1, 37)) + list(range(55, 91))
        shuffle(ids)
        img_names = ["{}".format(id) for id in ids]
        fh.write("\n".join(img_names))


if __name__ == "__main__":
    bbox_json_to_xml("SRTASK",
                     input_dir="/home/weiyu/Research/labelme/labelme/examples/semantic_segmentation/custom_data/bbox/json",
                     output_dir="/home/weiyu/Research/labelme/labelme/examples/semantic_segmentation/custom_data/bbox/xml")

    # mask_npy_to_sm(input_dir="/home/weiyu/Research/labelme/labelme/examples/semantic_segmentation/custom_data/masks/SegmentationClass",
    #                output_dir="/home/weiyu/Research/labelme/labelme/examples/semantic_segmentation/custom_data/masks/SegmentationMask")

    # write_train_test_files("/home/weiyu/Research/AffordanceNet/affordance-net/data/VOCdevkit2012/VOC2012/ImageSets/Main")

    # input_dir = "/home/weiyu/Research/labelme/labelme/examples/semantic_segmentation/custom_data/masks/SegmentationClass"
    # for filename in os.listdir(input_dir):
    #     new_filename = str(int(filename[5:].split(".")[0])) + "." + filename.split(".")[1]
    #     os.rename(os.path.join(input_dir, filename), os.path.join(input_dir, new_filename))