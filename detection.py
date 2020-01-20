import torch, torchvision
import os
import argparse
# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator

def convert(boxes, classes, class_names):
    boxes = boxes.tensor.cpu().numpy()

    labels = [class_names[i] for i in classes]
    return boxes, labels

def filter_by_label(boxes, labels, filter_label):
    num_instances = len(boxes)
    ret_boxes = []
    ret_labels = []
    for i in range(num_instances):
        if labels[i] == filter_label:
            ret_boxes.append(boxes[i])
            ret_labels.append(labels[i])
    for i in range(len(ret_boxes)):
        ret_boxes[i] = [int(c) for c in ret_boxes[i]]
    return ret_boxes, ret_labels






def parseArgs(argv=None):
    parser = argparse.ArgumentParser(
        description='detectron2 bounding box extractor')
    parser.add_argument('-l', '--list', default='/home/bjenei/corpus/head_detection_sets/train.csv', type=str,
                        help='Path to csv containing image paths', required=False)
    parser.add_argument('-o', '--outfile', default='detectron2_extract/extracted.csv', type=str,
                        help='Path to output file', required=False)
    parser.add_argument('-i', '--imagedir', default=None, type=str,
                    help='Path to directory with output visualized images, if None, no actual visualization takes place.', required=False)
    

    global args
    args = parser.parse_args(argv)

if __name__ == '__main__':
    parseArgs()
    if not os.path.exists(os.path.dirname(args.outfile)):
        os.makedirs(os.path.dirname(args.outfile))
    visualize = args.imagedir != None
    if visualize and not os.path.exists(args.imagedir):
        os.makedirs(args.imagedir)

    print('Loading model...')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"
    predictor = DefaultPredictor(cfg)
    
    n_files = sum (1 for _ in open(args.list))
    in_file = open(args.list, 'r')
    out_file = open(args.outfile, 'w')

    print('Inference...')
    counter = 0

    for line in in_file:
        counter += 1
        img_path = line.strip().split('\t')[0]
        print('[', counter, '/', n_files,']', img_path)
        im = cv2.imread(img_path)

        outputs = predictor(im)

        class_names =  MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes", None)
        boxes = outputs["instances"].pred_boxes
        classes = outputs["instances"].pred_classes

        boxes, labels = convert(boxes, classes, class_names)
        boxes, labels = filter_by_label(boxes, labels, 'person')
        
        flat_boxes = [str(c) for box in boxes for c in box]
        text = []
        text.append(img_path)
        text.extend(flat_boxes)
        text = '\t'.join(text) + '\n'
        out_file.write(text)
        if visualize:
            out_image_path = os.path.join(args.imagedir, os.path.basename(img_path))
            print(out_image_path)
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite(out_image_path, v.get_image()[:, :, ::-1])
        
    out_file.close()
    in_file.close()
