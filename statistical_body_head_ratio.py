import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import os
import numpy as np
import cv2 as cv

def gatherBodyBoxes(path):
    skipped = 0
    file = open(path, 'r')
    body_boxes = []
    head_boxes = []
    lines = file.readlines()
    for line in lines:
        parts = (line.strip().split('\t'))
        img_path = parts[0]
        parts = parts[1:]
        counter = 0
        hasBody = False
        body_box = np.array([0, 0, 0, 0])
        head_box = np.array([0, 0, 0, 0])
        print('img_path', img_path)
        image = cv.imread(img_path)
        height, width, channels = image.shape
        for p in parts:
            if counter in range(1, 5):
                head_box[(counter-1) % 4] = int(p)
            if counter in range(5, 9):
                body_box[(counter-1) % 4] = int(p)
            if counter == 0 and p == '1':
                hasBody = True
            if hasBody and counter == 8:
                # only save body if it has a body

                if body_box[3] < height:
                    body_boxes.append(np.asarray(body_box))
                    head_boxes.append(np.asarray(head_box))
                else:
                    skipped += 1

                counter = 0
                continue
            elif counter == 4 and not hasBody:
                counter = 0
                continue
            counter += 1
    return head_boxes, body_boxes, skipped

def computeRatios(head_boxes, body_boxes, out_path, skipped):
    assert len(head_boxes) == len(body_boxes)
    mean_ratios = np.array([0.0, 0.0, 0.0, 0.0])
    width_ratios = []
    height_ratios = []
    denom = 0
    for i in range(len(head_boxes)):
        head_box = head_boxes[i]
        body_box = body_boxes[i]
        h_width =  head_box[2] - head_box[0]
        h_height = head_box[3] - head_box[1]
        h_ctr_x =  head_box[0] + 0.5*h_width
        h_ctr_y = head_box[1] + 0.5*h_height

        b_width =  body_box[2] - body_box[0]
        b_height = body_box[3] - body_box[1]
        b_ctr_x =  body_box[0] + 0.5*b_width
        b_ctr_y =  body_box[1] + 0.5*b_height

        # check validity
        if h_width <= 0 or h_height <= 0 or b_width <=0 or b_height <= 0:
            continue

        width_ratio = b_width / h_width
        height_ratio = b_height / h_height
        center_x_width_ratio =  (np.sign(b_ctr_x - h_ctr_x) * float(np.abs(b_ctr_x - h_ctr_x) + 1.0)) / b_width
        y_width = (np.sign(b_ctr_y - h_ctr_y) * float(np.abs(b_ctr_y - h_ctr_y + 1.0)))
        if y_width == 0:
            y_width = 1.0
        center_y_height_ratio =  b_height / y_width

        mean_ratios[0] += width_ratio
        mean_ratios[1] += height_ratio
        mean_ratios[2] += center_x_width_ratio
        mean_ratios[3] += center_y_height_ratio
        denom += 1
        width_ratios.append(width_ratio)
        height_ratios.append(height_ratio)
    with open(out_path, 'w') as f:
        final_width_ratio = mean_ratios[0] / denom
        final_height_ratio = mean_ratios[1] / denom
        final_ctr_x_ratio = mean_ratios[2] / denom
        final_ctr_y_ratio = mean_ratios[3] / denom
        f.write('width_ratio\theight_ratio\tctr_x_ratio\tctr_y_ratio\tskipped\n')
        print((str(final_width_ratio) + '\t' + str(final_height_ratio) + '\t' + str(final_ctr_x_ratio) + '\t' + str(final_ctr_y_ratio) + '\t'+  str(skipped) +'\n'))
        f.write((str(final_width_ratio) + '\t' + str(final_height_ratio) + '\t' + str(final_ctr_x_ratio) + '\t' + str(final_ctr_y_ratio) + '\t'+  str(skipped) +'\n'))
    return width_ratios, height_ratios

def parseArgs(argv=None):
    parser = argparse.ArgumentParser(
        description='Statistical head body ratio')
    parser.add_argument('--csv', type=str, default='film1.csv',
                        help='Path to csv file containing head and body bounding boxes', required=False)
    parser.add_argument('--outFile', type=str, default='results/ratio/shbr.txt',
                        help='Path to output file', required=False)

    global args
    args = parser.parse_args(argv)

if __name__ == '__main__':
    parseArgs()
    out_path = args.outFile
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    head_boxes, body_boxes, skipped = gatherBodyBoxes(args.csv)
    width_ratios, height_ratios = computeRatios(head_boxes, body_boxes, out_path, skipped)
    print(np.mean(width_ratios), np.mean(height_ratios), skipped)

