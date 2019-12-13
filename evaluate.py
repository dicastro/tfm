#! /usr/bin/env python

import argparse
import os
import numpy as np
import json
from annotations import parse_voc_annotation, parse_txt_annotation
import yolo_generator
import yolo_tiny_generator
from utils.utils import normalize, evaluate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model

def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    ###############################
    #   Create the validation generator
    ###############################

    if config['model']['data_load_method'] == 'voc':
        valid_ints, labels = parse_voc_annotation(
            config['valid']['valid_annot'], 
            config['valid']['valid_image_folder'], 
            config['valid']['cache_name'],
            config['model']['labels']
        )
    elif config['model']['data_load_method'] == 'txt':
        valid_ints, labels = parse_txt_annotation(
            config['valid']['valid_annot'], 
            config['valid']['valid_image_folder'], 
            config['valid']['cache_name'],
            config['model']['labels']
        )
    else:
        raise Exception('Unsupported data_load_method: \'{}\''.format(config['model']['data_load_method']))

    labels = labels.keys() if len(config['model']['labels']) == 0 else config['model']['labels']
    labels = sorted(labels)
    
    if not config['model']['type'] or config['model']['type'] == 'v3':
        print('Evaluating YOLOv3 model...')
        batch_generator = getattr(yolo_generator, 'BatchGenerator')
    elif config['model']['type'] == 'tiny':
        print('Evaluating YOLO Tiny model...')
        batch_generator = getattr(yolo_tiny_generator, 'BatchGenerator')

    valid_generator = batch_generator(
        instances           = valid_ints, 
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = 0,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.0, 
        norm                = normalize
    )

    ###############################
    #   Load the model and do evaluation
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']

    infer_model = load_model(config['train']['saved_weights_name'])

    # compute mAP for all the classes
    downsample = 32 # ratio between network input's size and network output's size, 32 for YOLOv3
    
    net_w, net_h = 416, 416

    if config['model']['min_input_size'] == config['model']['max_input_size']:
        net_w = config['model']['min_input_size']//downsample*downsample
        net_h = config['model']['min_input_size']//downsample*downsample
    
    nms_thresh = 0.45

    if config['valid']['duplicate_thresh']:
        nms_thresh = config['valid']['duplicate_thresh']

    average_precisions = evaluate(infer_model, valid_generator, net_w=net_w, net_h=net_h, obj_thresh=config['train']['ignore_thresh'], nms_thresh=nms_thresh)

    # print the score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))           

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')    
    
    args = argparser.parse_args()
    _main_(args)
