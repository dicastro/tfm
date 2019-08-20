import numpy as np
import os
import xml.etree.ElementTree as ET
import pickle
import cv2

def parse_voc_annotation(ann_dir, img_dir, cache_name, labels=[]):
    if os.path.exists(cache_name):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    else:
        all_insts = []
        seen_labels = {}
        
        for ann in sorted(os.listdir(ann_dir)):
            img = {'object':[]}

            try:
                tree = ET.parse(ann_dir + ann)
            except Exception as e:
                print(e)
                print('Ignore this bad annotation: ' + ann_dir + ann)
                continue
            
            for elem in tree.iter():
                if 'filename' in elem.tag:
                    img['filename'] = img_dir + elem.text
                if 'width' in elem.tag:
                    img['width'] = int(elem.text)
                if 'height' in elem.tag:
                    img['height'] = int(elem.text)
                if 'object' in elem.tag or 'part' in elem.tag:
                    obj = {}
                    
                    for attr in list(elem):
                        if 'name' in attr.tag:
                            obj['name'] = attr.text

                            if obj['name'] in seen_labels:
                                seen_labels[obj['name']] += 1
                            else:
                                seen_labels[obj['name']] = 1
                            
                            if len(labels) > 0 and obj['name'] not in labels:
                                break
                            else:
                                img['object'] += [obj]
                                
                        if 'bndbox' in attr.tag:
                            for dim in list(attr):
                                if 'xmin' in dim.tag:
                                    obj['xmin'] = int(round(float(dim.text)))
                                if 'ymin' in dim.tag:
                                    obj['ymin'] = int(round(float(dim.text)))
                                if 'xmax' in dim.tag:
                                    obj['xmax'] = int(round(float(dim.text)))
                                if 'ymax' in dim.tag:
                                    obj['ymax'] = int(round(float(dim.text)))

            if len(img['object']) > 0:
                all_insts += [img]

        cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
        with open(cache_name, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)    
                        
    return all_insts, seen_labels

def parse_txt_annotation(ann_txt, img_dir, cache_name, labels=[]):
    if os.path.exists(cache_name):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    else:
        all_insts = []
        seen_labels = {}

        with open(ann_txt) as annotations_file:
            annotation_lines = annotations_file.readlines()

            all_img_defs = []
            
            for i_line, annotation_line in enumerate(annotation_lines):
                annotation_line = annotation_line.strip()
                
                annotation_parts = annotation_line.split(' ')
                
                objects = []
                
                for bbox in annotation_parts[1:]:
                    bbox_parts = bbox.split(',')
                    
                    label_index = int(bbox_parts[4])
                    if label_index >= len(labels):
                        print('WARN: There is an annotation with a label index ({}) not defined in current labels: {}'.format())
                        break
                    
                    label = labels[label_index]

                    if label in seen_labels:
                        seen_labels[label] += 1
                    else:
                        seen_labels[label] = 1

                    object = {
                        'name': label,
                        'xmin': int(bbox_parts[0]),
                        'ymin': int(bbox_parts[1]),
                        'xmax': int(bbox_parts[0]) + int(bbox_parts[2]),
                        'ymax': int(bbox_parts[1]) + int(bbox_parts[3])
                    }
                    
                    objects.append(object)
                
                image = cv2.imread(annotation_parts[0])
                
                img_def = {
                    'filename': annotation_parts[0],
                    'height': image.shape[0],
                    'width': image.shape[1],
                    'object': objects
                }
                
                all_img_defs.append(img_def)

        cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
        with open(cache_name, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_insts, seen_labels
