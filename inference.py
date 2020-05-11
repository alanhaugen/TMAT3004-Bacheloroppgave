import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import os
import matplotlib.pyplot as plt

# model_zoo has a lots of pre-trained model
from detectron2 import model_zoo

# DefaultTrainer is a class for training object detector
from detectron2.engine import DefaultTrainer
# DefaultPredictor is class for inference
from detectron2.engine import DefaultPredictor

# detectron2 has its configuration format
from detectron2.config import get_cfg
# detectron2 has implemented Visualizer of object detection
from detectron2.utils.visualizer import Visualizer

# from DatasetCatalog, detectron2 gets dataset and from MetadatCatalog it gets metadata of the dataset
from detectron2.data import DatasetCatalog, MetadataCatalog

# BoxMode support bounding boxes in different format
from detectron2.structures import BoxMode

# COCOEvaluator based on COCO evaluation metric, inference_on_dataset is used for evaluation for a given metric
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# build_detection_test_loader, used to create test loader for evaluation
from detectron2.data import build_detection_test_loader

def video_read_write(video_path):
    """
    Read video frames one-by-one, flip it, and write in the other video.
    video_path (str): path/to/video
    """
    video = cv2.VideoCapture(video_path)
    
    # Check if camera opened successfully
    if not video.isOpened(): 
        print("Error opening video file")
        return
    
    # create video writer
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    isStreamOpen = False
    while video.isOpened():
        ret, frame = video.read()
        
        if ret:
            outputs = predictor(frame)

            #print(outputs)
            v = Visualizer(frame[:, :, ::-1],
                           metadata=test_metadata, 
                           scale=0.8
            )

            instances = outputs["instances"].to("cpu")

            v = v.draw_instance_predictions(instances)

            plt.imsave('outputs/frame_intermediate.png', v.get_image())

            if isStreamOpen == False:
                img = cv2.imread('outputs/frame_intermediate.png')
                height, width, layers = img.shape
                size = (width,height)
                out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc(*'DIVX'), frames_per_second, size)
                isStreamOpen = True

            img = cv2.imread('outputs/frame_intermediate.png')

            cv2.putText(img, 'num instances: ' + str(len(instances)), (5,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
            out.write(img)

            print ('num instances: ' + str(len(instances)))
        else:
            break
    
    out.release()
    video.release()
    
    return

# write a function that loads the dataset into detectron2's standard format
def get_fish_dicts(data_root, txt_file):
    dataset_dicts = []
    filenames = []
    csv_path = os.path.join(data_root, txt_file)
    with open(csv_path, "r") as f:
        for line in f:
            filenames.append(line.rstrip())

    for idx, filename in enumerate(filenames):
        record = {}

        image_path = os.path.join(data_root, filename)

        height, width = cv2.imread(image_path).shape[:2]

        record['file_name'] = image_path
        record['image_id'] = idx
        record['height'] = height
        record['width'] = width

        image_filename = os.path.basename(filename)
        image_name = os.path.splitext(image_filename)[0]
        annotation_path = os.path.join(data_root, 'labels', '{}.txt'.format(image_name))
        annotation_rows = []

        with open(annotation_path, "r") as f:
            for line in f:
                temp = line.rstrip().split(" ")
                annotation_rows.append(temp)

        objs = []
        for row in annotation_rows:
            xmin = int(float(row[1]))
            ymin = int(float(row[2]))
            xmax = int(float(row[3]))
            ymax = int(float(row[4]))

            obj= {
                'bbox': [xmin, ymin, xmax, ymax],
                'bbox_mode': BoxMode.XYXY_ABS,
                # alternatively, we can use bbox_mode = BoxMode.XYWH_ABS
                # 'bbox': [xmin, ymin, bwidth, bheight],
                # 'bbox_mode': BoxMode.XYWH_ABS,
                'category_id': int(row[0]),
                'iscrowd': 0
            }

            objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts

if __name__ == "__main__":
    # configurations

    data_root = 'data'
    train_txt = 'fish_train.txt'
    test_txt  = 'fish_test.txt'

    train_data_name = 'fish_train'
    test_data_name  = 'fish_test'

    thing_classes = ['atlantic_cod', 'saithe']

    output_dir = 'outputs'

    def count_lines(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    train_img_count = count_lines(os.path.join(data_root, train_txt))

    # Register train and test data
    # dataset can be registered only once with one name

    # register train data
    DatasetCatalog.register(name=train_data_name,
                            func=lambda: get_fish_dicts(data_root, train_txt))
    train_metadata = MetadataCatalog.get(train_data_name).set(thing_classes=thing_classes)

    # register test data
    DatasetCatalog.register(name=test_data_name,
                            func=lambda: get_fish_dicts(data_root, test_txt))
    test_metadata = MetadataCatalog.get(test_data_name).set(thing_classes=thing_classes)

    # lets visualize the data

    test_data_dict = get_fish_dicts(data_root, test_txt)

    for d in random.sample(test_data_dict, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1],
                                metadata=test_metadata,
                                scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        plt.figure(figsize = (12, 12))
        plt.imshow(vis.get_image())
        plt.show()

    # detectron2 configuration

    # default confugration
    cfg = get_cfg()

    # Do inference on CPU
    #cfg.MODEL.DEVICE='cpu'

    # update configuration with RetinaNet configuration
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # We have registered the train and test data set with name traffic_sign_train and traffic_sign_test.
    # Let's replace the detectron2 default train dataset with our train dataset.
    cfg.DATASETS.TRAIN = (train_data_name,)

    # No metric implemented for the test dataset, we will have to update cfg.DATASET.TEST with empty tuple
    cfg.DATASETS.TEST = ()

    # data loader configuration
    cfg.DATALOADER.NUM_WORKERS = 4

    # Update model URL in detectron2 config file
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # slover configuration

    # batch size
    cfg.SOLVER.IMS_PER_BATCH = 4

    # choose a good learning rate
    cfg.SOLVER.BASE_LR = 0.001

    # We need to specify the number of iteration for training in detectron2, not the number of epochs.
    # lets convert number of epoch to number or iteration (max iteration)

    epoch = 20
    max_iter = int(epoch * train_img_count / cfg.SOLVER.IMS_PER_BATCH)
    max_iter = 500

    cfg.SOLVER.MAX_ITER = max_iter

    # number of output class
    # we have only one class that is Traffic Sign
    cfg.MODEL.RETINANET.NUM_CLASSES = len(thing_classes)

    # update create ouptput directory
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # training

    # Create a trainer instance with the configuration.
    trainer = DefaultTrainer(cfg)

    # if rseume=False, because we don't have trained model yet. It will download model from model url and load it
    trainer.resume_or_load(resume=False)

    # start training
    #trainer.train()

    # inference on our fine-tuned model

    # By default detectron2 save the model with name model_final.pth
    # update the model path in configuration that will be used to load the model
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")


    # update RetinaNet score threshold
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5

    cfg.DATASETS.TEST = (test_data_name,)

    # create a predictor instance with the configuration (it has our fine-tuned model)
    # this predictor does prdiction on a single image
    predictor = DefaultPredictor(cfg)

    # create directory for evaluation
    eval_dir = os.path.join(cfg.OUTPUT_DIR, 'coco_eval')
    os.makedirs(eval_dir, exist_ok=True)

    # create evaluator instance with coco evaluator
    evaluator = COCOEvaluator(dataset_name=test_data_name,
                              cfg=cfg,
                              distributed=False,
                              output_dir=eval_dir)

    # create validation data loader
    val_loader = build_detection_test_loader(cfg, test_data_name)

    # start validation
    #inference_on_dataset(trainer.model, val_loader, evaluator)

    # Run inference on video
    video_read_write('in.mp4')
