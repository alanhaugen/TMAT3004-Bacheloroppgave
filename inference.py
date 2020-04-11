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
    
    i = 0
    while video.isOpened():
        ret, frame = video.read()
        
        if ret:
            outputs = predictor(frame)

            #print(outputs)
            v = Visualizer(frame[:, :, ::-1],
                           metadata=test_metadata, 
                           scale=0.8
            )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            plt.imsave('anpd_out/frame_{}.png'.format(str(i).zfill(3)), v.get_image())
            #output_file.write(v.get_image())
            i += 1
        else:
            break
    
    img_array = []
    for iterator in range(0, i):
        img = cv2.imread('outputs/frame_{}.png'.format(str(iterator).zfill(3)))
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc(*'DIVX'), frames_per_second, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
    video.release()
    #output_file.release()
    
    return

if __name__ == "__main__":
    # detectron2 configuration

    train_data_name = 'fish_train'
    test_data_name  = 'fish_test'

    thing_classes = ['atlantic_cod', 'seithe']

    output_dir = 'outputs'

    # default confugration
    cfg = get_cfg()

    # update configuration with RetinaNet configuration
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))

    # We have registered the train and test data set with name traffic_sign_train and traffic_sign_test. 

    # No metric implemented for the test dataset, we will have to update cfg.DATASET.TEST with empty tuple
    cfg.DATASETS.TEST = ()

    # data loader configuration
    cfg.DATALOADER.NUM_WORKERS = 4

    # Update model URL in detectron2 config file
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")

    # number of output class
    # we have only one class that is Traffic Sign
    cfg.MODEL.RETINANET.NUM_CLASSES = len(thing_classes)

    # update create ouptput directory 
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

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
    inference_on_dataset(trainer.model, val_loader, evaluator)

    video_read_write('in.mp4')
