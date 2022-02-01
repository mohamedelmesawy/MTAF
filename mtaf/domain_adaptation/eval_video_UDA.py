from logging import raiseExceptions
import os.path as osp
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from mtaf.utils.func import per_class_iu, fast_hist
from mtaf.utils.serialization import pickle_dump, pickle_load

import cv2
import IPython
from io import BytesIO
from PIL import Image
from torchvision import transforms as t

from six.moves import urllib

import os
import moviepy.video.io.ImageSequenceClip
import glob

def create_label_colormap(no_class=7, dataset=None):
    """Creates a label colormap used in Cityscapes segmentation benchmark.
    Returns:
        A Colormap for visualizing segmentation results.
    """
    # GTA 19 Classes
    if  (dataset == 'GTA') or ( no_class == 19) :
        colormap = np.array([
            # COLOR           # Index in the Original Cityscape
            [128,  64,  128],   # 7,    # road
            [244,  35,  232],   # 8,    # sidewalk
            [70,   70,   70],   # 11,   # building
            [102,  102, 156],   # 12,   # wall
            [190,  153, 153],   # 13,   # fence
            [153,  153, 153],   # 17,   # pole
            [250,  170,  30],   # 19,   # traffic light
            [220,  220,   0],   # 20,   # traffic sign
            [107,  142,  35],   # 21,   # vegetation
            [152,  251, 152],   # 22,   # terrain
            [70,   130, 180],   # 23,   # sky - RAM Modified
            [220,   20,  60],   # 24,   # person
            [255,    0,   0],   # 25,   # rider
            [0,      0, 142],   # 26,   # car
            [0,      0,  70],   # 27,   # truck
            [0,     60, 100],   # 28,   # bus
            [0,     80, 100],   # 31,   # train
            [0,      0, 230],   # 32,   # motorcycle
            [119,   11,  32],   # 33,   # bicycle
            # 0,    # void [Background] is the last 20th (counting from 1) class.
            [0,      0,   0],
        ], dtype=np.uint8)
    elif (dataset == 'CITYSCAPES') or (no_class == 35):
        # Cityscape   35 Classes
        colormap = np.array([
            #  Color 35 Class    # Id/Index in Cityscape
            [  0,   0,   0],    # 0
            [  0,   0,   0],    # 1
            [  0,   0,   0],    # 2
            [  0,   0,   0],    # 3
            [  0,   0,   0],    # 4
            [111,  74,   0],    # 5
            [ 81,   0,  81],    # 6
            [128,  64, 128],    # 7
            [244,  35, 232],    # 8
            [250, 170, 160],    # 9
            [230, 150, 140],    # 10
            [ 70,  70,  70],    # 11
            [102, 102, 156],    # 12
            [190, 153, 153],    # 13
            [180, 165, 180],    # 14
            [150, 100, 100],    # 15
            [150, 120,  90],    # 16
            [153, 153, 153],    # 17
            [153, 153, 153],    # 18
            [250, 170,  30],    # 19
            [220, 220,   0],    # 20
            [107, 142,  35],    # 21
            [152, 251, 152],    # 22
            [ 70, 130, 180],    # 23
            [220,  20,  60],    # 24
            [255,   0,   0],    # 25
            [  0,   0, 142],    # 26
            [  0,   0,  70],    # 27
            [  0,  60, 100],    # 28
            [  0,   0,  90],    # 29
            [  0,   0, 110],    # 30
            [  0,  80, 100],    # 31
            [  0,   0, 230],    # 32
            [119,  11,  32],    # 33
            [  0,   0, 142],    # 34
    ], dtype=np.uint8)

    elif no_class == 7:
      colormap = np.array([
      [128,  64, 128],       # 0
      [ 70,  70,  70],       # 1
      [153, 153, 153],       # 2
      [107, 142,  35],       # 3
      [ 70, 130, 180],       # 4
      [220,  20,  60],       # 5
      [  0,   0, 142],       # 6
      [  0,   0,   0],       # 7
      
      ], dtype=np.uint8)

    return colormap

def label_to_color_image(label):
    label = np.where(label==255, 7,label)
    colormap = create_label_colormap()  # By default no_class = 7
    return colormap[label]

################ RAM video fns ######################
def video_segmentation_from_url(video_url, SAMPLE_VIDEO, num_frames, model, cfg):
  if not osp.isfile(SAMPLE_VIDEO):
      print('downloading the sample video...')
      SAMPLE_VIDEO = urllib.request.urlretrieve(video_url)[0]
  print('running deeplab on the sample video...')
  video = cv2.VideoCapture(SAMPLE_VIDEO)
  
  try:
      for i in range(num_frames):
          _, frame = video.read()
          if not _: break
          original_im = Image.fromarray(frame[..., ::-1])
          run_segmentation_video_frame(SAMPLE_VIDEO, original_im, i, model, cfg)
          # print(len(cfg.TEST.MODEL))
          IPython.display.clear_output(wait=True)
  except KeyboardInterrupt:
      plt.close()
      print("Stream stopped.")

def video_segmentation_from_sequence_images(directory_path, video_name, model, cfg):
  directory_path = directory_path +'/*png'
  video_images = glob.glob(directory_path)
  directory_path.sort()
  if len(video_images) < 1 :
    return
  try:
      i = 0
      for image in video_images:
        image = Image.open(image)
        run_segmentation_video_frame(video_name, image, i, model, cfg)
        i += 1
  except KeyboardInterrupt:
      plt.close()
      print("Stream stopped.")


def run_segmentation_video_frame(video_name, original_im, index, model, cfg):
    """Inferences DeepLab model on a video file and stream the visualization."""
    image = preprocess_cityscapes_video_fram(original_im)
    tensor = torch.from_numpy(np.flip(image,axis=0).copy()).unsqueeze(0)
    device = cfg.GPU_ID
    if cfg.TEST.MODEL[0] == 'DeepLabv2':
        _, pred_main = model(tensor.cuda(device))
    elif cfg.TEST.MODEL[0] == 'DeepLabv2MTKT':
        _, pred_main_list= model(tensor.cuda(device))
        pred_main = pred_main_list[0]
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TEST.MODEL[0]}")
    
    interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]),
                                             mode='bilinear', align_corners=True)
    output = interp(pred_main).cpu().data[0].numpy() # if model_weight == 1 else output_
    
    output = output.transpose(1, 2, 0)
    seg_map = np.argmax(output, axis=2)

    save_image_segmentation_frame(video_name, original_im, seg_map, index, cfg)

def preprocess_cityscapes_video_fram(img):
    # if cfg.TRAIN.IMG_MEAN == cfg.TEST.IMG_MEAN
    mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    # if (interpolation==Image.BICUBIC, rgb==True) in get_image fn in mtaf.dataset.base_dataset -> BaseDataset
    img = img.convert('RGB')
    # if cfg.TRAIN.INPUT_SIZE_TARGET == cfg.TEST.INPUT_SIZE_TARGET == (640, 320)
    img = img.resize((640, 320), Image.BICUBIC) # if 
    image = np.asarray(img, np.float32)
    image = image[:, :, ::-1]  # change to BGR
    image -= mean
    return image.transpose((2, 0, 1))

def save_image_segmentation_frame(video_name, original_im, seg_map, index, cfg):
    """Visualizes segmentation overlay view and stream it with IPython display."""
    image = np.array(original_im)
    all_images = image
    seg_image = label_to_color_image(seg_map).astype(np.uint8)

    seg_image = cv2.resize(seg_image.astype('uint8'), dsize=(image.shape[1], image.shape[0]))
    
    background = image.copy()
    overlay = seg_image.copy()
    added_image = cv2.addWeighted(background, 0.4, overlay, 0.75, 0)
    all_images = np.hstack((all_images, added_image))

    padding_length = all_images.shape[0]//8
    padding = all_images.copy()[:padding_length, :]
    padding[:, :] = 0

    # Rescaling *****************
    ratio = image.shape[1]/image.shape[0]
    h = 800
    w = int(ratio * h)
    all_images = cv2.resize(all_images.astype('uint8'), dsize=(w * 2, h))

    name = "Cityscapes" # cfg.TARGETS[i_target]
    EXP_NAME = cfg.EXP_NAME
    exp_name = "BASELINE" if "baseline" in EXP_NAME else 'MTKT'
    video_name = "mit_driveseg_sample" # fix name from url
    
    pred_path = f'../../eval_video_out/{exp_name}/{name}/video/{name}_{video_name}_{index:03}.png'
    all_images.transpose((1, 2, 0))
    cv2.imwrite(pred_path, cv2.cvtColor(all_images, cv2.COLOR_RGB2BGR))

def reconstruct_video(video_name):
  stack_frames_for_video()
  make_video(video_name)

def stack_frames_for_video():
  cityscap_to_stack_resized = cv2.imread('../../eval_video_out/stacked/cityscap_to_stack_resized.png')
  Cityscapes_files = glob.glob("../../eval_video_out/BASELINE/Cityscapes/video/*png")
  # if len = 0 return
  Cityscapes_files.sort()
  print(Cityscapes_files)
  for file in Cityscapes_files:
    cityscapes_baseline = cv2.imread(file)
    cityscapes_mtkt = cv2.imread(file.replace('BASELINE','MTKT'))
    cityescapes_stacked = np.vstack((cityscapes_baseline,cityscapes_mtkt))
    cityescapes_stacked = np.hstack((cityescapes_stacked,cityscap_to_stack_resized))
    
    path_to_save = file.replace("BASELINE",'stacked')
    cv2.imwrite(path_to_save, cityescapes_stacked)

def make_video(video_name):
  image_folder='../../eval_video_out/stacked/Cityscapes/video'
  fps=30
  image_files = [os.path.join(image_folder,img)
                for img in os.listdir(image_folder)
                if img.endswith(".png")]
  image_files.sort()
  clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
  if '/' in video_name:
    video_name = video_name.splite('/')[0]
  clip.write_videofile('../../eval_video_out/stacked/Cityscapes_mit.mp4')

################ End RAM video fns ##################

def evaluate_domain_adaptation(models, test_loader_list, cfg,
                               verbose=True):
    device = cfg.GPU_ID
    interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]),
                         mode='bilinear', align_corners=True)
    # eval
    if cfg.TEST.MODE == 'single':
        eval_single(cfg, models,
                    device, test_loader_list, interp,
                    verbose)
    elif cfg.TEST.MODE == 'best':
        eval_best(cfg, models,
                  device, test_loader_list, interp,
                  verbose)
    else:
        raise NotImplementedError(f"Not yet supported test mode {cfg.TEST.MODE}")

#######################################################################################
def eval_single(cfg, models,
                device, test_loader_list, interp,
                verbose):
    assert len(cfg.TEST.RESTORE_FROM) == len(models), 'Number of models are not matched'
    for checkpoint, model in zip(cfg.TEST.RESTORE_FROM, models):
        load_checkpoint_for_evaluation(model, checkpoint, device)
    # eval
    print("Evaluating model ", cfg.TEST.RESTORE_FROM[0])
    num_targets = len(cfg.TARGETS)

    video_url = 'https://github.com/lexfridman/mit-deep-learning/raw/master/tutorial_driving_scene_segmentation/mit_driveseg_sample.mp4'
    ########### RAM Video ############
    video_name = 'mit_driveseg_sample.mp4'
    num_frames_to_take = 598  # uncomment to use the full sample video
    # num_frames_to_take = 30
    video_segmentation_from_url(video_url, video_name, num_frames_to_take, model, cfg)
    # sequence_images_path = '../../../../DATASETS/CITYSCAPS_DATASET/leftImg8bit/demoVideo/stuttgart_01'
    # video_segmentation_from_sequence_images(sequence_images_path, 'stuttgart_01', model, cfg)

    sttacking = False # make True if you run BASELINE & MTKT both on same video
    if sttacking:
      reconstruct_video(video_name)
    ##################################


def eval_best(cfg, models,
              device, test_loader_list, interp,
              verbose):
    assert len(models) == 1, 'Not yet supported multi models in this mode'
    assert osp.exists(cfg.TEST.SNAPSHOT_DIR[0]), 'SNAPSHOT_DIR is not found'
    start_iter = cfg.TEST.SNAPSHOT_STEP
    step = cfg.TEST.SNAPSHOT_STEP
    max_iter = cfg.TEST.SNAPSHOT_MAXITER
    all_res_list = []
    cache_path_list = []
    num_targets = len(cfg.TARGETS)
    for target in cfg.TARGETS:
        cache_path = osp.join(osp.join(cfg.TEST.SNAPSHOT_DIR[0], target), 'all_res.pkl')
        cache_path_list.append(cache_path)
        if osp.exists(cache_path):
            all_res_list.append(pickle_load(cache_path))
        else:
            all_res_list.append({})
    cur_best_miou = -1
    cur_best_model = ''
    cur_best_miou_list = []
    cur_best_model_list = []
    for i in range(num_targets):
        cur_best_miou_list.append(-1)
        cur_best_model_list.append('')
    for i_iter in range(start_iter, max_iter, step): #
        print(f'Loading model_{i_iter}.pth')
        restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{i_iter}.pth')
        if not osp.exists(restore_from):
            # continue
            if cfg.TEST.WAIT_MODEL:
                print('Waiting for model..!')
                while not osp.exists(restore_from):
                    time.sleep(5)
        print("Evaluating model", restore_from)
        load_checkpoint_for_evaluation(models[0], restore_from, device)
        computed_miou_list = []
        for i_target in range(num_targets):
            print("On target", cfg.TARGETS[i_target])
            all_res = all_res_list[i_target]
            cache_path = cache_path_list[i_target]
            test_loader = test_loader_list[i_target]
            if i_iter not in all_res.keys():
                # eval
                hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
                # for index, batch in enumerate(test_loader):
                #     image, _, _, name = batch
                test_iter = iter(test_loader)
                for index in tqdm(range(len(test_loader))):
                    image, label, _, name = next(test_iter)


                    with torch.no_grad():
                        if cfg.TEST.MODEL[0] == 'DeepLabv2':
                            _, pred_main = models[0](image.cuda(device))
                        elif cfg.TEST.MODEL[0] == 'DeepLabv2MTKT':
                            _, pred_main_list= models[0](image.cuda(device))
                            pred_main = pred_main_list[0]
                        else:
                            raise NotImplementedError(f"Not yet supported {cfg.TEST.MODEL[0]}")
                        if cfg.TARGETS[i_target]=='Mapillary':
                            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
                        else:
                            interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]),
                                                 mode='bilinear', align_corners=True)
                        output = interp(pred_main).cpu().data[0].numpy()
                        output = output.transpose(1, 2, 0)
                        output = np.argmax(output, axis=2)
                    label = label.numpy()[0]
                    hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
                    if verbose and index > 0 and index % 500 == 0:
                        print('{:d} / {:d}: {:0.2f}'.format(
                            index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
                inters_over_union_classes = per_class_iu(hist)
                all_res[i_iter] = inters_over_union_classes
                pickle_dump(all_res, cache_path)
            else:
                inters_over_union_classes = all_res[i_iter]
            computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
            computed_miou_list.append(computed_miou)
            if cur_best_miou_list[i_target] < computed_miou:
                cur_best_miou_list[i_target] = computed_miou
                cur_best_model_list[i_target] = restore_from
            print('\tTarget:', cfg.TARGETS[i_target])
            print('\tCurrent mIoU:', computed_miou)
            print('\tCurrent best model:', cur_best_model_list[i_target])
            print('\tCurrent best mIoU:', cur_best_miou_list[i_target])
            if verbose:
                name_classes = np.array(test_loader.dataset.info['label'], dtype=np.str)
                display_stats(cfg, name_classes, inters_over_union_classes)
        computed_miou = round(np.nanmean(computed_miou_list), 2)
        if cur_best_miou < computed_miou:
            cur_best_miou = computed_miou
            cur_best_model = restore_from
        print('\tMulti-target:', cfg.TARGETS)
        print('\tCurrent mIoU:', computed_miou)
        print('\tCurrent best model:', cur_best_model)
        print('\tCurrent best mIoU:', cur_best_miou)


def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)


def display_stats(cfg, name_classes, inters_over_union_classes):
    for ind_class in range(cfg.NUM_CLASSES):
        print(name_classes[ind_class]
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))
