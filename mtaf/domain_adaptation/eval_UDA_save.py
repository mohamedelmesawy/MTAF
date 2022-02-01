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


def get_image_name(image_path):
  return image_path.split('/')[-1][:-4]


def vis_segmentation(image_path, ground_truth, prediction, name, EXP_NAME):
    rows = 1
    columns = 4

    print(f'======================= {name} =======================')
    fig = plt.figure(figsize=(20, 4)) 
    
    plt.clf()
    if name == "Cityscapes":  # CHECK ME RAM
      image_path = "../../data/Cityscapes/leftImg8bit/val/" + image_path
    ### image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig.add_subplot(rows, columns, 1)
    plt.imshow(image)
    plt.title('Image')
    plt.axis('off')

    ### ground_truth
    fig.add_subplot(rows, columns, 2)
    ground_truth = label_to_color_image(ground_truth)
    plt.imshow(ground_truth)
    plt.title('Ground Truth')
    plt.axis('off')

    ### Prediction
    fig.add_subplot(rows, columns, 3)
    prediction = label_to_color_image(prediction)
    plt.imshow(prediction)
    plt.title('Prediction')
    plt.axis('off')

    ### image, Prediction opacity
    fig.add_subplot(rows, columns, 4)
    
    if name == "Cityscapes":
        prediction = cv2.resize(prediction.astype('uint8'), dsize=(image.shape[1], image.shape[0]))
    plt.imshow(image)
    plt.title('Image + Prediction Segmentation Overlay')
    plt.imshow(prediction, alpha=0.5)
    plt.axis('off')

    image_name = get_image_name(image_path)
    exp_name = "BASELINE" if "baseline" in EXP_NAME else 'MTKT'  
    plt.savefig(f'../../eval_out/{exp_name}/{name}/{name}_{image_name}.png', bbox_inches='tight', pad_inches=0.2)
    plt.clf()
    plt.close()


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


############################  EVALUATE SINGLE - EVALUATION ###########################################################
def eval_single(cfg, models,
                device, test_loader_list, interp,
                verbose):
    assert len(cfg.TEST.RESTORE_FROM) == len(models), 'Number of models are not matched'
    for checkpoint, model in zip(cfg.TEST.RESTORE_FROM, models):
        load_checkpoint_for_evaluation(model, checkpoint, device)
    # eval
    print("Evaluating model ", cfg.TEST.RESTORE_FROM[0])
    num_targets = len(cfg.TARGETS)
    
    ### Foreach Target - TWO TARGETS
    for i_target in range(num_targets):
        test_loader = test_loader_list[i_target]
        hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
        test_iter = iter(test_loader)

        IOU_for_each_image = []

        ### Foreach Image
        for index in tqdm(range(len(test_loader))):
            image, label, _, name = next(test_iter)

            with torch.no_grad():
                output = None
                ### Foreach Model check which model to use in prediction  INNERRRRRRRRRRR  take one Model
                for model, model_weight in zip(models, cfg.TEST.MODEL_WEIGHT):
                    print('Model Name: ', cfg.TEST.MODEL[0])
                    if cfg.TEST.MODEL[0] == 'DeepLabv2':
                        _, pred_main = model(image.cuda(device))
                    elif cfg.TEST.MODEL[0] == 'DeepLabv2MTKT':
                        _, pred_main_list= model(image.cuda(device))
                        pred_main = pred_main_list[0]
                    else:
                        raise NotImplementedError(f"Not yet supported {cfg.TEST.MODEL[0]}")

                    if cfg.TARGETS[i_target]=='Mapillary':
                        interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
                    else:                      
                        interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]),
                                             mode='bilinear', align_corners=True)
                    output_ = interp(pred_main).cpu().data[0].numpy()
                    if output is None:
                        output = model_weight * output_
                    else:
                        output += model_weight * output_
                    
                assert output is not None, 'Output is None'
                output = output.transpose(1, 2, 0)
                output = np.argmax(output, axis=2)

            label = label.numpy()[0]   #check me

            ##################### RAM Start SAVE AND DISPLAY #####################
            ram_img = image.numpy().transpose(2, 3, 1, 0)[:, :, :, 0]
            ram_label = label.copy()
            ram_output = output
            image_path = name[0]
            ######## RAM DISPLAYING ########
            if cfg.TARGETS[i_target] == "Mapillary":
                ram_label = ram_label[:, :3259]            
                ram_output = ram_output[:, :3259]

            vis_segmentation(image_path, ram_label, ram_output, cfg.TARGETS[i_target], cfg.EXP_NAME)
            ##################### RAM End  #####################
           
            hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)


        # IOU_for_each_image_np=np.array(IOU_for_each_image)
        # model = "BASELINE" if "baseline" in cfg.EXP_NAME else 'MTKT'  
        # np.save(f"../../eval_out/IOU_per_image/{cfg.TARGETS[i_target]}_{model}.npy",IOU_for_each_image_np)


        inters_over_union_classes = per_class_iu(hist)
        print('\tTarget:', cfg.TARGETS[i_target])
        print(f'mIoU = \t{round(np.nanmean(inters_over_union_classes) * 100, 2)}')
        if verbose:
            name_classes = np.array(test_loader.dataset.info['label'], dtype=np.str)
            display_stats(cfg, name_classes, inters_over_union_classes)

    
############################  EVALUATE BEST - TESTING ###########################################################
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
