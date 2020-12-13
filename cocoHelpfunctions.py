from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
from skimage import transform
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from skimage.color import gray2rgb

def loadTrainAnn(path):
    with open(path, 'rb') as file_pi:
        images, dataset_size, coco = pickle.load(file_pi)
        dataset_size = len(images)
        print("Dataset length: {}".format(dataset_size))
        return images, dataset_size, coco

def filterDataset(folder, classes=None, mode='train',examples_Per = 1.0):    
    # initialize COCO api for instance annotations
    annFile = '{}/annotations/instances_{}.json'.format(folder, mode)
    coco = COCO(annFile)
    images = []
    if classes!=None:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given categories
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)
    
    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)
    
    # Now, filter out the repeated images
    unique_images = []
    for i in range(int(len(images)*examples_Per)):
        if images[i] not in unique_images:
            unique_images.append(images[i])

    random.shuffle(unique_images)
    dataset_size = len(unique_images)
    print("Dataset size {}".format(dataset_size))
    return unique_images, dataset_size, coco

def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return None

def getImage(imageObj, img_folder, input_image_size):
    # Read and normalize an image
    train_img = io.imread(img_folder + '/' + imageObj['file_name'])/255.0
    # Resize
    train_img = cv2.resize(train_img, input_image_size)
    if (len(train_img.shape)==3 and train_img.shape[2]==3): # If it is a RGB 3 channel image
        return train_img
    else: # To handle a black and white image, increase dimensions to 3
        stacked_img = np.stack((train_img,)*3, axis=-1)
        return stacked_img
    
    
def getBinaryMask(imageObj, coco, catIds, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        new_mask = cv2.resize(coco.annToMask(anns[a]), input_image_size)
        
        #Threshold because resizing may cause extraneous values
        new_mask[new_mask >= 0.5] = 1
        new_mask[new_mask < 0.5] = 0

        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask


def dataGeneratorCoco(images, classes, coco, folder, input_image_size=(224,224), batch_size=4, mode='train'):
    
    img_folder = '{}/{}'.format(folder, "train2017")
    dataset_size = len(images)
    catIds = coco.getCatIds(catNms=classes)
    
    c = 0
    while(True):
        img = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')
        mask = np.zeros((batch_size, input_image_size[0], input_image_size[1], 1)).astype('float')

        for i in range(c, c+batch_size): #initially from 0 to batch_size, when c = 0
            imageObj = images[i]
            
            ### Retrieve Image ###
            train_img = getImage(imageObj, img_folder, input_image_size)
            
            ### Create Mask ###
            train_mask = getBinaryMask(imageObj, coco, catIds, input_image_size)
                    
            
            # Add to respective batch sized arrays
            img[i-c] = train_img
            mask[i-c] = train_mask
            
        c+=batch_size
        if(c + batch_size >= dataset_size):
            c=0
            random.shuffle(images)
        yield img, mask

def augmentationsGenerator(gen, augGeneratorArgs,preprocessing, seed=None):
    # Initialize the image data generator with args provided
    image_gen = ImageDataGenerator(**augGeneratorArgs)
    
    # Remove the brightness argument for the mask. Spatial arguments similar to image.
    augGeneratorArgs_mask = augGeneratorArgs.copy()
    _ = augGeneratorArgs_mask.pop('brightness_range', None)
    # Initialize the mask data generator with modified args
    mask_gen = ImageDataGenerator(**augGeneratorArgs_mask)
    
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    
    for img, mask in gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation of the images 
        # will end up different from the augmentation of the masks
        g_x = image_gen.flow(255*img, 
                             batch_size = img.shape[0], 
                             seed = seed, 
                             shuffle=True)
        g_y = mask_gen.flow(mask, 
                             batch_size = mask.shape[0], 
                             seed = seed, 
                             shuffle=True)
        
        #img_aug = procees_input(next(g_x))
        #img_aug = next(g_x)/255.0


        img_aug = preprocessing(next(g_x))
        mask_aug = next(g_y)

        yield img_aug, mask_aug

def get_test_Data(images, classes, coco, folder, input_image_size=(224,224)):
    
    img_folder = '{}/{}'.format(folder, 'val2017')
    dataset_size = len(images)
    catIds = coco.getCatIds(catNms=classes)

    imgs = np.zeros((dataset_size, input_image_size[0], input_image_size[1], 3)).astype('float')
    masks = np.zeros((dataset_size, input_image_size[0], input_image_size[1])).astype(np.uint8)

    for i in range(0, dataset_size):
        imageObj = images[i]

            ### Retrieve Image ###
        train_img = getImage(imageObj, img_folder, input_image_size)
            
            ### Create Mask ###
        train_mask = getBinaryMask(imageObj, coco, catIds, input_image_size)
        train_mask = train_mask.reshape((input_image_size[0], input_image_size[1]))

        imgs[i] = train_img
        masks[i] = train_mask

    return imgs, masks

def get_Mean_IoU(y_true,y_pred):
    iou_coefs = []
    dataset_size = y_pred.shape[0]
    for i in range(0,dataset_size):
        iou_coefs.append(iou_coef(y_true[i],y_pred[i]))
    meanIou = np.round(np.sum(np.asarray(iou_coefs))/len(iou_coefs),2)
    print("MeanIOU : {}".format(meanIou))

def iou_coef(y_true, y_pred):
  intersection = np.logical_and(y_true, y_pred)
  union = np.logical_or(y_true, y_pred)
  iou_score = np.sum(intersection) / np.sum(union)
  return iou_score

def get_ypred_and_ytrue(imgs,masks,model):
    y_pred_arr = []
    dataset_size,h,w,_ = imgs.shape
    y_pred = model.predict(imgs)
    y_pred = y_pred > 0.5
    return (masks.reshape((dataset_size,h,w)),y_pred.reshape((dataset_size,h,w)))

from sklearn.metrics import confusion_matrix, classification_report
def get_pixel_precision(y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true.ravel(),y_pred.ravel()).ravel()
    pixel_precision = np.round((tp+tn)/(tn+fp+fn+tp),2)

    print("Pixel precision: {}".format(pixel_precision))

def getRGBAmask(input_image):
    x,y = np.where(input_image == True)
    if input_image.ndim < 3:
        input_image = gray2rgb(input_image)
    result = np.concatenate((input_image, np.zeros((input_image.shape[0], input_image.shape[1], 1))), axis=2)
    result[x,y,:] = (1,0,0,1)
    return result

def plotResult(img,mask,model):
    plt.rcParams.update({'font.size': 12})
    predictions = model.predict(img)
    prediction = predictions[0][:,:,0]
    img = img[0]
    prediction_tresholded = (prediction > 0.5).astype(np.uint8)
    plt.figure(figsize=(20,10))
    plt.subplot(141)
    plt.imshow(img)
    plt.imshow(getRGBAmask(mask),alpha=0.4)
    plt.title("Test img with segmentation mask")
    plt.subplot(142)
    plt.imshow(prediction)
    plt.title("Prediction")
    plt.subplot(143)
    plt.imshow(prediction_tresholded)
    plt.title("Prediction with thresh = 0.5")
    plt.subplot(144)
    plt.imshow(img)
    plt.imshow(getRGBAmask(prediction_tresholded),alpha=0.4)
    plt.title("Image overlap")