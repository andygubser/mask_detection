#%tensorflow_version 1.x
from mrcnn import utils
import os
import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import skimage.io
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

class PredictionConfig(Config):
    NAME = "oxygenmask_cfg"
    NUM_CLASSES = 4 #background + masks and nomasks and wearing incorrect
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def imgToArray(absoluteFilePath):
    image = skimage.io.imread(absoluteFilePath)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image

def plot_prediction(imageSourceDir, model, cfg):
    classNameById = {1: "with_mask", 2: "without_mask", 3: "mask_weared_incorrect"}

    imageFiles = [os.path.join(imageSourceDir, fileName) for fileName in os.listdir(imageSourceDir) if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    print(imageFiles)
    return None
    for i in range(len(imageFiles)):
        imagePath = imageFiles[i]
        image = imgToArray(imagePath)
        mask, _ = dataset.load_mask(i)
        scaled_image = modellib.mold_image(image, cfg)
        sample = np.expand_dims(scaled_image, 0)
        y = model.detect(sample, verbose=0)[0]

        pyplot.subplot(len(images), 2, i * 2 + 2)
        pyplot.imshow(image)
        pyplot.title('Oxygen Mask Prediction')
        ax = pyplot.gca()
        for i in range(len(y['rois'])):
            y1, x1, y2, x2 = y['rois'][i]
            width, height = x2 - x1, y2 - y1
            predClassId = int(yhat['class_ids'][i])
            className = classNameById[predClassId]
            print("className: {0}".format(className))
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            ax.add_patch(rect) # draw box
            pyplot.show()

def load_model():
    cfg = PredictionConfig()
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    model_path = 'mask_rcnn_oxygenmask_cfg_0005.h5'
    model.load_weights(model_path, by_name=True)
    return model

model = load_model()
imageSourceDir = 'C:\Projects\Python\DeepLearningInVision\mask_detection\sources'

plot_prediction(imageSourceDir, model, cfg)