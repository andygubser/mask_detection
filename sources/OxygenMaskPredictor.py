import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import skimage.io
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import mrcnn.model as modellib
import time

class PredictionConfig(Config):
    NAME = "oxygenmask_cfg"
    NUM_CLASSES = 4 #background + masks and nomasks and wearing incorrect
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def imgToArray(absoluteFilePath): # copied from MaskRCNN
    image = skimage.io.imread(absoluteFilePath)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image

def create_plot_prediction(absolutePathImage, model, cfg):
    print("predict masks for image: {0}".format(absolutePathImage))
    classNameById = {1: "with_mask", 2: "without_mask", 3: "mask_weared_incorrect"}
    image = imgToArray(absolutePathImage)
    scaled_image = modellib.mold_image(image, cfg)
    sample = np.expand_dims(scaled_image, 0)
    y_predict = model.detect(sample, verbose=0)[0]
    plt.imshow(image)
    plt.title('Oxygen Mask Prediction')
    ax = plt.gca()
    for i in range(len(y_predict['rois'])):
        y1, x1, y2, x2 = y_predict['rois'][i]
        width, height = x2 - x1, y2 - y1
        predClassId = int(y_predict['class_ids'][i])
        className = classNameById[predClassId]
        plt.text(x1, y1, className, fontsize=10, color="white", bbox=dict(facecolor='red', alpha=0.2))
        print("className: {0}".format(className))
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        ax.add_patch(rect) # draw box
    return plt

def init_model_config():
    print("[x] to close")
    config = PredictionConfig()
    sourcePath = os.path.dirname(os.path.abspath(__file__))
    model = MaskRCNN(mode='inference', model_dir=sourcePath, config=config)
    modelAbsolutePath = os.path.join(sourcePath, "mask_rcnn_oxygenmask_cfg_0005.h5")
    model.load_weights(modelAbsolutePath, by_name=True)
    return model, config

def get_lastImage(imageBasePath):
    imageFiles = [os.path.join(imageBasePath, fileName) for fileName in os.listdir(imageBasePath) if
                  fileName.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    oldestImageFile = None

    if len(imageFiles) > 0:
        try:
            oldestImageFile = max(imageFiles, key=os.path.getctime)
        except:
            print("image files have been removed")

    return oldestImageFile

model,config = init_model_config()

sourcePath = os.path.dirname(os.path.abspath(__file__))
imageBasePath = "C:/Users/fabiantrottmann/Dropbox/_Predictions"
imageFiles = [os.path.join(imageBasePath, fileName) for fileName in os.listdir(imageBasePath) if fileName.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

lastImage = None
while True:
    currentImage = get_lastImage(imageBasePath)
    if currentImage != None and lastImage != currentImage:
        print(currentImage)
        lastImage = currentImage
        if plt != None:
            plt.close()
        plt = create_plot_prediction(currentImage, model, config)
        plt.get_current_fig_manager().window.state('zoomed')
        plt.pause(2)
        plt.draw()
        plt.show(block=False)