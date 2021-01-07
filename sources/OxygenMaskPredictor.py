import os
import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import skimage.io
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import mrcnn.model as modellib

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

def plot_prediction(absolutePathImage, model, cfg):
    print("predict masks for image: {0}".format(absolutePathImage))

    classNameById = {1: "with_mask", 2: "without_mask", 3: "mask_weared_incorrect"}
    image = imgToArray(absolutePathImage)
    scaled_image = modellib.mold_image(image, cfg)
    sample = np.expand_dims(scaled_image, 0)
    y_predict = model.detect(sample, verbose=0)[0]

    pyplot.subplot(len(imageFiles), 2, i * 2 + 2)
    pyplot.imshow(image)
    pyplot.title('Oxygen Mask Prediction')
    ax = pyplot.gca()
    for i in range(len(y_predict['rois'])):
        y1, x1, y2, x2 = y_predict['rois'][i]
        width, height = x2 - x1, y2 - y1
        predClassId = int(y_predict['class_ids'][i])
        className = classNameById[predClassId]
        print("className: {0}".format(className))
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        ax.add_patch(rect) # draw box
    pyplot.show()

def get_lastImage(imageBasePath):

    return imageBasePath

cfg = PredictionConfig()
sourcePath = os.path.dirname(os.path.abspath(__file__))

model = MaskRCNN(mode='inference', model_dir=sourcePath, config=cfg)
model.load_weights('mask_rcnn_oxygenmask_cfg_0005.h5', by_name=True)

imageBasePath = sourcePath
imageFiles = [os.path.join(imageBasePath, fileName) for fileName in os.listdir(imageBasePath) if fileName.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

print("[x] to close")
while True:
    absoluteFilePath = get_lastImage()
    plot_prediction(absoluteFilePath, model, cfg)



