Install:
cd ask_detection\sources\venv\Scripts
.\activate
cd ..\..\Mask_RCNN
pip3 install -r requirements.txt
python3 setup.py install

test:
pip show mask-rcnn

run program
cd C:\Projects\Python\DeepLearningInVision\mask_detection\sources\venv\Scripts
.\python.exe C:\Projects\Python\DeepLearningInVision\mask_detection\sources\OxygenMaskPredictor.py