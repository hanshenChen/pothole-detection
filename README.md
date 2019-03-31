# pothole-detection
pothole detection -keras with tensorflow

Implementation from the paper "Pothole Detection Using Location-Aware Convolutional Neural networks".

## Note:
The public experimental code has been changed a lot of times in doing the experiment,which will cause a lot of redundant issues,
it was not design in accordance with the priciples and norms of software engineering. 

## The structure of project
- dataset<br>
- preprocessing<br>
- models<br>
- result<br>
- src<br>
- train.py<br>
- test.py<br>

## Requirements
- Tensorflow
- Keras
- Pothole dataset: <br>
   The dataset(each image is 800 °¡ 600 in JPG format) was first released in the Data Science Hackathon, a computer vision challenge sponsored by IBM,the Machine Intelligence Institute of Africa, and Cortex Logic, which took place in Johannesburg in September 2017. <br>
   you can download the dataset from[Training & Test Data and Python Notebooks](https://drive.google.com/open?id=0B1IZ6xxwxyvTcWNOWHAxeVgyTlU)<br>
   <br>
   The orginal images(each image is 3680 °¡ 2760 in JPG format) was first released in paper:<br>
   [1] S. Nienaber, M.J. Booysen, R.S. Kroon, °∞Detecting potholes using simple image processing<br>
       techniques and real-world footage°±, SATC, July 2015, Pretoria, South Africa.<br>
   [2] S. Nienaber, R.S. Kroon, M.J. Booysen , °∞A Comparison of Low-Cost Monocular Vision Techniques<br>
       for Pothole Distance Estimation°±, IEEE CIVTS, December 2015, Cape Town, South Africa.<br>
   you can download the orginal image from [here](http://goo.gl/Uj38Sf)Ã·»°¬Î£∫va7g<br>

## Usage
### step1: The re-organized dataset are created by a simple preprocessing operation.
1°¢because the files name of two dataset is not the same, 
   we have to find the same picture and rename it at the same with the help of some Similarity Image Finder software;<br>
2°¢run preprocessing/b3org_to_roi.py to crop the road images from the origianl image and create the new labels;<br>
3°¢hold out 800 training images as a validation set;<br>
4°¢the road images are resized to 352*244 to meet the LCNN model requirement. 
5°¢run preprocessing/s6create_heatmap.py to create the ground truth of heatmap;<br>
6°¢run preprocessing/create_patch.py to create the patch dataset.

Or: The re-organized dataset can directly downloaded from [here](https://pan.baidu.com/s/1XLpablhy4xHKxNVZCoXrPA) <br>

### step2: Training the LCNN model
1°¢Download pre-triand weights(ImageNet classification),and put them under the $PRJ_ROOT.The default backbone is resnet50.
2°¢If you want to change the folder for your own path of the dataset, you will need to change the XXX path in the trainlcnn.py file;
3°¢Train:run trainlcnn.py. By default, trained networks are saved under: $PRJ_ROOT/models;<br>
note:During training, the best model is chosen the lowest error on the validation set;<br>

### step3: Training the PCNN model
source code will coming soon<br>
note:During training, the best model is chosen the lowest error on the validation set.<br>

### step4: Testing
1°¢you can change the path of dataset and other parameters in testwhole.py;<br>
2°¢run testwhole.py;<br>
2°¢if set Debug=True in code, The results will be output under $PRJ_ROOT/result;<br>
   
#whole source code will coming soon

