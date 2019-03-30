# pothole-detection
pothole detection -keras with tensorflow

Implementation from the paper "Pothole Detection Using Location-Aware Convolutional Neural networks".

## The structure of project
- dataset<br>
- preprocessing<br>
- network<br>
- result<br>
- utils<br>
- train.py<br>
- test.py<br>

## Requirements
- tensorflow
- keras
- pothole dataset: 
   The dataset(each image is 800 ¡Á 600 in JPG format) was first released in the Data Science Hackathon, a computer vision challenge sponsored by IBM,<br>
   the Machine Intelligence Institute of Africa, and Cortex Logic, which took place in Johannesburg in September 2017. <br>
   you can download [Training & Test Data and Python Notebooks](https://drive.google.com/open?id=0B1IZ6xxwxyvTcWNOWHAxeVgyTlU)<br>
   <br>
   The orginal images(each image is 3680 ¡Á 2760 in JPG format) was first released in paper:<br>
   [1] S. Nienaber, M.J. Booysen, R.S. Kroon, ¡°Detecting potholes using simple image processing<br>
       techniques and real-world footage¡±, SATC, July 2015, Pretoria, South Africa.<br>
   [2] S. Nienaber, R.S. Kroon, M.J. Booysen , ¡°A Comparison of Low-Cost Monocular Vision Techniques<br>
       for Pothole Distance Estimation¡±, IEEE CIVTS, December 2015, Cape Town, South Africa.<br>
   you can download the orginal image from [here](http://goo.gl/Uj38Sf)<br>

## Usage
### step1: The re-organized dataset are created by a simple preprocessing operation.
1¡¢because the files name of two dataset is not the same, 
   we have to find the same picture and rename it at the same with the help of some simily software;<br>
2¡¢run b3org_to_roi.py to crop the road images from the origianl image and create the new labels;<br>
3¡¢run s6create_heatmap.py to create the ground truth of heatmap;<br>
4¡¢hold out 800 training images as a validation set;<br>

Or: The re-organized dataset can directly downloaded from [here](http://goo.gl/Uj38Sf)<br>

### step2: Training the model
1¡¢Download pre-triand models and weights,and put them under the $PRJ_ROOT.The default backbone is resnet50.
2¡¢If you want to change the folder for your own path of the dataset, you will need to change the XXX path in the ***.py file;
3¡¢Train. By default, trained networks are saved under:

### step3: Testing
1¡¢you can change XXX and other parameters in test.py;<br>
2¡¢run test.py;<br>
2¡¢if set Debug=True in code, The results will be output under $PRJ_ROOT/XXX;<br>
   
#whole source code will coming soon

