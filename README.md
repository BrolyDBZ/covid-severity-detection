# covid-severity-detection
## classification model
- classification model creating using pretrained architecture (InceptionV3, Resnet) to classify the disease in stages(0-Non_infection to 3- Highly Infected) and deduce which architecutre gives better accuracy.

## segmentation model
- segmentation model (U-Net) created using pretrained architecutre (Resnet) to get the semantic segmentation for disease.
- predicting the severity by calculating the dice loss and % spread of Infection in Lungs.

## Training and testing
- Hyperparameterization.
- Data Pre-Processing, Data-Augmentation.
- Test result, classification matrix, accurracy.

## Result
- **classification :** 92-98% accurate
- **segmentation :** upto 95% accurate
- **sample result**

![segmentation result](https://github.com/BrolyDBZ/covid-severity-detection/blob/main/covid%20severity%20detection/result.jpg)

