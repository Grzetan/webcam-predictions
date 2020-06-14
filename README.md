# webcam-predictions
Train models using your own datasets made using webcam to make predictions.
# Setup
To run this app you must have python 3 installed and working webcam.
- Download repo and unzip it 
- Go to root folder and type pip install -r requirements.txt
- Next run python main.py
- If every thing went well you sholud see starting screen with 5 buttons with text 'Empty'
# Training new model
- Click on a button with text 'Empty' on it
- Next you sholud see a screen with 3 input boxes. The first one is for the name of the model, second one for number of neurons in last layer and the third one for number of images of each class. Fill them in and click continue.
- Now wait till the model is trained.
- Done. If the predicted class is the first one you sholud see number 1 on your screen, if it's the second one, number two...
# Predicting from existing model
- After starting the app just click on the model you want to predict from.
# Deleting existing models
- In the main menu click the red cross next to the model you want to delete.
# Modyfing options
- Go to 'config.py' file and change options like number of epochs, batch size, learning rate... .
# Custom scripts after prediction
- Check the index of the model of which scripts you want to change (After starting the app check on which place is your model located).
- In 'prediction_functions.py' file go to the function with your model's index.
- Now instead of printing number to console you can add your own script
# Custom model
- To change main model architecture go to 'model.py' file and change whatever you want :)

