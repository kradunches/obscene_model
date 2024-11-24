# obscene_model
In this project a model trained on a dataset with obscene language determines whether the text entered by the user in html form is offensive or not.

Data for model was taken from Kaggle repository:
https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/overview

# Files
+ ```jigsaw-toxic-comment-train.csv``` - the data on which the model was trained, more detailed information can be found in the kaggle repository, the link to which was above
+ ```requirements.txt``` - all necessary Python packages, just execute command ```pip install -r requirements.txt```
+ ```console_input_model.py``` - console version of the application. If you run this ```.py``` file, then after training the model the user will be able to classify messages from the program console
+ ```server.py``` - web version of the programm. In this file the model is trained and the Flask server is launched, which will classify messages
+ ```/templates/index.html``` - user interface for ```server.py```

# Usage
1. Clone repository on your local computer: ```git clone https://github.com/kradunches/obscene_model```
2. Run the Flask server: ```python server.py```
3. Wait for the server to start and then open your browser to '''localhost:5000'''
4. If entered message is obscene it will be highlighted in red color, and if the message does not contains obscene language it will be highlighted in grey color
