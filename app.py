from flask import Flask, render_template,url_for,request
# Importing libraries
import numpy as np
import pandas as pd
from scipy.stats import mode
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
# from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])


# Defining the Function
# Input: string containing symptoms separated by commmas
# Output: Generated predictions by models
def predictDisease(symptoms):
    DATA_PATH = "data/Training.csv"
    data = pd.read_csv(DATA_PATH).dropna(axis = 1)
    X = data.iloc[:,:-1]
    symptoms = X.columns.values

    # value using LabelEncoder
    encoder = LabelEncoder()

    # Creating a symptom index dictionary to encode the
    # input symptoms into numerical form
    symptom_index = {}
    for index, value in enumerate(symptoms):
        symptom = " ".join([i.capitalize() for i in value.split("_")])
        symptom_index[symptom] = index

    data_dict = {
	    "symptom_index":symptom_index,
	    "predictions_classes":encoder.classes_
    }
    symptoms = symptoms.split(",")
	
	# creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
		
	# reshaping the input data and converting it
	# into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
	
    # load modal

    final_svm_model = joblib.load('model/svm_model.joblib')
	# generating individual outputs
	# rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
	# nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
	
	# making final prediction by taking mode of all predictions
    final_prediction = mode([svm_prediction])[0][0]
    predictions = {
		# "rf_model_prediction": rf_prediction,
		# "naive_bayes_prediction": nb_prediction,
		"svm_model_prediction": svm_prediction,
		"final_prediction":final_prediction
	}
    return render_template('result.html', predictions)
# def predict():
#     df = pd.read_csv("data/Youtube01-Psy.csv")
#     df_data = df[['CONTENT', 'CLASS']]
#     # Features and Labels
#     df_x = df_data['CONTENT']
#     df_y = df_data.CLASS

#      #Extract the features with countVectorizer
#     corpus = df_x
#     cv = CountVectorizer()
#     X = cv.fit_transform(corpus)
#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size = 0.33, random_state = 42)
    
#     #Navie Bayes
#     clf = MultinomialNB()
#     clf.fit(X_train, y_train)
#     clf.score(X_test, y_test)
    
#     #Save Model
#     joblib.dump(clf, 'model.pkl')
#     print("Model dumped!")
    
#     #ytb_model = open('spam_model.pkl', 'rb')
#     clf = joblib.load('model.pkl')
#     if request.method == 'POST':
#         comment = request.form['comment']
#         data = [comment]
#         vect = cv.transform(data).toarray()
#         my_prediction = clf.predict(vect)
#     return render_template('result.html', prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)    