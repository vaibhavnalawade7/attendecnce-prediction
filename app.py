from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import joblib

model = joblib.load('lr_model.pkl')

# Save a model
joblib.dump(model, 'lr_model.pkl')

# Load a model


# Load the trained models
lr_model = pickle.load(open("lr_model.pkl", "rb"))
dt_model = pickle.load(open("dt_model.pkl", "rb"))

# Load the one-hot encoders
ohe_cat = pickle.load(open("ohe_cat.pkl", "rb"))
ohe_batch = pickle.load(open("ohe_batch.pkl", "rb"))


app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    # Get the user inputs from the request object
    data = request.get_json(force=True)
    distance = data['distance']
    batch = data['batch']
    job_status = data['job_status']
    course_status = data['course_status']
    ug_status = data['ug_status']
    
    # Transform the user inputs into a dataframe
    df = pd.DataFrame([[distance, batch, job_status, course_status, ug_status]],
                      columns=['Distance', 'Batch', 'Job_status', 'Course_status', 'UG_Status'])
    
    # One-hot encode categorical features
    cat_cols = ['Job_status', 'Course_status']
    cat_df = df[cat_cols]
    cat_df = ohe_cat.transform(cat_df).toarray()
    
    batch_col = ['Batch']
    batch_df = df[batch_col]
    batch_df = ohe_batch.transform(batch_df).toarray()
    
    # Combine the one-hot encoded features with the numerical features
    num_cols = ['Distance']
    num_df = df[num_cols]
    
    features = np.concatenate([cat_df, batch_df, num_df], axis=1)
    
    # Make predictions using the logistic regression and decision tree models
    lr_prediction = lr_model.predict(features)[0]
    dt_prediction = dt_model.predict(features)[0]
    
    # Return the predictions as a JSON response
    return jsonify(lr_prediction=lr_prediction, dt_prediction=dt_prediction)


pickle.dump(lr_model, open("lr_model.pkl", "wb"))
pickle.dump(dt_model, open("dt_model.pkl", "wb"))
pickle.dump(ohe_cat, open("ohe_cat.pkl", "wb"))
pickle.dump(ohe_batch, open("ohe_batch.pkl", "wb"))

if __name__ == "__main__":
    app.run(debug=True)