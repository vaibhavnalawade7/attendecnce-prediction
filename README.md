# Attendance Base Predictive Model For Course Completion

This is a Flask-based project for building a predictive model that predicts the course completion rate based on attendance. The aim of this project is to provide insights into how attendance can impact the course completion rate, and to help educational institutions make data-driven decisions.

## Dataset

The dataset used for this project was created by the author and can be found at https://www.kaggle.com/datasets/vaibhavnalawade7/attendance. The dataset consists of the following columns:

- Batch: The batch number of the student
- UG_Status: The undergraduate status of the student (e.g. UG, PG)
- Job_status: The job status of the student (e.g. Employed, Unemployed)
- Course_status: The status of the course (e.g. Completed, In Progress)
- Distance: The distance from the student's home to the educational institution

## Model Building

The project uses a logistic regression model and a decision tree model to predict the course completion rate based on attendance. The models were trained on the provided dataset using scikit-learn. The categorical features were one-hot encoded using scikit-learn's OneHotEncoder, and the numerical features were standardized using scikit-learn's StandardScaler.

## Usage

To use this project, you can run the Flask application and send a POST request to the following endpoint:

http://localhost:5000/


The POST request should include the following parameters:

- distance: The distance from the student's home to the educational institution
- batch: The batch number of the student
- job_status: The job status of the student
- course_status: The status of the course
- ug_status: The undergraduate status of the student

The Flask application will return the course completion rate prediction as a JSON response, using both the logistic regression and decision tree models.

## Files

The project includes the following files:

- lr_model.pkl: The saved logistic regression model
- dt_model.pkl: The saved decision tree model
- ohe_cat.pkl: The saved one-hot encoder for categorical features
- ohe_batch.pkl: The saved one-hot encoder for the batch feature
- app.py: The main Flask application file
- README.md: The readme file

## Requirements

The project requires the following libraries:

- Flask
- pandas
- numpy
- scikit-learn

You can install the required libraries using the following command:
pip install -r requirements.txt


