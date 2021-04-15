from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from flask_sqlalchemy import SQLAlchemy
pd.set_option('max_rows', 10)
pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns




covid_data_path = "J:\\full_grouped.csv"
swine_data_path = "J:\swine_flu.csv"
covid_data = pd.read_csv(covid_data_path)
swine_data = pd.read_csv(swine_data_path)



# Knn Imputer
from sklearn.impute import KNNImputer
nan = np.nan

#Designate the features to become X
features=['Deaths']
X= swine_data[features]
#Apply KNN imputer
imputer = KNNImputer(n_neighbors=2, weights="uniform")
ImputedX=imputer.fit_transform(X)

# Convert output to a data frame to show the stats
imputed_df = pd.DataFrame.from_records(ImputedX)
imputed_df.columns = features
imputed_df['Country'] = swine_data['Country']
imputed_df['Cases'] = swine_data['Cases']
imputed_df['Update Time'] = swine_data['Update Time']
# print('---------------------------------------')
missing_0_values_count = imputed_df.isnull().sum()
# print(missing_0_values_count)


# Categorical Encoders
import category_encoders as ce
enc = ce.OrdinalEncoder(cols=["Country","Update Time"],handle_missing='return_nan',return_df= True)

#We now fit the model and transform the data and put it in X which is a dataframe
X=enc.fit_transform(imputed_df)


# Outlier Detection
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)
totalOutliers=0
for pred in y_pred:
    if pred == -1:
        totalOutliers=totalOutliers+1
print ("Number of predicted outliers:",totalOutliers)

# Removing Outliers
X['Country'] = imputed_df['Country']
X['Update Time'] = imputed_df['Update Time']
mask2 = (y_pred != -1)
# print(len(preds),len(X))
Cleaned_SwineFrame = X[mask2]
print('******************************************************')
print(len(Cleaned_SwineFrame))


# Converting to log to show in boxplot
# plt.figure(figsize=(8,6))
# transformed_df = {'transformed_cases': imputed_df["Cases"].apply(np.log), 'transformed_death': imputed_df["Deaths"].replace(0,np.nan).apply(np.log)}
# print(transformed_df["transformed_death"])
# df = pd.DataFrame (transformed_df, columns = ['transformed_cases','transformed_death'])
# sns.boxplot(data=df)
# sns.boxplot(imputed_df["Deaths"].apply(np.log))
# plt.show()

# --------------------- COVID Data Pre Processing -------------------
# Missing Values
# missing_covid_values_count = covid_data.isnull().sum()
# missing_covid_values_count

covid_data_filtered_frame = covid_data[['Country/Region', 'Confirmed', 'Deaths', 'Date']]
covid_data_filtered_frame.sample(3)


# Encoder Categorical
# import category_encoders as ce
enc1 = ce.OrdinalEncoder(cols=["Country/Region","Date"],handle_missing='return_nan',return_df= True)

#We now fit the model and transform the data and put it in X which is a dataframe
COVIDX=enc1.fit_transform(covid_data_filtered_frame)
# print(COVIDX.sample(3))


# Outlier Detection In COVID Data
covid_clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_covid_pred = covid_clf.fit_predict(COVIDX)
totalOutliers=0
for pred in y_covid_pred:
    if pred == -1:
        totalOutliers=totalOutliers+1
# print("Number of predicted outliers:", len(covid_data))
# print("Number of predicted outliers:", totalOutliers)

COVIDX['Country/Region'] = covid_data_filtered_frame['Country/Region']
COVIDX['Date'] = covid_data_filtered_frame['Date']
# print(COVIDX[COVIDX['Deaths']>10])

mask3 = (y_covid_pred != -1)
# print(len(preds),len(X))
Cleaned_COVIDFrame = COVIDX[mask3]
# print(len(Cleaned_COVIDFrame))


app = Flask(__name__)

import json
from json import JSONEncoder
import numpy

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:root@localhost:5432/pandemic_db'
db = SQLAlchemy(app)


class COVIDENTRY(db.Model):
    id=db.Column(db.Integer, primary_key=True)
    confirm=db.Column(db.Integer)
    deaths=db.Column(db.Integer)
    country=db.Column(db.String(80))
    date=db.Column(db.Date)

    def __init__(self, confirm, deaths, country, date):
        self.confirm = confirm
        self.deaths = deaths
        self.country = country
        self.date = date

    def __repr__(self):
        return '<User %>' % self.name

class SWINEENTRY(db.Model):
    id=db.Column(db.Integer, primary_key=True)
    confirm=db.Column(db.Integer)
    deaths=db.Column(db.Integer)
    country=db.Column(db.String(80))
    date=db.Column(db.Date)

    def __init__(self, confirm, deaths, country, date):
        self.confirm = confirm
        self.deaths = deaths
        self.country = country
        self.date = date

    def __repr__(self):
        return '<User %>' % self.name


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

@app.route('/load-adult-data-to-db')
def getUser():

    covidEntries = []
    swineEntries = []
    for ind in Cleaned_COVIDFrame.index:
        covidEntries.append(
            COVIDENTRY( 
                confirm= int(covid_data['Confirmed'][ind]),
                deaths= int(covid_data['Deaths'][ind]),
                country=str(covid_data["Country/Region"][ind]),
                date=covid_data['Date'][ind]
            )
        )
    

    for ind in Cleaned_SwineFrame.index:
        # if (pd.isnull(swine_data['Deaths'][ind]) == False):
        swineEntries.append(
            SWINEENTRY( 
                country= str(swine_data['Country'][ind]),
                confirm= int(swine_data['Cases'][ind]),
                deaths=int(swine_data['Deaths'][ind]),
                date=swine_data['Update Time'][ind]
            )
        )
    
    try:
        db.session.bulk_save_objects(covidEntries)
        db.session.bulk_save_objects(swineEntries)
        db.session.commit()
        json = {
            'name':'Entries Added',
        }
        return jsonify(json)
    except Exception as e:
        return (str(e))




# @app.route('/')
# def hello():
#     encodedNumpyData = json.dumps(adult_data.columns.to_numpy(), cls=NumpyArrayEncoder)
#     return jsonify({
#         "csv_columns": json.loads(encodedNumpyData)
#     })

# @app.route('/sum', methods=["GET", "POST"])
# def sum():
#     print(request)
#     data = request.get_json()
#     if "a" not in data:
#         return jsonify({
#             "error": 'A variable not found'
#         })
#     # print(data)
#     return jsonify({
#         "SUM": data["a"] + data["b"]
#     })


# @app.route('/bye')
# def bye():
#     try:
#         users=[
#             User(name= 'Joseph'),
#             User(name= 'Simon'),
#             User(name= 'Fawad'),
#         ]
#         db.session.bulk_save_objects(users)
#         db.session.commit()
#         json = {
#             'name':'Users Added ',
#         }
#         return jsonify(json)
#     except Exception as e:
#         return (str(e))

# @app.route('/get-user')
# def getUser():
#     try:
#         user = User.query.filter_by(name='usama').count()
#         json = {
#             'name': user,
#         }
#         return jsonify(json)
#     except Exception as e:
#         return (str(e))




if __name__ == "__main__":
    app.run(host="127.0.0.1", port=80, debug=True)
