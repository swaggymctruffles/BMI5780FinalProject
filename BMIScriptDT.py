# %%
import time
import pandas as pd
import numpy as np
import random as rn

# Standardize the data
from sklearn.preprocessing import StandardScaler

# Modeling 
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import label_binarize

# Hyperparameter tuning
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score

# Plot
import matplotlib.pyplot as plt

# Model evaluation
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, auc, roc_auc_score
from sklearn import metrics
start_time = time.time()

df = pd.read_csv('readdata.csv', sep=',', low_memory=False)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
df

# create a dictionary to map old column names to new column names
column_mapping = {
    'SEQN': 'Sequence Number',
    'H1CD0006': 'Final Estimation Weight',
    'H1CD0012': 'Headaches',
    'H1CD0013': 'Nose Bleeds',
    'H1CD0014': 'Tinnitus',
    'H1CD0015': 'Dizziness',
    'H1CD0016': 'Fainting',
    'H1CD0017': 'Stroke',
    'H1CD0018': 'Stroke Review Summary',
    'H1CD0019': 'Paralyzed',
    'H1CD0020': 'Sore Throats',
    'H1CD0021': 'Shortness of Breath When Climbing Stairs',
    'H1CD0022': 'Shortness of Breath When Doing Physi ...',
    'H1CD0023': 'Shortness of Breath When Idle',
    'H1CD0024': 'Shortness of Breath When Excited or  ...',
    'H1CD0025': 'Wake Up at Night Because of Shortnes ...',
    'H1CD0026': 'Chest Pain-In Past Few Years',
    'H1CD0027': 'Chest Pain-Where Does It Bother You?',
    'H1CD0028': 'Chest Pain-Does It Move Around?',
    'H1CD0029': 'Chest Pain-Duration?',
    'H1CD0030': 'Chest Pain-When Does It Usually Come?(exercise)',
    'H1CD0031': 'Chest Pain-When Does It Usually Come?(mood)',
    'H1CD0032': 'Chest Pain-Take Medication for It?',
    'H1CD0033': 'Heart Pain-In Part Few Year?',
    'H1CD0034': 'Heart Pain-Where Does It Bother You?',
    'H1CD0035': 'Heart Pain-Does It Move Around:',
    'H1CD0036': 'Heart Pain-Duration?',
    'H1CD0037': 'Heart Pain-When Does It Usually Come?(exercise)',
    'H1CD0038': 'Heart Pain-When Does It Usually Come?(mood)',
    'H1CD0039': 'Hear Pain-Late Medication for It?',
    'H1CD0040': 'Heart Beat-Acting Funny?',
    'H1CD0041': 'Heart Beat-Beating Hard?',
    'H1CD0042': 'Swollen Ankles at Bedtime?',
    'H1CD0043': 'Leg Cramps?',
    'H1CD0044': 'Varicose Veins?',
    'H1CD0045': 'Rheumatic Fever Diagnosis?',
    'H1CD0046': 'Chorea or St. Vitus Dance Diagnosis?',
    'H1CD0047': 'Hardening of the Arteries Diagnosis?',
    'H1CD0048': 'High Blood Pressure-Think You Have It?',
    'H1CD0049': 'High Blood Pressure- How Long Ago Di ...',
    'H1CD0050': 'High Blood Pressure-Had It Within Pa ...',
    'H1CD0051': 'High Blood Pressure-Take Medication?',
    'H1CD0052': 'Heart Trouble-Think You Have It?',
    'H1CD0053': 'Heart Trouble-How Long Did It Start?',
    'H1CD0054': 'Heart Trouble-Had It Within Past 12  ...',
    'H1CD0055': 'Heart Trouble-Take Medication?',
    'H1CD0056': 'Time of Examination(01-21)',
    'H1CD0058': 'Average Systolic Blood Pressure',
    'H1CD0061': 'Average Diastolic Blood Pressure',
    'H1CD0064': 'Average Systolic Blood Pressure Recode',
    'H1CD0066': 'Average Diastolic Blood Pressure Recode',
    'H1CD0068': 'When Venipuncture Was Done in Relati ...',
    'H1CD0069': 'Ocular Fundi-Right Eye',
    'H1CD0070': 'Ocular Fundi-Left Eye',
    'H1CD0071': 'Ocular Fundi Condition-Right Eye',
    'H1CD0072': 'Ocular Fundi Condition-Left Eye',
    'H1CD0073': 'Venous Engorgement-Right Eye',
    'H1CD0074': 'Venous Engorgement-Left Eye',
    'H1CD0075': 'Disc. Abnormality-Right Eye',
    'H1CD0076': 'Disc. Abnormality-Left Eye',
    'H1CD0077': 'Lens Opacities-Right Eye',
    'H1CD0078': 'Lens Opacities-Left Eye',
    'H1CD0079': 'Ocular Fundi-Other-Right Eye',
    'H1CD0080': 'Ocular Fundi-Other -Left Eye',
    'H1CD0081': 'Neck-Venous Engorgement',
    'H1CD0082': 'Peripheral Arteries-All Normal',
    'H1CD0083': 'Periphearal Arteries-Right Side-Supe ...',
    'H1CD0084': 'Periphearal Arteries-Left Side-Brachial ',
    'H1CD0085': 'Peripheral Arteries-Right Side-Radial',
    'H1CD0086': 'Peripheral Arteries-Left Side Superf ...',
    'H1CD0087': 'Peripheral Arteries-Left Side  Brachial',
    'H1CD0088': 'Peripheral Arteries-Left Side-Radical',
    'H1CD0089': 'Quality of Aterial Pulsations-All Normal',
    'H1CD0090': 'Quality of Arterial Pulsations-Right 1...',
    'H1CD0091': 'Quality of Arterial Pulsations-Right 2...',
    'H1CD0092': 'Quality of Arterial Pulsations-Right 3...',
    'H1CD0093': 'Quality of Arterial Pulsations-Left  1...',
    'H1CD0094': 'Quality of Arterial Pulsations-Left  2...',
    'H1CD0095': 'Quality of Arterial Pulsations-Left  3...',
    'H1CD0096': 'Lower Extremities-Right',
    'H1CD0097': 'Lower Extremities-Left',
    'H1CD0098': 'Lower Extremities-Right-Varicosities',
    'H1CD0099': 'Lower Extremities-Right-Dependent Edema',
    'H1CD0100': 'Lower Extremities-Right Ulcers',
    'H1CD0101': 'Lower Extremities-Left-Varicosities',
    'H1CD0102': 'Lower Extremities-Left-Dependent Edema',
    'H1CD0103': 'Lower Extremities-Left-Ulcers',
    'H1CD0104': 'Thrills',
    'H1CD0105': 'Apical Impulse',
    'H1CD0106': 'Apical Impulse-Interspace (As Given)',
    'H1CD0107': 'Heart Sounds-Normal Chocked',
    'H1CD0108': 'Heart Sounds-A2',
    'H1CD0109': 'Heart Sounds-P2',
    'H1CD0110': 'Heart Sounds-M1',
    'H1CD0111': 'Heart Sounds-Third Heart Sound',
    'H1CD0112': 'Heart Sounds-Spilling of Second Soun ...',
    'H1CD0113': 'Heart Sounds-Other',
    'H1CD0114': 'Significant Murmurs-Type of Systolic ...',
    'H1CD0115': "SIGNIFICANT MURMURS- PHYSICIAN'S IMP ...",
    'H1CD0116': "EXAMINING PHYSICIAN'S IMPRESSION-HYP ...",
    'H1CD0117': "EXAMINING PHYSICIAN'S IMPRESSION-PER ...",
    'H1CD0118': "EXAMINING PHYSICIAN'S IMPRESSION-ORG ...",
    'H1CD0119': "EXAMINING PHYSICIAN'S IMPRESSION-ANG ...",
    'H1CD0120': "EKG-SUMMARY OF 3 READINGS 1",
    'H1CD0122': "EKG-SUMMARY OF 3 READINGS 2",
    'H1CD0123': "EKG-SUMMARY OF 3 READINGS 3",
    'H1CD0124': "EKG-SUMMARY OF 3 READINGS 4",
    'H1CD0125': "EKG-SUMMARY OF 3 READINGS 5",
    'H1CD0126': "EKG-SUMMARY OF 3 READINGS 6",
    'H1CD0127': "EKG-SUMMARY OF 3 READINGS 7",
    'H1CD0128': "EKG-SUMMARY OF 3 READINGS 8",
    'H1CD0129': "EKG-SUMMARY OF 3 READINGS 9",
    'H1CD0130': "EKG-SUMMARY OF 3 READINGS 10",
    'H1CD0131': "EKG-SUMMARY OF 3 READINGS 11",
    'H1CD0132': "EKG-READING 1 1",
    'H1CD0134': "EKG-READING 1 2",
    'H1CD0135': "EKG-READING 1 3",
    'H1CD0136': "EKG-READING 1 4",
    'H1CD0137': "EKG-READING 1 5",
    'H1CD0138': "EKG-READING 1 6",
    'H1CD0139': "EKG-READING 1 7",
    'H1CD0140': "EKG-READING 1 8",
    'H1CD0141': "EKG-READING 1 9",
    'H1CD0142': "EKG-READING 1 10",
    'H1CD0143': "EKG-READING 1 11",
    'H1CD0144': "EKG-READING 2 1",
    'H1CD0145': "EKG-READING 2 2",
    'H1CD0146': "EKG-READING 2 3",
    'H1CD0147': "EKG-READING 2 4",
    'H1CD0148': "EKG-READING 2 5",
    'H1CD0149': "EKG-READING 2 6",
    'H1CD0150': "EKG-READING 2 7",
    'H1CD0151': "EKG-READING 2 8",
    'H1CD0152': "EKG-READING 2 9",
    'H1CD0153': "EKG-READING 2 10",
    'H1CD0154': "EKG-READING 2 11",
    'H1CD0155': "EKG-READING 2 12",
    'H1CD0156': "EKG-READING 3 1",
    'H1CD0158': "EKG-READING 3 2",
    'H1CD0159': "EKG-READING 3 3",
    'H1CD0160': "EKG-READING 3 4",
    'H1CD0161': "EKG-READING 3 5",
    'H1CD0162': "EKG-READING 3 6",
    'H1CD0163': "EKG-READING 3 7",
    'H1CD0164': "EKG-READING 3 8",
    'H1CD0165': "EKG-READING 3 9",
    'H1CD0166': "EKG-READING 3 10",
    'H1CD0167': "EKG-READING 3 11",
    'H1CD0168': "EKG-READING-AVERAGE RATE",
    'H1CD0171': "EKG READING-AVERAGE PR INTERVAL",
    'H1CD0173': "EKG READING-AVERAGE INTERVAL",
    'H1CD0175': "S-SAVE DEPTH IN V1 (SV1)",
    'H1CD0178': "R-SAVE DEPTH IN V5 (SVS)",
    'H1CD0181': "CHEST X-RAY-EXISTENCE OF LESION-READ 1...",
    'H1CD0182': "CHEST X-RAY-HEART ENLARGEMENT-READING 1",
    'H1CD0183': "CHEST X-RAY-OTHER CV ABNORMALITY-REA 1...",
    'H1CD0185': "CHEST X-RAY-EXISTENCE OF LESION-READ 2...",
    'H1CD0186': "CHEST X-RAY-HEART ENLARGEMENT-READING 2",
    'H1CD0187': "CHEST X-RAY-OTHER CV ABNORMALITY-REA 2...",
    'H1CD0189': "CHEST X-RAY-EXISTENCE OF LESION-READ 3...",
    'H1CD0190': "CHEST X-RAY-HEART ENLARGEMENT-READING 3",
    'H1CD0191': "CHEST X-RAY-OTHER CV ABNORMALITY-REA 3...",
    'H1CD0193': "CHEST X-RAY-FINAL EVALUATION OF HEAR ...",
    'H1CD0194': "SERUM CHOLESTEROL VALUES (MG%)",
    'H1CD0196': "STROKE DIAGNOSIS",
    'H1CD0197': "CARDIOVASCULAR CONDITION DIAGNOSIS",
    'H1CD0198': "HEART CONDITION DIAGNOSIS",
    'H1CD0199': "HIGH BLOOD PRESSURE DIAGNOSIS",
    'H1CD0200': "VARICOSE VEINS DIAGNOSIS",
    'H1CD0201': "CORONARY HEART DISEASE DIAGNOSIS 1",
    'H1CD0202': "RHEUMATIC HEART DISEASE DIAGNOSIS",
    'H1CD0203': "HYPERTENSIVE HEART DISEASE DIAGNOSIS",
    'H1CD0204': "HEART DISEASE SUMMARY (IN DESCENDING ...",
    'H1CD0205': "BLOOD PRESSURE SUMMARY",
    'H1CD0206': "HYPERTENSIVE HEART DISEASE SUMMARY",
    'H1CD0207': "RHEUMATIC HEART DISEASE SUMMARY",
    'H1CD0208': "CORONARY HEART DISEASE SUMMARY",
    'H1CD0209': "OTHER HEART DISEASE SUMMARY",
    'H1CD0210': "HEART DISEASE SUMMARY",
    'H1CD0211': "POSSIBLE HEART DISEASE",
    'H1CD0212': "HEART DISEASE DIAGNOSIS",
    'H1CD0213': "CORONARY HEART DISEASE DIAGNOSIS",
    'H1CD0214': "HYPERTENSIVE HEART DISEASE DIAGNOSIS 2",
    'H1CD0215': "HYPERTENSION DIAGNOSIS",
    'H1CD0216': "HYPERTENSION DIAGNOSIS BY PERSONAL P ...",
    'H1CD0217': "PERIPHERAL VASCULAR DISEASE  DIAGNOS ...",
    'H1CD0218': "CORONARY HEART DISEASE DIAGNOSIS BY  ...",
    'H1CD0219': "HYPERTENSIVE HEAR DISEASE DIAGNOSIS  ...",
    'H1CD0220': "RHEUMATIC HEAR DISEASE DIAGNOSIS BY  ...",
    'H1CD0221': "OTHER HEART DISEASE DIAGNOSIS BY PER ...",
    'H1CD0222': "UNUSED-ALL BLANKS"
}

# rename columns using the dictionary
df = df.rename(columns=column_mapping)
df = df.drop(['UNUSED-ALL BLANKS'], axis=1) # remove unused column
df = df.drop(['EKG-READING 2 1'], axis=1) # drops the extra column, seems to be a 'whether or not they got it' column

numerical_columns = ['Time of Examination(01-21)','Average Systolic Blood Pressure','Average Diastolic Blood Pressure',"EKG-READING-AVERAGE RATE","EKG READING-AVERAGE PR INTERVAL","EKG READING-AVERAGE INTERVAL","S-SAVE DEPTH IN V1 (SV1)","R-SAVE DEPTH IN V5 (SVS)"]

for column in numerical_columns:
    df[column] = df[column].astype(int)  # Convert to int
    df[column] = df[column].replace(999, pd.NA) # Replace 999s with NaN, more info in documentation of dataset
    df[column] = df[column].replace(99 , pd.NA) # were about 25 empty spots with &, replaced in excel with 999

df[numerical_columns] = df[numerical_columns].fillna(df.median())  # Replace NaN with median of each column
#print(df[numerical_columns].median()) # print median of each column
    
end_time = time.time()
elapsed_time = end_time - start_time
print('Data read.')
print(f"Elapsed time: {elapsed_time} seconds")

# Drop diagnoses so they don't interfere with the model, these can be predicted in future runs with some tweaking
df = df.drop(["STROKE DIAGNOSIS",
    "CARDIOVASCULAR CONDITION DIAGNOSIS",
    #"HEART CONDITION DIAGNOSIS", # analysis of this is in the paper, uncomment if you want to analyze others
    "HIGH BLOOD PRESSURE DIAGNOSIS",
    "VARICOSE VEINS DIAGNOSIS",
    "CORONARY HEART DISEASE DIAGNOSIS 1",
    "RHEUMATIC HEART DISEASE DIAGNOSIS",
    "HYPERTENSIVE HEART DISEASE DIAGNOSIS"],axis=1)
# Drop summaries so they don't interfere with the model, comment lines if you want to analyze
df = df.drop(["HEART DISEASE SUMMARY (IN DESCENDING ...","BLOOD PRESSURE SUMMARY",
              "HYPERTENSIVE HEART DISEASE SUMMARY","RHEUMATIC HEART DISEASE SUMMARY",
              "CORONARY HEART DISEASE SUMMARY","OTHER HEART DISEASE SUMMARY",
              "HEART DISEASE SUMMARY","POSSIBLE HEART DISEASE","HEART DISEASE DIAGNOSIS",
              "CORONARY HEART DISEASE DIAGNOSIS","HYPERTENSIVE HEART DISEASE DIAGNOSIS 2",
              "HYPERTENSION DIAGNOSIS","HYPERTENSION DIAGNOSIS BY PERSONAL P ...",
              "PERIPHERAL VASCULAR DISEASE  DIAGNOS ...",
              "CORONARY HEART DISEASE DIAGNOSIS BY  ...",
              "HYPERTENSIVE HEAR DISEASE DIAGNOSIS  ...",
              "RHEUMATIC HEAR DISEASE DIAGNOSIS BY  ...",
              "OTHER HEART DISEASE DIAGNOSIS BY PER ..."], axis=1)

seed = 5780
np.random.seed(seed)
rn.seed(seed)

column_to_predict = 'HEART CONDITION DIAGNOSIS'

# split into X and y
Y_data = df[column_to_predict]
X_data = df.drop([column_to_predict], axis=1)
Y_data = Y_data.astype('float32')
X_data = X_data.astype('float32')

# Standardize numerical columns
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

print(Y_data.shape)
print(X_data.shape)

#test and train sets, 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=seed)

print(f'The training set has {len(X_train)} samples and the test set has {len(X_test)} samples.')

end_time = time.time()
elapsed_time = end_time - start_time
print('Data split.')
print(f"Elapsed time: {elapsed_time} seconds")

# Decision Tree
from sklearn.metrics import make_scorer, accuracy_score

def delayed_accuracy(y_true, y_pred): # This is to delay the accuracy score
    time.sleep(0)  # delay so my CPU doesn't cook itself
    return accuracy_score(y_true, y_pred) # return the same thing as before, just delayed
delayed_scorer = make_scorer(delayed_accuracy)

dt = DecisionTreeClassifier()

# Hyperparameter search for DT

'''dt_param_grid = { # match all
    'criterion':  ['gini', 'entropy'],
    'max_depth':  [None, 2, 4, 6, 8, 10, 20, 30, 50, 100],
    'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
    'splitter': ['best', 'random']
    }'''
dt_param_grid = { # only matches best for speed
    'criterion':  ['entropy'],
    'max_depth':  [None],
    'max_features': [None],
    'splitter': ['random']
    }

# change verbose to any interger if you want to print out the message
dt_grid_search = GridSearchCV(estimator=dt, param_grid=dt_param_grid, n_jobs = -1, cv = 6, verbose=3, scoring=delayed_scorer)

# SMOTE in training data
sm = SMOTE(sampling_strategy=1.0,random_state=seed)
x_train_s, y_train_s = sm.fit_resample(X_train, y_train)

# Print before SMOTE
print("Before OverSampling, counts of label '1 (No HD)': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '2 (HD)': {} \n".format(sum(y_train == 2)))
  
# Print after SMOTE
print('After OverSampling, the shape of train_X: {}'.format(x_train_s.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_s.shape))
  
print("After OverSampling, counts of label 'No Heart Disese': {}".format(sum(y_train_s == 1)))
print("After OverSampling, counts of label 'Heart Disease': {}".format(sum(y_train_s == 2)))

# Fit the model using training data after SMOTE
dt_grid_search.fit(x_train_s, y_train_s)

dt_best_grid = dt_grid_search.best_estimator_
dt_best_params = dt_grid_search.best_params_

# 5. Model evaluation and plotting

yscore_raw = dt_best_grid.predict_proba(X_test)
yscore = [s[1] for s in yscore_raw]
fpr, tpr, thresh = roc_curve(y_test, yscore, pos_label=2)
#auc = roc_auc_score(y_test, yscore)

# Calculate the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)

#create ROC curve
plt.plot(fpr, tpr, label='%s ROC (area = %0.3f)' % ('Decision Tree', roc_auc))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic-Decision Tree')
plt.legend(loc="lower right")
plt.savefig('Decision Tree_MOF.png')

# print the best hyperparameters
print(f'The best parameters are {dt_best_params}')

# The best parameters are {'criterion': 'entropy', 'max_depth': None, 'max_features': None, 'splitter': 'random'}

y_hat = dt_best_grid.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_hat)

# Extract the true positives, true negatives, false positives, and false negatives
tn, fp, fn, tp = cm.ravel()

# Calculate sensitivity (true positive rate)
sensitivity = tp / (tp + fn)

# Calculate specificity (true negative rate)
specificity = tn / (tn + fp)

# Calculate accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)

# Print the sensitivity, specificity, and accuracy
print(f"Sensitivity: {sensitivity:.2f}") #
print(f"Specificity: {specificity:.2f}") #
print(f"Accuracy: {accuracy:.2f}") #


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

# %%

# From Copilot - Feature Importance
from sklearn.inspection import permutation_importance

# Assume X_val is your validation data and y_val are the corresponding labels
result = permutation_importance(dt_best_grid, x_train_s, y_train_s, n_repeats=4, random_state=seed)

# Get importance scores
importances = result.importances_mean

# Get feature names
feature_names = x_train_s.columns.tolist()

# Print feature importance
for feature_name, importance in zip(feature_names, importances):
    print(f"{feature_name}: {importance}")
combo = sorted(zip(importances,feature_names),reverse=True)

# Print 10 most important features
for i in range(1,11):
    print(f'Rank #{i}: {combo[i]}')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

# %%