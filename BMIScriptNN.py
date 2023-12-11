
import pandas as pd
import time
import numpy as np
start_time = time.time()

df = pd.read_csv('readdata.csv', sep=',', low_memory=False)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
df

nan_rows = df[df.isnull().any(axis=1)]
#print(nan_rows.head())

nan_cols_labels = df.columns[df.isnull().any()]
#print(nan_cols_labels)


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


ndf = df.copy() # create a copy of the dataframe to add names to, use df for ML
#df.to_csv('readdata.csv', index=False) # save to csv file

def dictmerge(d1, d2): # dictionaries cannot be concat, so this is a workaround
    d3 = d1.copy()
    d3.update(d2)
    return d3


# creates a dict for all categoricals, not strictly necessary but makes it easier to read later
# copilot rewrite: keeping comments, make codes a dict with integer keys 

# defines different meanings of catergoricals, originally done in lists but some are switched to dict as they were causing problems
codes3  = {1:'Yes, every few days, severe',2:'Yes, every few days, not severe',3:'Yes, less often, severe',
        4:'Yes, less often, not severe',5:'No or ?',7:'other'} # headaches, 7 instead of 9? not sure why
codes4  = {1:'Yes, every few days, severe',2:'Yes, every few days, not severe',3:'Yes, less often, severe',
        4:'Yes, less often, not severe',5:'No or ?',9:'other'} # nose bleeds, tinnitus, dizziness, fainting, sore throats etc
codes5  = codes4
codes6  = codes4 
codes7  = {1:'Yes',2:'No',3:'?',9:'Other'}
codes8  = {1: 'Yes, past 12 months, saw doctor', 2: 'Yes, past 12 months, no doctor or ?', 
                  3: 'Yes, not in past 12 months, saw doctor', 4: 'Yes, not in past 12 months, no doctor or ?', 
                  5: 'No or ?', 9:'other'} # stroke diagnosis
codes9  = {1: 'History and physical findings', 2: 'History only', 3: 'Negative'} # stroke review summary
codes10 = {1:'Yes',2:'No',3:'?',9:'Other'}
codes11 = codes10
codes12 = codes4
codes13 = codes4
codes14 = codes4
codes15 = codes4
codes16 = codes4
codes17 = codes4
codes18 = {1: 'Front', 2: 'Back', 3: 'Right Side', 4: 'Middle', 5: 'Left Side', 6: 'Somewhere Else', 7: 'More than one place', 9: 'Other', 0: 'No chest pain'} # chest pain location
codes19 = {1: 'Stays in one place', 2: 'Moves around', 3: '?', 9: 'Other', 0: 'No chest pain'}
codes20 = {1: 'Just a few minutes', 2: 'Five minutes to an hour', 3: 'More than an hour', 9: 'Other', 0: 'No chest pain'}
codes21 = {1: 'When exercising', 2: 'When quiet', 3: 'Makes no difference', 9: 'Other', 0: 'No chest pain'} # chest pain trigger
codes22 = {1: 'When upset', 2: 'Makes no difference', 9: 'Other', 0: 'No chest pain'} # chest pain trigger 2
codes23 = {1: 'Yes', 2: 'No', 3: '?', 9: 'Other', 0: 'No chest pain'}
codes24 = codes4
codes25 = codes18
codes26 = codes19
codes27 = codes20
codes28 = codes21
codes29 = codes22
codes30 = codes23
codes31 = codes4
codes32 = codes4
codes33 = {1: 'Yes, gone by morning', 2: 'Yes, not gone by morning', 3: 'Yes, ? gone by morning', 4: 'No', 5: '?', 9: 'Other'}
codes34 = codes4
codes35 = codes8 # varicose veins diagnosis
codes36 = {1: 'Yes, past 12 months, taking medication', 2: 'Yes, past 12 months, no medication or ?', 3: 'Yes, not in past 12 months, taking medication', 4: 'Yes, not in past 12 months, no medication or ?', 5: 'No or ?'} # Rheumatic fever diagnosis
codes37 = {1: 'Yes', 2: 'No', 3: 'Other'} # Chorea or St. Vitus Dance diagnosis
codes38 = {1: 'Yes, within past 12 months', 2: 'Yes, not within past 12 months', 3: 'Yes, ? within past 12 months', 4: 'No', 5: '?', 9: 'Other'} # Hardening of the arteries diagnosis
codes39 = {1: 'Yes, confirmed by doctor', 2: 'Yes, not confirmed by doctor', 3: 'No', 4: '?', 9: 'Other'} # High blood pressure - think you have it?
codes40 = {1: '1 year', 2: '1-5 years', 3: '>5 years', 9: 'other', 0: 'No HPB'}
codes41 = {1: 'Yes', 2: 'No', 3: '?', 9: 'Other', 0: 'No HPB'}
codes42 = {1: 'Yes, taking medication', 2: 'Not taking medication', 3: '?', 9: 'Other', 0: 'No HPB'}
codes43 = codes39 # Heart trouble - think you have it?
codes44 = codes40 # Heart trouble - how long ago diagnosed?
codes45 = {1: 'Yes', 2: 'No', 3: '?', 9: 'Other', 0: 'No heart trouble'}
codes46 = {1: 'Yes, taking medication', 2: 'Not taking medication', 3: '?', 9: 'Other', 0: 'No heart trouble'}
codes47 = 0 # Time of examination, do not turn to categorical
codes48 = 0 # avg systolic blood pressure, do not turn to categorical
codes49 = 0 # avg diastolic blood pressure, do not turn to categorical
codes50 = {1: '<90', 2: '90-100', 3: '100-109', 4: '110-119', 5: '120-129', 6: '130-139', 7: '140-149', 8: '150-159', 9: '160-169', 10: '170-179', 11: '180-189', 12: '190-199', 13: '200-209', 14: '210-219', 15: '220-229', 16: '230-239', 17: '240-249', 18: '250-259', 19: '>260'} # Avg systolic blood pressure coded
codes51 = {1: '<50', 2: '50-54', 3: '55-59', 4: '60-64', 5: '65-69', 6: '70-74', 7: '75-79', 8: '80-84', 9: '85-89', 10: '90-94', 11: '95-99', 12: '100-104', 13: '105-109', 14: '110-114', 15: '115-119', 16: '120-124', 17: '125-129', 18: '130-134', 19: '>135'} # Avg diastolic blood pressure coded
codes52 = {1: 'Before first BP', 2: 'Between 1st and 2nd BP', 3: 'Between 2nd and 3rd BP', 4: 'After all BP', 0: 'other'} # When venipuncture was done in relation to BP readings
codes53 = {1:'Normal',2:'Fundus not visualized',3:'Globe absent',4:'None of the above'} # Ocular fundi - right eye
codes54 = codes53 # Ocular fundi - left eye
codes55 = {1: 'Papilledepa', 2: 'Hemmorage and/or Exudates', 3: 'Other positives', 4: 'No abnormality', 5: 'None/Unknown'}
codes56 = codes55 # Ocular fundi condition - left eye
codes57 = {1:'Present',2:'Not present'} # Venous engorgement - right eye
codes58 = codes57 # Venous engorgement - left eye
codes59 = codes57 # Disc abnormality - right eye
codes60 = codes57 # Disc abnormality - left eye
codes61 = codes57 # Lens opacities - right eye
codes62 = codes57 # Lens opacities - left eye
codes63 = codes57 # Ocular fundi - other - right eye
codes64 = codes57 # Ocular fundi - other - left eye
codes65 = {1:'Yes',2:'No',9:'other',0:'no entry'} # Neck - venous engorgement
codes66 = {1:'Yes',2:'No',9:'other',0:'no entry'}  # Peripheral arteries - all normal
codes67 = {1: 'Normal', 2: 'Scleratic only', 3: 'Tortucus only', 4: 'Both', 5: 'Not done', 9: 'other', 0: 'All normal'} # Peripheral arteries - right side - superficial temporal
codes68 = codes67 # Peripheral arteries - right brachial
codes69 = codes67 # Peripheral arteries - right radial
codes70 = codes67 # Peripheral arteries - left superficial temporal
codes71 = codes67 # Peripheral arteries - left brachial
codes72 = codes67 # Peripheral arteries - left radial
codes73 = {1:'Yes',2:'No'} # Quality of arterial pulsations - all normal
codes74 = {0: 'All Normal', 1: 'Norral(?)', 2: 'Bounding', 3: 'Diminished', 4: 'Not palpable', 5: 'Not done', 9: 'other'}
codes75 = codes74 # Peripheral arteries - right side - dorsalis pedis
codes76 = codes74 # Peripheral arteries - right side - posterior tibial
codes77 = codes74 # Peripheral arteries - left side - radial
codes78 = codes74 # Peripheral arteries - left side - dorsalis pedis
codes79 = codes74 # Peripheral arteries - left side - posterior tibial
codes80 = {1:'Normal',2:'Not done',3:'Neither'} # Lower Extremities - right
codes81 = codes80 # Lower Extremities - left
codes82 = {1:'Checked',2:'Not checked',3:'All normal'} # Lower Extremities - right - varicosities
codes83 = codes82 # Lower Extremities - right - dependent edema
codes84 = codes82 # Lower Extremities - right - ulcers
codes85 = codes82 # Lower Extremities - left - varicosities
codes86 = codes82 # Lower Extremities - left - dependent edema
codes87 = codes82 # Lower Extremities - left - ulcers
codes88 = {1: 'Aortic Systolic', 2: 'Apical Systolic', 3: 'Apical Diastolic', 4: 'Pulmonic Systolic', 5: 'other', 6: 'None', 0: 'No entry'} # Heart thrills
codes89 = {1: 'Not felt', 2: 'MCL at or inside', 3: 'MCL outside', 0: 'No entry'} # Apical impulse
codes90 = {1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 0: 'No entry'} # Apical impulse - interspace
codes91 = {1:'Normal',2:'No, no other entries checked',3:'No, other entries checked'} # Heart sounds - normal checked (like the box)
codes92 = {1:'Accentuated',2:'Diminished',3:'No entry',0:'All normal'} # Heart sounds - A2
codes93 = codes92 # Heart sounds - P2
codes94 = codes92 # Heart sounds - M1
codes95 = {1:'Present',2:'Absent',0:'All normal'} # Heart sounds - third heart sound
codes96 = codes95 # Heart sounds - spilling of second sound abnormal
codes97 = codes95 # Heart sounds - other (like other things present)
codes98 = {1: 'Grade 3', 2: 'Grade 4', 3: 'Murmur with thrill at base', 4: 'Others', 0: 'None'}  # Significant murmurs - type of systolic murmur
codes99 = {1: 'Pheumatic', 2: 'Congenital', 3: 'Aortic Stenosis', 4: 'Other', 5: 'No HD', 0: 'If codes 4,0 in item 98'}
codes100 = {1:'Positive',2:'Negative',3:'Suspect',0:'No entry'} # Examining physician's impression - hypertenion
codes101 = codes100 # Examining physician's impression - peripheral arteriosclerosis
codes102 = codes100 # Examining physician's impression - organic heart disease
codes103 = codes100 # Examining physician's impression - angina pectoris
codes104 = {
    1: 'Normal EKG',
    2: 'Unsatisfactory EKG',
    3: 'No myocardial infarction',
    4: 'Anterior myocardial infarction',
    5: 'Posterior myocardial infarction',
    6: 'Anterior myocardial infarction outside criteria',
    7: 'Posterior myocardial infarction outside criteria',
    8: 'Ant and Post Myocardial Infarction',
    10: 'other'
} # EKG - summary of 3 readings

# for below bits
code105 = {1:'No axis deviation',2:'Left axis deviation',3:'Right axis deviation',4:'Left axis outside criteria',5:'Right axis outside criteria'} 
code106 = {1:'No ventricular hypertrophy',2:'Left ventricular hypertrophy',3:'Right ventricular hypertrophy',4:'Left VH outside criteria',
        5:'Right " " outside criteria',6:'Both L/R VH'}
code107 = {
    1: 'No abnormalities on ST segment or junction',
    2: 'Subendocardial Ischemia',
    3: 'Subendocardial Ischemia/digitalis',
    4: 'Current of injury',
    5: 'Subendocardial ischemia outside criteria',
    6: 'Subendocardial Ischemia/digitalis outside criteria',
    7: 'Current of injury outside criteria',
    8: 'other'
}
code108 = {
    1: 'No abnormal T-wave',
    2: 'Non-specific T-Wave',
    3: 'Left ventricular ischemia',
    4: 'Non-specific T-wave, outside criteria',
    5: 'Left vent. Ischemia outside criteria'
}
code109 = {
    1: 'No AV conduction abnormalities',
    2: 'Complete AV block',
    3: 'Partial AV block',
    4: 'First degree AV block',
    5: 'WPW',
    6: 'WPW outside criteria',
    7: 'other'
}
code110 = {
    1: 'No ventricular conduction abnormalities',
    2: 'Left bundle branch block',
    3: 'Right bundle branch block',
    4: 'Increased right bundle branch block',
    5: 'I-V block',
    6: 'Left bundle block and I-V block',
    7: 'Incr. right bundle branch block outside criteria',
    8: 'other'
}
code111 = {
    1: 'No arrhythmias',
    2: 'Ventricular tachycardia',
    3: 'Auricular fibrillation',
    4: 'Aur. Hod(?). or supra-vent tachycardia',
    5: 'Abnormal ventricular rhythm',
    6: 'Abnormal nodal rhythm',
    7: 'other',
}
code112 = {
    1: 'Neither low QRS or high T-wave',
    2: 'Low QRS',
    3: 'High T-wave',
    4: 'Both'
}  # EKG - summary of 3 readings 
code113 = {
    1: 'No premature contractions',
    2: 'Rare atrial premature contractions',
    3: 'Rare ventricular premature contractions',
    4: 'Rare nodal premature contractions',
    5: 'Frequent atrial premature contractions',
    6: 'Frequent ventricular premature contractions',
    7: 'Frequent nodal premature contractions',
    8: 'other'
}
code114 = {1:'No misc findings',2:'Misc findings'} # EKG - summary of 3 readings # EKG - reading 1
code115 = dictmerge(codes104, {11:'Not examined by this reader'}) # EKG - summary of 3 readings
code116 = dictmerge(code105, {0:'Codes 1,2,10,or 11 in item 115'}) # EKG - summary of 3 readings
# continue

matching104 = {0:'Codes 1,2,or 10 in item 104'}
matching115 = {0:'Codes 1,2,10,or 11 in item 115'}
matching126 = {0:'Codes 1,2,10,or 11 in item 126'}
matching137 = {0:'Codes 1,2,10,or 11 in item 137'}
codes105 = dictmerge(matching104, code105) # EKG - summary of 3 readings
codes106 = dictmerge(matching104, code106) # EKG - summary of 3 readings
codes107 = dictmerge(matching104, code107) # EKG - summary of 3 readings
codes108 = dictmerge(matching104, code108) # EKG - summary of 3 readings
codes109 = dictmerge(matching104, code109) # EKG - summary of 3 readings
codes110 = dictmerge(matching104, code110) # EKG - summary of 3 readings
codes111 = dictmerge(matching104, code111) # EKG - summary of 3 readings
codes112 = dictmerge(matching104, code112) # EKG - summary of 3 readings
codes113 = dictmerge(matching104, code113) # EKG - summary of 3 readings
codes114 = dictmerge(matching104, code114) # EKG - summary of 3 readings
codes115 = code115 # EKG - reading 1
codes116 = dictmerge(matching115, code105) # EKG - reading 1
codes117 = dictmerge(matching115, code106) # EKG - reading 1
codes118 = dictmerge(matching115, code107) # EKG - reading 1
codes119 = dictmerge(matching115, code108) # EKG - reading 1
codes120 = dictmerge(matching115, code109) # EKG - reading 1
codes121 = dictmerge(matching115, code110) # EKG - reading 1
codes122 = dictmerge(matching115, code111) # EKG - reading 1
codes123 = dictmerge(matching115, code112) # EKG - reading 1
codes124 = dictmerge(matching115, code113) # EKG - reading 1
codes125 = dictmerge(matching115, code114) # EKG - reading 1
codes126 = dictmerge(codes104, {11:'Not examined by this reader'}) # EKG - reading 2
codes127 = dictmerge(matching126, codes105) # EKG - reading 2
codes128 = dictmerge(matching126, codes106) # EKG - reading 2
codes129 = dictmerge(matching126, codes107) # EKG - reading 2
codes130 = dictmerge(matching126, codes108) # EKG - reading 2
codes131 = dictmerge(matching126, codes109) # EKG - reading 2
codes132 = dictmerge(matching126, codes110) # EKG - reading 2
codes133 = dictmerge(matching126, codes111) # EKG - reading 2
codes134 = dictmerge(matching126, codes112) # EKG - reading 2
codes135 = dictmerge(matching126, codes113) # EKG - reading 2
codes136 = dictmerge(matching126, codes114) # EKG - reading 2
codes137 = dictmerge(codes104, {11:'Not examined by this reader'}) # EKG - reading 3
codes138 = dictmerge(matching137, codes105) # EKG - reading 3
codes139 = dictmerge(matching137, codes106) # EKG - reading 3
codes140 = dictmerge(matching137, codes107) # EKG - reading 3
codes141 = dictmerge(matching137, codes108) # EKG - reading 3
codes142 = dictmerge(matching137, codes109) # EKG - reading 3
codes143 = dictmerge(matching137, codes110) # EKG - reading 3
codes144 = dictmerge(matching137, codes111) # EKG - reading 3
codes145 = dictmerge(matching137, codes112) # EKG - reading 3
codes146 = dictmerge(matching137, codes113) # EKG - reading 3
codes147 = dictmerge(matching137, codes114) # EKG - reading 3
codes148 = 0 # avg rate, do not turn to categorical
codes149 = 0 # avg PR interval, do not turn to categorical
codes150 = 0 # avg QRS interval, do not turn to categorical
codes151 = 0 # S wave depth in V1, do not turn to categorical
codes152 = 0 # R wave depth in V5, do not turn to categorical
codes153 = {1:'No lesion',2:'Indefinite',3:'Definite',0:'Unsatisfactory or missing X-Ray'} # Chest X-Ray - existence of lesion
codes154 = {
    1: 'None',
    2: 'Definite enlargement',
    3: 'Borderline enlargement',
    4: 'Definite LV',
    5: 'Borderline LV',
    6: 'Definite other chamber',
    7: 'Borderline other chamber',
    8: 'Definite combination of others',
    9: 'Borderline combination of others',
    0: 'Unsatisfactory or missing XRay'
}
codes155 = {
    1: 'None',
    2: 'Definite Calcification(aorta)',
    3: 'Borderline Calcification(aorta)',
    4: 'Definite shape of aorta',
    5: 'Borderline shape of aorta',
    6: 'Definite calcification and shape',
    7: 'Borderline calcification and shape',
    8: 'Definite inc. pulmonary vascularity',
    9: 'Borderline inc. pulmonary vascularity',
    10: 'Definite IPV and calcification or shape of aorta',
    11: 'Borderline IPV and calcification or shape of aorta',
    0: 'Unsatisfactory or missing XRay'
}
codes156 = codes153 # Chest X-Ray - existence of lesion reading 2
codes157 = codes154 # Chest X-Ray - existence of heart enlargement reading 2
codes158 = codes155 # Chest X-Ray - other cardiovascular abnormality reading 2
codes159 = codes153 # Chest X-Ray - existence of lesion reading 3
codes160 = codes154 # Chest X-Ray - existence of heart enlargement reading 3
codes161 = codes155 # Chest X-Ray - other cardiovascular abnormality reading 3
codes162 = {0:'Negative',1:'Definite 1',2:'Definite 2',3:'Definite 3',4:'Definite 4',5:'Definite 5',6:'Borderline 6',7:'Borderline 7',8:'Unsatisfactory or missing XRay'} # Chest X-Ray - existence of heart enlargement
codes163 = {
    1: '<80',
    2: '80-99',
    3: '100-119',
    4: '120-139',
    5: '140-159',
    6: '160-179',
    7: '180-199',
    8: '200-219',
    9: '220-239',
    10: '240-259',
    11: '260-279',
    12: '280-299',
    13: '300-319',
    14: '320-339',
    15: '340-359',
    16: '360-379',
    17: '380-399',
    18: '400-419',
    19: '420-439',
    20: '440-459',
    21: '460-479',
    22: '480-499',
    23: '500-519',
    24: '>520',
    99: 'unknown'
}
codes164 = codes73 # stroke diagnosis
codes165 = codes73 # cardiovascular condition diagnosis
codes166 = codes73 # heart condition diagnosis
codes167 = codes73 # high blood pressure diagnosis
codes168 = codes73 # varicose veins diagnosis
codes169 = codes73 # coronary heart disease diagnosis
codes170 = codes73 # rheumatic heart disease diagnosis
codes171 = codes73 # hypertensive heart disease diagnosis
codes172 = {1:'Definite - Aortic stenosis',2:'Definite - other',3:'Suspect - Definite enlargement on chest X-Ray',
            4:'Av block, right bundle block, right or left or both hypertrophy',5:'No HD'} # heart disease summary
codes173 = {
    1: 'Definite hypertension',
    2: 'Borderline hypertension',
    3: 'No hypertension',
    4: 'No entry'
}
codes174 = {
    1: 'Definite hypertension, LVH, and 35+',
    2: 'Definite hypertention, enlargement in XRay',
    3: 'Suspect hypertention, borderline HT and enlargement in XRay',
    4: 'Suspect hypertention, borderline HT and LVH and 35+',
    5: 'Borderline HT and definite enlargement in X-Ray',
    6: 'No HD',
    7: 'No hypertensive HD',
    8: 'Definite, under tratment for HT & LVH and 35+',
    9: 'Definite, under treatment for HT and enlargement in XRay'
}
codes175 = {
    1: 'Diastolic murmur',
    2: 'Grade 4 systolic murmur and Rheumatic HD',
    3: 'Grade3 systolic murmur and Rheumatic HD with hist of RHD or chorea',
}
codes176 = {
    1: 'Definite - Myocardial infarction',
    2: 'Definite - angina pectoris',
    3: 'Definite - L vent. ischemia',
    4: 'Suspect - Angina pectoris',
    5: 'No Coronary HD',
}
codes177 = {
    1: 'Congenital heart disease',
    2: 'Syphilitic heart disease',
    3: 'Traumatic heart disease',
    4: 'No heart disease',
}
codes178 = {
    1: 'Definite hypertensive, rheumatic, coronary, or other HD',
    2: 'Suspect HT coronary or other HD',
    3: 'No HD',
}
codes179 = {
    1: 'EKG abnormalities',
    2: 'Left axis deviation and corr. history',
    3: '1st degree AV block and corr. history',
    4: 'Venous engor., thrils, sig. sys. murmurs, splitting of second sound, other heart sound',
    5: 'No HD diagnosis/Code 0 in items 174,175,176,or 177',
} # Rhermatic heart disease summary
codes180 = {1:'Exam +, Physician +',2:'Exam +, Physician -',3:'Exam -, Physician +',4:'Exam -, Physician -',5:'Exam +, no Phys',6:'Exam -, no Phys',0:'other'} # heart disease diagnosis
codes181 = codes180 # coronary heart disease diagnosis
codes182 = codes180 # hypertensive heart disease diagnosis
codes183 = codes180 # hypertension diagnosis
codes184 = {1:'Positive',2:'Negative',3:'?',4:'N, A, Unknown, DK',0:'No physician exam'} # hypertension diagnosis by personal physician
codes185 = codes184 # peripheral vascular disease diagnosis
codes186 = codes184 # coronary heart disease diagnosis by personal physician
codes187 = codes184 # hypertensive heart disease diagnosis by personal physician
codes188 = codes184 # rheumatic heart disease diagnosis by personal physician
codes189 = codes184 # other heart disease diagnosis by personal physician


# names categoricals
for column in ndf.columns[2:]:
    code = eval('codes'+str(ndf.columns.get_loc(column)+1))
    #print('codes'+str(ndf.columns.get_loc(column)+1), column)
    #print(sorted(df[column].unique()))
    #if column == 'EKG-READING 2 1': # skips this one because it's messed up, drop later? - dropped now earlier in code, now data lines up
        #continue
    if code == 0: # skips all codes that = 0
        continue
    # elif len(code) != len(df[column].unique()): # checks that we have the same number of codes as unique values
    #    raise ValueError('Error in column: '+str(column)+'. Length mismatch in '+str('codes'+str(df.columns.get_loc(column)+1))
    #                    +'. There are '+str(len(code))+' codes and '+str(len(df[column].unique()))+' values. '
    #                    + '\nUnique values: '+str(sorted(df[column].unique()))+'.'
    #                    + '\nUnique codes: '+str(code)+'.')
        # tells you if not and what to fix, deprecated as I changed the lists to dicts
    # turns to categorical if all others pass
    # print(df[column].unique())
    try:
        ndf[column] = pd.Categorical(ndf[column], categories=eval(str(sorted(ndf[column].unique()))), ordered=True)
        ndf[column] = ndf[column].cat.rename_categories(code)
    except NameError:
        print('Error in column '+column+':'+str(NameError))
    #print(df[column].unique())
    #print(df[column].value_counts())
    #print('\n')
with open(r'readabledata.csv', 'w') as f:
    f.write(ndf.to_csv(index=False))
    
end_time = time.time()
elapsed_time = end_time - start_time
print('Data read.')
print(f"Elapsed time: {elapsed_time} seconds")

# item 126 might be a 'whether or not it was done', because it matches the 'not examined by this reader' number in 
# the 1 category.

# Drop diagnoses so they don't interfere with the model, these can be predicted in future runs with some tweaking
df = df.drop(["STROKE DIAGNOSIS",
    "CARDIOVASCULAR CONDITION DIAGNOSIS",
    #"HEART CONDITION DIAGNOSIS",
    "HIGH BLOOD PRESSURE DIAGNOSIS",
    "VARICOSE VEINS DIAGNOSIS",
    "CORONARY HEART DISEASE DIAGNOSIS 1",
    "RHEUMATIC HEART DISEASE DIAGNOSIS",
    "HYPERTENSIVE HEART DISEASE DIAGNOSIS"],axis=1)# Drop summaries so they don't interfere with the model, comment lines you want to annalyze
df = df.drop(["HEART DISEASE SUMMARY (IN DESCENDING ...","BLOOD PRESSURE SUMMARY","HYPERTENSIVE HEART DISEASE SUMMARY","RHEUMATIC HEART DISEASE SUMMARY","CORONARY HEART DISEASE SUMMARY","OTHER HEART DISEASE SUMMARY","HEART DISEASE SUMMARY","POSSIBLE HEART DISEASE","HEART DISEASE DIAGNOSIS","CORONARY HEART DISEASE DIAGNOSIS","HYPERTENSIVE HEART DISEASE DIAGNOSIS 2","HYPERTENSION DIAGNOSIS","HYPERTENSION DIAGNOSIS BY PERSONAL P ...","PERIPHERAL VASCULAR DISEASE  DIAGNOS ...","CORONARY HEART DISEASE DIAGNOSIS BY  ...","HYPERTENSIVE HEAR DISEASE DIAGNOSIS  ...","RHEUMATIC HEAR DISEASE DIAGNOSIS BY  ...","OTHER HEART DISEASE DIAGNOSIS BY PER ..."], axis=1)

# import libraries from NN_MOF.py
import numpy as np
import random as rn
#import tensorflow as tf

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, classification_report, confusion_matrix, auc, roc_auc_score
from sklearn import metrics

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

# Neural Network
from sklearn.metrics import make_scorer, accuracy_score

nn = MLPClassifier(max_iter=200)

# Hyperparameter search for MLP.

'''nn_param_search = {
    'hidden_layer_sizes': [(10,30,10)],
    'activation': ['tanh'],
    'solver': ['sgd'],
    'alpha': [0.0001], # L2 reguarization
    'learning_rate': ['adaptive'],
    }'''


nn_param_search = {
    'hidden_layer_sizes': [(60,)],
    'activation': ['relu'], # 'tanh', 'logistic', 'identity'
    'solver': ['adam'], # 'sgd', 
    'alpha': [0.1], # 0.0001, 0.05, 
    'learning_rate': ['constant'], #'adaptive','invscaling', 
    }

def delayed_accuracy(y_true, y_pred): # This is to delay the accuracy score
    time.sleep(8)  # delay so my CPU doesn't cook itself
    return accuracy_score(y_true, y_pred) # return the same thing as before, just delayed

delayed_scorer = make_scorer(delayed_accuracy)

# change verbose to any interger if you want to print out the message
nn_grid_search=GridSearchCV(nn, nn_param_search, n_jobs=-1, cv=6, verbose=3, scoring=delayed_scorer) 

# n_jobs = -1 uses all processors, or you can specify the number of processors to use

# SMOTE in training data
sm = SMOTE(sampling_strategy=1.0,random_state=seed)
x_train_s, y_train_s = sm.fit_resample(X_train, y_train)

# Print before and after SMOTE
unique_values = df[column_to_predict].unique()
print("Unique values in the column to predict:"+str(unique_values))

print("Before OverSampling, counts of label 1:'Heart Disease': {} ".format(sum(y_train == 1)))
print("Before OverSampling, counts of label 2:'No Heart Disease': {} \n".format(sum(y_train == 2)))
    
print("After OverSampling, counts of label 'Heart Disease': {}".format(sum(y_train_s == 1)))
print("After OverSampling, counts of label 'No Heart Disease': {} \n".format(sum(y_train_s == 2))) 

print('Before OverSampling, the shape of train_X: {}'.format(X_train.shape))
print('Before OverSampling, the shape of train_y: {} \n'.format(y_train.shape))

print('After OverSampling, the shape of train_X: {}'.format(x_train_s.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_s.shape))

# Fit the model using training data after SMOTE
print('Grid Searching...')
nn_grid_search.fit(x_train_s, y_train_s)
nn_best_grid = nn_grid_search.best_estimator_
nn_best_params = nn_grid_search.best_params_
print('Grid Search Finished.')

import pickle
#import datetime
## Save the model
#current_datetime = datetime.datetime.now()
#currtime = current_datetime.strftime(r"%Y-%m-%d")
#with open('nn_best_grid_test'+currtime+'.pkl', 'wb') as f:
#    pickle.dump(nn_best_grid, f)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

# %%
yscore_raw = nn_grid_search.predict_proba(X_test)
yscore = [s[1] for s in yscore_raw]
# print(y_test,yscore)
fpr, tpr, thresh = roc_curve(y_test, yscore, pos_label=2)

# Calculate the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Neural Network (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Plot the random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Neural Network')
plt.legend(loc="lower right")
plt.savefig('Neural Network_MOF.png')

# print the best hyperparameters
print(f'The best parameters are {nn_best_params}')

#The best parameters are {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': (60,), 'learning_rate': 'constant', 'solver': 'adam'}

# Predict the labels of the test set: y_pred
y_hat = nn_best_grid.predict(X_test)

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
result = permutation_importance(nn_best_grid, x_train_s, y_train_s, n_repeats=6, random_state=seed)

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
