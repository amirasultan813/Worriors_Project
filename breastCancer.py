import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest, skew
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, cross_val_predict,  KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from scipy.stats import zscore
import pickle
data=pd.read_csv('METABRIC_RNA_Mutation.csv')
pd.set_option("display.max_column",None)
data.head()
data.shape
total_empty=data.isnull().sum().sort_values(ascending=False)
percent_empty=(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_values=pd.concat([total_empty,percent_empty],axis=1,keys=['Total','Percent'])
missing_values.head(13)
clinical_attr_dropped=data.columns[31:]
data2=data.drop(clinical_attr_dropped,axis=1)
data2.head()
data2.info()
plt.figure(figsize=(15,5))
sns.boxplot(x='lymph_nodes_examined_positive', y='overall_survival_months', hue='overall_survival', data=data2) #ax=axes[0, 0]
plt.xlabel("Lymph nodes examined positive", fontsize=18)
plt.xticks(fontsize=12)
plt.ylabel("Overall survival (months)", fontsize=18)
plt.yticks(fontsize=12)
plt.title("Overall survival vs. positive lymph nodes", fontsize=20)
plt.show()
#Group by overall survival to find the survival months mean and std. 
surviv_months = data2.groupby(by='overall_survival')
surviv_monthsMean = surviv_months.mean()['overall_survival_months']
surviv_monthsStd = surviv_months.std()['overall_survival_months']
print (surviv_monthsMean)
print(surviv_monthsStd)
#treatment list 
treat_list = ["type_of_breast_surgery", "chemotherapy", "hormone_therapy", "radio_therapy"]
#correlations between the treatments 
data2_treat = data2[treat_list]
data2_treat["type_of_breast_surgery"] = data2_treat["type_of_breast_surgery"].apply(lambda x: 1 if "MASTECTOMY" in str(x) else 0)
data2_treat.head()
data2_treat.rename(columns={"chemotherapy": "Chemotherapy", "hormone_therapy": "Hormone Therapy", "radio_therapy": "Radio Therapy"}, inplace=True)
data2_treatCorr = data2_treat.corr() 
data2_treat.head()
#Display heatmap 
sns.heatmap(data2_treatCorr, cmap = "YlGnBu", annot=True, fmt=".2f")
plt.show()
# overall survival % in 100%.
SurgeryCounts = (data2_treat["type_of_breast_surgery"].value_counts()[1]/data2_treat["type_of_breast_surgery"].value_counts().sum())*100
chemoCounts = (data2_treat["Chemotherapy"].value_counts()[1]/data2_treat["Chemotherapy"].value_counts().sum())*100
hormoneCounts = (data2_treat["Hormone Therapy"].value_counts()[1]/data2_treat["Hormone Therapy"].value_counts().sum())*100
radioCounts = (data2_treat["Radio Therapy"].value_counts()[1]/data2_treat["Radio Therapy"].value_counts().sum())*100
#Dictionary of the survival % for each treatment 
treatmentSurvivalDict = {'type_of_breast_surgery': SurgeryCounts, 'Chemotherapy': chemoCounts, 'Hormone Therapy':hormoneCounts, 'Radio Therapy': radioCounts}
#Plot of each treatment 
plt.figure(figsize=(7.5,5))
plt.bar(list(treatmentSurvivalDict.keys()), list(treatmentSurvivalDict.values()))
plt.xlabel("Treatment", fontsize=18)
plt.ylabel("Survival %", fontsize=18)
plt.xticks(size=12)
plt.yticks(size=12)
plt.tight_layout()
fig = plt.figure(figsize = (20, 25))
j = 0
number_clinical_col= ['age_at_diagnosis', 'lymph_nodes_examined_positive','mutation_count','nottingham_prognostic_index', 'overall_survival_months', 'tumor_size' ]
for i in data2[number_clinical_col].columns:
    plt.subplot(6, 3, j+1)
    j += 1
    sns.distplot(data2[i][data2['overall_survival']==1], color='g', label = 'Survived')
    sns.distplot(data2[i][data2['overall_survival']==0], color='r', label = 'Died')
    plt.legend(loc='best')
fig.suptitle('Clinical Data Analysis')
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()
color= "Spectral"
two_colors = [ sns.color_palette(color)[1], sns.color_palette(color)[5]]
three_colors = [ sns.color_palette(color)[5],sns.color_palette(color)[1], sns.color_palette(color)[0]]
died = data2[data2['overall_survival']==0]
survived = data2[data2['overall_survival']==1]
alive = data2[data2['death_from_cancer']=='Living']
died_cancer = data2[data2['death_from_cancer']=='Died of Disease']
died_not_cancer = data2[data2['death_from_cancer']=='Died of Other Causes']
fig, ax = plt.subplots(ncols=2, figsize=(15,3), sharey=True)
sns.boxplot(x='overall_survival_months', y='overall_survival', orient='h', data=data2, ax=ax[0], palette = two_colors, saturation=0.90)
sns.boxplot(x='age_at_diagnosis', y='overall_survival', orient='h', data=data2, ax=ax[1], palette = two_colors, saturation=0.90)
fig.suptitle('The Distribution of Survival time in months and age with Target Attribute', fontsize = 18)
ax[0].set_xlabel('Total survival time in Months')
ax[0].set_ylabel('survival')
ax[1].set_xlabel('Age at diagnosis')
ax[1].set_ylabel('')

plt.show()
fig, ax = plt.subplots(ncols=2, figsize=(15,3), sharey=True)
sns.boxplot(x='overall_survival_months', y='death_from_cancer', orient='h', data=data2, ax=ax[0], palette = three_colors, saturation=0.90)
sns.boxplot(x='age_at_diagnosis', y='death_from_cancer', orient='h', data=data2, ax=ax[1], palette = three_colors, saturation=0.90)
fig.suptitle('The Distribution of Survival and Recurrence with Target Attribute', fontsize = 18)
ax[0].set_xlabel('Total survival time in Months')
ax[0].set_ylabel('survival')
ax[1].set_xlabel('Age at diagnosis')
ax[1].set_ylabel('')

plt.show()
fig, ax = plt.subplots(ncols=2, figsize=(15,3), sharey=True)
fig.suptitle('The Distribution of Survival and Recurrence with Target Attribute', fontsize = 18)
ax[0].hist(died['overall_survival_months'], alpha=0.9, color=sns.color_palette(color)[0], label='Died')
ax[0].hist(survived['overall_survival_months'], alpha=0.9, color=sns.color_palette(color)[5], label='Survived')
ax[0].legend()
ax[1].hist(alive['overall_survival_months'], alpha=0.9, color=sns.color_palette(color)[5], label='Survived')
ax[1].hist(died_cancer['overall_survival_months'], alpha=0.9, color=sns.color_palette(color)[0], label='Died from cancer')
ax[1].hist(died_not_cancer['overall_survival_months'], alpha=0.9, color=sns.color_palette(color)[1], label='Died not from cancer')
ax[1].legend()
ax[0].set_xlabel('Total survival time in Months')
ax[0].set_ylabel('Number of patients')
ax[1].set_xlabel('Total survival time in Months')
ax[1].set_ylabel('')

plt.show()
fig, ax = plt.subplots(ncols=3, figsize=(15,3))
fig.suptitle('The Distribution of treatment and survival', fontsize = 18)
sns.countplot(died['chemotherapy'], color=sns.color_palette(color)[0], label='Died', ax=ax[0], saturation=0.90)
sns.countplot(x= survived['chemotherapy'] , color=sns.color_palette(color)[5], label='Survived', ax=ax[0], saturation=0.90)
#ax[0].legend()
ax[0].set(xticklabels=['No','Yes'])
sns.countplot(died['hormone_therapy'], color=sns.color_palette(color)[0], label='Died', ax=ax[1], saturation=0.90)
sns.countplot(x=  survived['hormone_therapy'], color=sns.color_palette(color)[5], label='Survived', ax=ax[1], saturation=0.90)
ax[1].legend()
ax[1].set(xticklabels=['No','Yes'])
sns.countplot(died['radio_therapy'], color=sns.color_palette(color)[0], label='Died', ax=ax[2], saturation=0.90)
sns.countplot(x=  survived['radio_therapy'], color=sns.color_palette(color)[5], label='Survived', ax=ax[2], saturation=0.90)
#ax[2].legend()
ax[2].set(xticklabels=['No','Yes'])
ax[0].set_xlabel('Chemotherapy')
ax[0].set_ylabel('Number of patients')
ax[1].set_xlabel('Hormonal therapy')
ax[1].set_ylabel('')
ax[2].set_xlabel('Radio therapy')
ax[2].set_ylabel('')

plt.show()
fig, ax = plt.subplots( figsize=(10,5))
fig.suptitle('The Distribution histopathological class and survival', fontsize = 18)
sns.countplot(x='neoplasm_histologic_grade', hue='overall_survival' ,data = data2, palette=two_colors , ax=ax, saturation=0.90)
ax.legend([ 'Died', 'Survived'])
ax.set_xlabel('histopathological class')
ax.set_ylabel('Number of patients')

plt.show()
num_clinical_columns= ['age_at_diagnosis', 'lymph_nodes_examined_positive','mutation_count','nottingham_prognostic_index', 'overall_survival_months', 'tumor_size' ]
cat_clinical_columns = ['chemotherapy', 'cohort', 'neoplasm_histologic_grade','hormone_therapy', 'overall_survival', 'radio_therapy', 'tumor_stage' ]
data2[num_clinical_columns].describe(). T
cat_clinical_columns.extend(data2.select_dtypes(include=['object']).columns.tolist())
data2[cat_clinical_columns].astype('category').describe().T
no_treatment = data2[(data2['chemotherapy']==0) & (data2['hormone_therapy']==0) & (data2['radio_therapy']==0)]
print("Number of patients who had no treatment: " , no_treatment.shape[0])
print("Proportion of survival in this group: " , ("%.3f" %np.mean(no_treatment["overall_survival"])))
print("Baseline Proportion of survival in all groups: ", ("%.3f" %np.mean(data2["overall_survival"])))
print("Mean age: " + "%.3f" %np.mean(data2['age_at_diagnosis']))
print("Mean tumour diameter: " + "%.3f" %np.mean(data2['tumor_size']))
print("Probability of survival: "+ "%.3f" %(data2["overall_survival"].value_counts()/data2["overall_survival"].count()).iloc[1])
# drop mutations
genetic_features_to_drop = data.columns[520:]
genetic_data = data.drop(genetic_features_to_drop, axis=1)
# droping clinical data
genetic_features_to_drop = genetic_data.columns[4:35]
genetic_data = genetic_data.drop(genetic_features_to_drop, axis=1)
genetic_data = genetic_data.drop(['age_at_diagnosis','type_of_breast_surgery', 'cancer_type'], axis=1)
genetic_data = genetic_data.iloc [:,:-174]
genetic_data['overall_survival']= data['overall_survival']

genetic_data.head()
genetic_features_to_drop
#Find Maximum values and std in each column, std is always 1 because the datapoints are z-scores
max_values = genetic_data.max()
std = genetic_data.std(axis = 0, skipna = True)
max_data = pd.concat([max_values, std], axis = 1, keys = ['max_values', 'std'])
max_data.sort_values(by='max_values', ascending = False).head()
#Finding minimum values and std in each column, std is always 1 because the datapoints are z-scores
min_values = genetic_data.min()
std = genetic_data.std(axis = 0, skipna = True)
min_data = pd.concat([min_values, std], axis = 1, keys = ['min_values', 'std'])
min_data.sort_values(by='min_values', ascending = True).head()
fig = plt.figure(figsize = (20, 25))
j = 0
gene_list = ['rab25', 'eif5a2', 'pik3ca', 'kit', 'fgf1', 'myc', 'egfr', 'notch3', 'kras', 'akt1', 'erbb2', 'pik3r1', 'ccne1', 'akt2', 'aurka']
for i in genetic_data.drop(['patient_id'], axis=1).loc[:,gene_list].columns:
    plt.subplot(6, 4, j+1)
    j += 1
    sns.distplot(genetic_data[i][genetic_data['overall_survival']==0], color='g', label = 'survived')
    sns.distplot(genetic_data[i][genetic_data['overall_survival']==1], color='r', label = 'died')
    plt.legend(loc='best')
fig.suptitle('Clinical Data Analysis')
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()
print('Maximum value possible in genetic data:', genetic_data.drop(['patient_id','overall_survival'], axis = 1).max().max())
print('Minimum value possible in genetic data:', genetic_data.drop(['patient_id','overall_survival'], axis = 1).min().min())
#Finding number of outliers in each column
Q1 = genetic_data.quantile(0.25)
Q3 = genetic_data.quantile(0.75)
IQR = Q3 - Q1
((genetic_data < (Q1 - 1.5 * IQR)) | (genetic_data > (Q3 + 1.5 * IQR))).sum().sort_values(ascending = False).head(10)
# droping clinical and genetic data
mutation_features_to_drop = data.columns[4:520]
mutation_data = data.drop(mutation_features_to_drop, axis=1)
mutation_data = mutation_data.drop(['age_at_diagnosis','type_of_breast_surgery', 'cancer_type'], axis=1)

# if there is a mutation=1, no-mutation=0
for column in mutation_data.columns[1:]:
    mutation_data[column]=pd.to_numeric(mutation_data[column], errors='coerce').fillna(1).astype(int)
mutation_data.insert(loc=1 , column='overall_survival', value=data['overall_survival'])

mutation_data.head()
BOLD = '\033[1m'
END = '\033[0m'
# using a stratfied k fold because we need the distribution of the to classes in all of the folds to be the same.
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print('Baseline accuracy:' )
print(data["overall_survival"].value_counts()/data["overall_survival"].count())
categorical_columns = data2.select_dtypes(include=['object']).columns.tolist()
unwanted_columns = ['patient_id','death_from_cancer' ]
categorical_columns = [ele for ele in categorical_columns if ele not in unwanted_columns] 
# Getting dummies for all categorical columns
dummies_clinical_data = pd.get_dummies(data2.drop('patient_id',axis=1 ), columns= categorical_columns, dummy_na=False)
dummies_clinical_data.dropna(inplace = True)
dummies_clinical_data.head(3)
dummies_clinical_data.shape
dummies_clinical_data=dummies_clinical_data.drop(['cancer_type_Breast Sarcoma', 'cancer_type_detailed_Breast Invasive Ductal Carcinoma','cancer_type_Breast Cancer','cancer_type_detailed_Breast','cancer_type_detailed_Breast Invasive Lobular Carcinoma','cancer_type_detailed_Breast Invasive Mixed Mucinous Carcinoma','cancer_type_detailed_Breast Mixed Ductal and Lobular Carcinoma','cancer_type_detailed_Metaplastic Breast Cancer','3-gene_classifier_subtype_ER+/HER2- High Prolif','3-gene_classifier_subtype_ER+/HER2- Low Prolif','3-gene_classifier_subtype_ER-/HER2-',
'3-gene_classifier_subtype_HER2+'], axis=1)
dummies_clinical_data.head(3)
tumor_unwanted=dummies_clinical_data.drop(['oncotree_code_BREAST','oncotree_code_IDC','oncotree_code_ILC','oncotree_code_IMMC','oncotree_code_MBC','oncotree_code_MDLC','cohort','neoplasm_histologic_grade','nottingham_prognostic_index','overall_survival_months','death_from_cancer','type_of_breast_surgery_BREAST CONSERVING','type_of_breast_surgery_MASTECTOMY','cellularity_Moderate','pam50_+_claudin-low_subtype_Basal','pam50_+_claudin-low_subtype_Her2','pam50_+_claudin-low_subtype_LumA','pam50_+_claudin-low_subtype_LumB','pam50_+_claudin-low_subtype_NC','pam50_+_claudin-low_subtype_Normal','pam50_+_claudin-low_subtype_claudin-low','er_status_measured_by_ihc_Negative','er_status_measured_by_ihc_Positve','er_status_Negative','er_status_Positive','her2_status_measured_by_snp6_GAIN','her2_status_measured_by_snp6_LOSS','her2_status_measured_by_snp6_NEUTRAL','her2_status_measured_by_snp6_UNDEF','her2_status_Negative','her2_status_Positive','tumor_other_histologic_subtype_Ductal/NST','tumor_other_histologic_subtype_Lobular','tumor_other_histologic_subtype_Medullary','tumor_other_histologic_subtype_Metaplastic','tumor_other_histologic_subtype_Mixed','tumor_other_histologic_subtype_Mucinous','tumor_other_histologic_subtype_Other','tumor_other_histologic_subtype_Tubular/ cribriform','inferred_menopausal_state_Post','inferred_menopausal_state_Pre','integrative_cluster_1','integrative_cluster_10','integrative_cluster_2','integrative_cluster_3','integrative_cluster_4ER+','integrative_cluster_5','integrative_cluster_6','integrative_cluster_7','integrative_cluster_8','integrative_cluster_9','pr_status_Negative','pr_status_Positive','integrative_cluster_4ER-'], axis=1)
tumor_unwanted.head(2)
x=tumor_unwanted.iloc[:,:8]
y=tumor_unwanted.iloc[:,8]
y.head(2)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)
from sklearn.linear_model import LinearRegression
model_tumor=LinearRegression()
model_tumor.fit(x_train,y_train)
y_predict=model_tumor.predict(x_test)
y_predict
from sklearn.metrics import mean_squared_error
errors = mean_squared_error(y_test, y_predict)
print(errors)
tumor_unwanted.head(1)
y2=tumor_unwanted['mutation_count']
mutation_data=tumor_unwanted.drop(['mutation_count'],axis=1)
mutation_data.head(1)
x2=mutation_data.iloc[:,:]
from sklearn.model_selection import train_test_split
x_train2,x_test2,y_train2,y_test2=train_test_split(x2,y2,test_size=.2)
from sklearn.linear_model import LinearRegression
model_mutation=LinearRegression()
model_mutation.fit(x_train2,y_train2)
y_predict2=model_mutation.predict(x_test2)
y_predict2
from sklearn.metrics import mean_squared_error
error2 = mean_squared_error(y_test2, y_predict2)
print(error2)
filename = 'model.pkl'
pickle.dump(model_mutation, open(filename, 'wb'))

