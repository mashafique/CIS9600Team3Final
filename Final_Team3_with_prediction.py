#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 21:46:14 2021

@author: abelroman
"""

import seaborn as sns
import pandas as pd
import numpy as np
import nltk
import warnings
import textstat
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from review_clean1 import review_clean
from dmba import regressionSummary
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from document1 import document_features
from dmba import plotDecisionTree, classificationSummary, regressionSummary
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
import pandas_bokeh
from bokeh.resources import INLINE
import bokeh.io
bokeh.io.output_notebook(INLINE) 
from cascore import ca_score_model
from nb import get_NBmetrics


price=pd.read_csv('/Users/abelroman/Desktop/python_team3/prices.csv')
price=price.rename(columns = {'Spending ':'Spending'})
price['Brand Name'].replace(to_replace=r'\*$',value="", inplace=True, regex=True)
price['Brand Name'].replace(to_replace=r'-',value=" / ", inplace=True, regex=True)
price = price.groupby(['Brand Name'])['Spending'].mean().reset_index()

price.columns = [s.strip().replace(' ', '') for s in price.columns]

# Perform cleaning, normalization, and tokenization of data

price['drugName_list2']=price['BrandName'].str.lower()

# Tokenize the drug names
price['drugName_list2']=price['drugName_list2'].apply(lambda BrandName: nltk.word_tokenize(BrandName))

# Create list of drug names and remove 20,000 most common words as per Google
drugs = price['drugName_list2'].tolist()
words_common = [line.strip() for line in open('/Users/abelroman/Desktop/python_team3/20k.txt', 'r')]
drugs = [token for token in drugs if token not in words_common]
drugs_names2=[name for sublist in drugs for name in sublist]

# Perform cleaning on the drug names
price['drugName_list2'].replace(to_replace=r"&#039;",value="'", inplace=True, regex=True)
price['drugName_list2'].replace(to_replace=r'^\"',value="", inplace=True, regex=True)
price['drugName_list2'].replace(to_replace=r'\"$',value="", inplace=True, regex=True)
price['drugName_list2'].replace(to_replace=r'http\S+',value="", inplace=True, regex=True)
price['drugName_list2'].replace(to_replace=r'http',value="", inplace=True, regex=True)
price['drugName_list2'].replace(to_replace=r'(\d)',value="", inplace=True, regex=True)
price['drugName_list2'].replace(to_replace=r'@\S+',value="", inplace=True, regex=True)
price['drugName_list2'].replace(to_replace=r'[^A-Za-z0-9(),!?@\'\`\"\_\n]',value="", inplace=True, regex=True)
price['drugName_list2'].replace(to_replace=r'@',value="at", inplace=True, regex=True)

price["drugName_list3"] = [" ".join(w) for w in price["drugName_list2"]]

print(price)

# combining test and training dataset
warnings.filterwarnings('ignore')
Train = open("/Users/abelroman/Desktop/python_team3/drugsComTrain_raw.csv")
Test = open("/Users/abelroman/Desktop/python_team3/drugsComTest_raw.csv")
Train1 = pd.read_csv(Train)
Test1 = pd.read_csv(Test)
Main = pd.concat([Train1, Test1], ignore_index=True)
print(Main.columns)
print(len(Main))
Main = Main[~Main["condition"].str.contains("users found this comment helpful", na=False)]
Main = Main.drop('uniqueID', 1)
Main['sentiment']= np.where(Main['rating']>=7,'Positive','Negative')

# Show the coverage for each sentiment by its rating
sns.boxplot(x='sentiment', y="rating", data=Main).set_title('Rating vs Sentiment')
plt.show()

# analysis of the data we are using: 
Main['rating'] = Main[Main['rating'] > 0]['rating']
Main['usefulCount'] = Main[Main['usefulCount'] > 10]['usefulCount']
Main = Main[Main['rating'].notna()]
Main = Main[Main['usefulCount'].notna()]
print(Main.corr())
print(len(Main))

print('\n')
print('Mean useful count of each condition')
rev = Main.groupby(['condition'])['usefulCount'].mean().sort_values(ascending=False)
print(rev)

#PCA analysis of full dataset
print('\n')
print('PCA Analysis')
print('pcsSummary')

pcs = PCA(n_components=2)
pcs.fit(Main[['rating', 'usefulCount']])
pcsSummary = pd.DataFrame({'Standard deviation': np.sqrt(pcs.explained_variance_), 'Proportion of variance': pcs.explained_variance_ratio_, 'Cumulative proportion': np.cumsum(pcs.explained_variance_)})
pcsSummary = pcsSummary.transpose()
pcsSummary.Colummns = ['PC1', 'PC2']
print(pcsSummary.round(4))

print('\n')
print('Components')
pcsComponents_df = pd.DataFrame(pcs.components_.transpose(), columns=['PC1', 'PCS'],index=['rating','usefulCount'])

print(pcsComponents_df)

print('\n')
print('Scores')
scores = pd.DataFrame(pcs.transform(Main[['rating','usefulCount']]),
                      columns=['PC1', 'PC2'])

drugname = []
# Patient condition example (Cough) it is case sensitive!
symp = str(input("What is the patients condition?:"))
symptom=(Main[Main['condition'] == symp])

print(symptom)

symptom['drugName_list']=symptom['drugName'].str.lower()

# Tokenize drug names
symptom['drugName_list']=symptom['drugName_list'].apply(lambda drugName: nltk.word_tokenize(drugName))

# Create list of drug names and remove 20,000 most common words as per Google
drugs = symptom["drugName_list"].tolist()
words_common = [line.strip() for line in open('/Users/abelroman/Desktop/python_team3/20k.txt', 'r')]
drugs = [token for token in drugs if token not in words_common]
drugs_names=[name for sublist in drugs for name in sublist]

symptom['drugName_list'].replace(to_replace=r"&#039;",value="'", inplace=True, regex=True)
symptom['drugName_list'].replace(to_replace=r'^\"',value="", inplace=True, regex=True)
symptom['drugName_list'].replace(to_replace=r'\"$',value="", inplace=True, regex=True)
symptom['drugName_list'].replace(to_replace=r'http\S+',value="", inplace=True, regex=True)
symptom['drugName_list'].replace(to_replace=r'http',value="", inplace=True, regex=True)
symptom['drugName_list'].replace(to_replace=r'(\d)',value="", inplace=True, regex=True)
symptom['drugName_list'].replace(to_replace=r'@\S+',value="", inplace=True, regex=True)
symptom['drugName_list'].replace(to_replace=r'[^A-Za-z0-9(),!?@\'\`\"\_\n]',value="", inplace=True, regex=True)
symptom['drugName_list'].replace(to_replace=r'@',value="at", inplace=True, regex=True)


symptom["drugName_list3"] = [" ".join(w) for w in symptom["drugName_list"]]

symptom = symptom.join(price.set_index('drugName_list3'), on='drugName_list3')

symptom['cleaned_review']=symptom['review'].str.lower()
symptom['condition']=symptom['condition'].str.lower()
symptom['cleaned_review'].replace(to_replace=r"&#039;",value="'", inplace=True, regex=True)
symptom['cleaned_review'].replace(to_replace=r'^\"',value="", inplace=True, regex=True)

symptom['cleaned_review'].replace(to_replace=r'\"$',value="", inplace=True, regex=True)
symptom['cleaned_review_raw_vader'] = symptom['cleaned_review']
symptom['cleaned_review'] = symptom['cleaned_review'].apply(lambda review: nltk.word_tokenize(review))

stop_words = nltk.corpus.stopwords.words('english')

words_commonK = pd.read_csv('/Users/abelroman/Desktop/python_team3/Identifying Corpus Specific Stop Words.csv', index_col=False)

stop_words = stop_words + list(words_commonK["Stop_Words"])

symptom['cleaned_review'] = symptom['cleaned_review'].apply(lambda review: [word for word in review if word not in (stop_words)])
symptom['cleaned_review'] = symptom['cleaned_review'].apply(lambda review: [word for word in review if word.isalpha()])

wnl = nltk.WordNetLemmatizer()
symptom['cleaned_review'] = symptom['cleaned_review'].apply(lambda review: [wnl.lemmatize(word) for word in review])

symptom['cleaned_review_raw_cv'] = symptom['cleaned_review'].apply(lambda review: ' '.join(review))

review_col = symptom["review"]

Grade = [textstat.flesch_kincaid_grade(i) for i in review_col]
symptom["Grade"] = Grade


#History medication takes out any medication already prescribed so the code wont give it as an option.
Med = str(input("History of medication for patient" + " " + symp + "? " + "(Y/N):"))
if Med == 'Y':
   DrgNme = str(input("Please input Drugs prescribed (once finished type exit): "))
elif Med == 'N':
    DrgNme = 'null'

#will give list of top ten drugs sorted by usefulcount
drugname.append(DrgNme)
Main_symp=symptom[~symptom['drugName'].isin(drugname)]
Main_symp.sort_values(by='usefulCount', ascending=False)

Drug = Main_symp.drugName.value_counts().sort_values(ascending=False)


print(Drug[:10])

#Will give chart of the top 10 drugs that could work
Drug[:10].plot(kind='bar')
plt.title('Top 10 Drugs used for' + " " + symp)
plt.xlabel('drugName')
plt.ylabel('Count');
plt.show()


#limit the review to atleast 5
test1 = Main_symp['drugName'].value_counts().to_dict()
Main_symp['drugName_count'] = Main_symp['drugName'].map(test1)
Main_symp = Main_symp[Main_symp['drugName_count'] > 5]


## correlation between usefulness count and rating?
plt.scatter(Main_symp.rating, Main_symp.usefulCount, c=Main_symp.rating.values, cmap='tab10')
plt.title('Useful Count vs Rating')
plt.xlabel('Rating')
plt.ylabel('Useful Count')
plt.xticks([i for i in range(1, 11)]);

plt.show()

# mean rating for each drug
print("Mean rating for each drug")
drug_cond = Main_symp.groupby("drugName").mean()[['rating']]
drug_cond1 = drug_cond.sort_values(by='rating', ascending=False)

print(drug_cond1.head(30))


print("Mean price for each drug price (per pill)")
Price_drg = Main_symp.groupby("drugName").mean()[['Spending']]
print(Price_drg)

Price_drg = Price_drg['Spending'].dropna()

plt.style.use('fivethirtyeight') 
Price_drg.plot(kind="bar")
plt.title("Drug Prices (Per Pill)")
plt.xlabel("Drug Name")
plt.ylabel("Price ($)")

plt.show()


Main_symp['review_clean'] = review_clean(Main_symp['review'])

#From the top 10 drug list for condition you can input a drug from that list to get more info
while True:
    choice = str(input("Which drug would you like more information?: (Type exit when done)"))
    if choice == "exit":
        break;
    drg_list=(Main_symp[Main_symp['drugName'] == choice])
    drg_list.sort_values(by='rating', ascending=False)
    print("Review Breakdow")
    print(drg_list["rating"].describe())
    print("Useful Count Breakdown")
    print(drg_list["usefulCount"].describe())
    drg_list.loc[(drg_list['rating'] >= 7), 'Review_Sentiment'] = 1
    drg_list.loc[(drg_list['rating'] < 7), 'Review_Sentiment'] = 0
    cnt = drg_list['Review_Sentiment'].value_counts()
    colors1 = ['Positive', 'Negative']
    labels = list(colors1)
    explode1 = (0, 0.1)
    
    plt.pie(cnt, explode=explode1, labels=colors1, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title('Positive & Negative Reviews for' + " " + choice, fontsize = 15)
    plt.show()
       
    
    print('\n')
    print("Top Reviews")
    for i in drg_list.review_clean.iloc[:3]:
        print(i, '\n')
        
    print('\n')
    print("Negative/less popular reviews")
    for i in drg_list.review_clean.iloc[-3:]:
        print(i, '\n')
        
    # popularity of drug chosen over the years
    drg_list['date'] = pd.to_datetime(drg_list['date'], errors = 'coerce')
    drg_list['Year'] = drg_list['date'].dt.year
    drg_list['month'] = drg_list['date'].dt.month
    drg_list['day'] = drg_list['date'].dt.day
    plt.rcParams['figure.figsize'] = (9, 9)
    sns.countplot(drg_list['Year'], palette ='colorblind')
    plt.title('The No. of Reviews each year for' + " " + choice, fontsize = 15)
    plt.xlabel('Year', fontsize = 10)
    plt.ylabel('Count of Reviews', fontsize = 10)
    plt.show()
  
    
review_words = Main_symp['cleaned_review'].values.tolist()
review_words = [word for review in review_words for word in review]
review_words_fdist = nltk.FreqDist(review_words)

    
Main_symp['not_dissatisfied']=Main_symp['rating']>=4
Main_symp['satisfied']=Main_symp['rating']>=7

satisfied_review_words = Main_symp[Main_symp['satisfied']==True]['cleaned_review'].values.tolist()

dissatisfied_review_words = Main_symp[Main_symp['not_dissatisfied']==False]['cleaned_review'].values.tolist()

satisfied_review_words = [word for review in satisfied_review_words for word in review]
dissatisfied_review_words = [word for review in dissatisfied_review_words for word in review]

satisfied_review_fdist = nltk.FreqDist(satisfied_review_words)
dissatisfied_review_fdist = nltk.FreqDist(dissatisfied_review_words)

satisfied_review_common_words=[w[0] for w in satisfied_review_fdist.most_common(2000)]

dissatisfied_review_common_words=[w[0] for w in dissatisfied_review_fdist.most_common(2000)]

d1 = {"Satisfied": satisfied_review_common_words, "Dissatisfied": dissatisfied_review_common_words}

Review_Common_words_df = pd.DataFrame(data=d1)


review_common_words = set(satisfied_review_common_words + dissatisfied_review_common_words)


total_review_list = Main_symp['review'].values.tolist()
satisfied_status_list = Main_symp['satisfied'].values.tolist()

total_review_list = [nltk.word_tokenize(review) for review in total_review_list[:5000]]
satisfied_status_list = satisfied_status_list[:5000]

documents = list(zip(total_review_list, satisfied_status_list))

featuresets = [(document_features(d, review_common_words), c) for (d,c) in documents]

featuresets_df = pd.DataFrame(data=featuresets,columns=['Feature', 'Satisfied'])


d = featuresets_df["Feature"][0]
key_list = list(d.keys())

size = int(.9*len(featuresets))
train_set, test_set = featuresets[:size], featuresets[size:]

test_set_X = featuresets_df["Feature"][size:]
test_set_y = featuresets_df["Satisfied"][size:]


classifier = nltk.NaiveBayesClassifier.train(train_set)

print(classifier.show_most_informative_features(15))

# Identify the predicted value of the classifier
test_y_est = [classifier.classify(test_val) for test_val in test_set_X]

# Identify the true value of the variable
test_y = list(test_set_y)

accuracy_NB, precision_NB, recall_NB, f1_NB = get_NBmetrics(test_y_est,test_y)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_NB, precision_NB, recall_NB, f1_NB))


sa = SentimentIntensityAnalyzer()
Main_symp['vader_polarity_scores']=Main_symp['cleaned_review_raw_vader'].apply(lambda review: sa.polarity_scores(review))

Main_symp['vader_compound_polarity_score']=Main_symp['vader_polarity_scores'].apply(lambda review: review['compound'])

vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), min_df=10, max_df=25000)
cv_review_list = Main_symp['cleaned_review_raw_cv'].values.tolist()
cv_review_vector = vectorizer.fit_transform(cv_review_list)

lda_model = LatentDirichletAllocation(n_components = 5, random_state = 1, n_jobs = -1)
lda_output = lda_model.fit_transform(cv_review_vector)

topic_names = ["topic_" + str(i) for i in range(lda_model.n_components)]
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns = topic_names)
dominant_topic = (np.argmax(df_document_topic.values, axis=1))
df_document_topic['dominant_topic'] = dominant_topic
df_document_topic['dominant_topic'] = 'topic_' + df_document_topic['dominant_topic'].astype(str)

print(df_document_topic.shape[0])
print(Main_symp.shape[0])

df_document_topic=df_document_topic.reset_index(drop=True)
Main_symp=Main_symp.reset_index(drop=True)

Main_symp=Main_symp.join(df_document_topic)
print(Main_symp.head(5))


review_words = Main_symp['cleaned_review'].values.tolist()
review_topic = Main_symp['dominant_topic'].values.tolist()
cfd_topic_words = list(zip(review_words, review_topic))
cfd = nltk.ConditionalFreqDist((review[1], word) for review in cfd_topic_words for word in review[0])

cfd['topic_0'].most_common(50)

cfd['topic_1'].most_common(50)

cfd['topic_2'].most_common(50)

cfd['topic_3'].most_common(50)

cfd['topic_4'].most_common(50)

d = {'medication':cfd['topic_0'].most_common(50), 
     'school':cfd['topic_1'].most_common(50), 
     'adderall':cfd['topic_2'].most_common(50), 
     'effect':cfd['topic_3'].most_common(50), 
     'adhd':cfd['topic_4'].most_common(50)}

TOPICS_df = pd.DataFrame(data=d)

STATS0_DF = Main_symp.groupby('dominant_topic', as_index=False).agg({"vader_compound_polarity_score":"mean","drugName":"count"})
Topics = TOPICS_df.columns

plt.rcdefaults()
fig, ax = plt.subplots()

ax.barh(Topics, STATS0_DF["vader_compound_polarity_score"], align='center')

#ax.set_yticks(STATS0_DF["dominant_topic"], labels=str(Topics)
ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Vader Compound Polarity Score')
ax.set_title('LDA TOPICS and Vader Score')

plt.show()


Main_symp['topic_0_compound_polarity_score'] = Main_symp["vader_compound_polarity_score"]*Main_symp["topic_0"]
Main_symp['topic_1_compound_polarity_score'] = Main_symp["vader_compound_polarity_score"]*Main_symp["topic_1"]
Main_symp['topic_2_compound_polarity_score'] = Main_symp["vader_compound_polarity_score"]*Main_symp["topic_2"]
Main_symp['topic_3_compound_polarity_score'] = Main_symp["vader_compound_polarity_score"]*Main_symp["topic_3"]
Main_symp['topic_4_compound_polarity_score'] = Main_symp["vader_compound_polarity_score"]*Main_symp["topic_4"]

import copy
source_df0 = copy.deepcopy(Main_symp) 
Main_symp=Main_symp[['drugName','condition','rating','Grade','Spending',"vader_compound_polarity_score",'topic_0_compound_polarity_score','topic_1_compound_polarity_score','topic_2_compound_polarity_score','topic_3_compound_polarity_score','topic_4_compound_polarity_score']]

conditions = Main_symp['condition'].values.tolist()
conditions_fdist = nltk.FreqDist(conditions)
total_count = len([w for (w,c) in conditions_fdist.most_common()])
common_count = sum([c for (w,c) in conditions_fdist.most_common(50)])
common_conditions = [w for (w,c) in conditions_fdist.most_common(50)]
print('total conditions:',total_count)
print('common condition coverage:',common_count/Main_symp.shape[0])


Main_symp['condition']= Main_symp['condition'].apply(lambda condition: condition if condition in common_conditions else 'other')

drug_efficacy_dfs = Main_symp.groupby(by=['drugName','condition'],as_index = False).mean()

drug_efficacy_dfs['effective_drug']=drug_efficacy_dfs['rating'].apply(lambda rating: rating>7)

drug_efficacy_dfs = pd.get_dummies(data=drug_efficacy_dfs, drop_first = False, columns=['condition'])

print(drug_efficacy_dfs.columns)

drug_efficacy_Predict = drug_efficacy_dfs

drug_efficacy_df = drug_efficacy_dfs.drop(columns=['vader_compound_polarity_score'])
print(drug_efficacy_df.head(10))

X= drug_efficacy_df.drop(columns=['drugName','rating','effective_drug','Spending'])
y= drug_efficacy_df['effective_drug']

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.3, random_state=1)
X_price= drug_efficacy_df.drop(columns=['drugName','rating','effective_drug'])
y_price= drug_efficacy_df['effective_drug']
y_price_ols= drug_efficacy_df['rating']

Xy = copy.deepcopy(X_price) 
Xy["y"] = y_price
Xy["y_ols"] = y_price_ols

Xy.dropna(inplace=True)

y_price = Xy["y"]
y_price_ols = Xy["y_ols"]

del Xy["y"]
del Xy["y_ols"]
X_price=Xy

X_price['Spending'] = np.log(X_price['Spending'])

train_Xols, valid_Xols, train_yols, valid_yols = train_test_split(X_price, y_price_ols, test_size=0.3, random_state=1)

efficacy_ols = LinearRegression()
efficacy_ols.fit(train_Xols,train_yols)

print(pd.DataFrame({'Predictor':X_price.columns,'coefficient':efficacy_ols.coef_}))

regressionSummary(train_yols,efficacy_ols.predict(train_Xols))
train_X, valid_X, train_y, valid_y = train_test_split(X_price, y_price, test_size=0.3, random_state=1)

param_grid = {
    'max_depth': [number for number in range(1,16)],
    'criterion':['gini', 'entropy'],
}

grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=1),param_grid,scoring='accuracy', cv=3, n_jobs=-1)
grid_search_dt.fit(train_X, train_y)
efficacy_dt = grid_search_dt.best_estimator_
print(grid_search_dt.best_params_)
print("")

#Score the optimized decision tree
prediction_dt_valid = efficacy_dt.predict(valid_X)
print("Accuracy: ", accuracy_score(valid_y, prediction_dt_valid))

prediction_dt_train = efficacy_dt.predict(train_X)

opt_dt_score = ca_score_model(train_y, prediction_dt_train, valid_y, prediction_dt_valid)


param_grid = {
    'n_estimators': [100,200,300],
    'criterion':['gini', 'entropy'],
    'bootstrap': [True, False],
}

grid_search_forest = GridSearchCV(RandomForestClassifier(random_state=1),param_grid,scoring='accuracy', cv=3, n_jobs=-1)
grid_search_forest.fit(train_X, train_y)
efficacy_forest = grid_search_forest.best_estimator_
print(grid_search_forest.best_params_)
print("")

# Score the optimized Random Forest
prediction_forest_valid = efficacy_forest.predict(valid_X)
print("Accuracy: ", accuracy_score(valid_y, prediction_forest_valid))

param_grid = {
    'loss':['deviance', 'exponential'],
    'n_estimators': [100,200,300]
}

grid_search_boost = GridSearchCV(GradientBoostingClassifier(random_state=1),param_grid,scoring='accuracy', cv=3, n_jobs=-1)
grid_search_boost.fit(train_X, train_y)
efficacy_boost = grid_search_boost.best_estimator_
print(grid_search_boost.best_params_)

prediction_boost_valid = efficacy_boost.predict(valid_X)
print("Accuracy: ", accuracy_score(valid_y, prediction_boost_valid))



prediction_boost_train = efficacy_boost.predict(train_X)

opt_boost_score = ca_score_model(train_y, prediction_boost_train, valid_y, prediction_boost_valid)

estimators = [
('gbm', GradientBoostingClassifier(loss= 'exponential', n_estimators= 300, random_state=1)),
('rf', RandomForestClassifier(bootstrap= True, criterion= 'entropy', n_estimators= 100, random_state=1)),
('dt', DecisionTreeClassifier(criterion= 'entropy', max_depth= 10, random_state = 1))
]                             


efficacy_stack = StackingClassifier(
estimators=estimators, final_estimator=LogisticRegression(penalty= 'l2', solver= 'newton-cg',C=1e42, random_state = 1)
)

efficacy_stack.fit(train_X, train_y)

prediction_stack_valid = efficacy_stack.predict(valid_X)
print(accuracy_score(valid_y, prediction_stack_valid))

prediction_stack_train = efficacy_stack.predict(train_X)

stack_score = ca_score_model(train_y, prediction_stack_train, valid_y, prediction_stack_valid)

print(stack_score)

voterf =efficacy_stack.named_estimators_['rf'].feature_importances_
votegbm =efficacy_stack.named_estimators_['gbm'].feature_importances_
votedt =efficacy_stack.named_estimators_['dt'].feature_importances_
votes=list(zip(voterf,votegbm,votedt))

voimp = dict(zip(train_X.columns,votes))
label=['RandomForest','GradientBoost','DecisionTree']
final=pd.DataFrame(voimp.values(), index=voimp.keys(),columns=label)
final['Avg_Imp'] = final.mean(axis=1)

poslabel=['Avg_Imp','RandomForest','GradientBoost','DecisionTree']
final.sort_values(by =['Avg_Imp'], ascending = False).plot(y=poslabel,kind = 'bar',color=['r', 'b', 'orange', 'g'],figsize=(15,5),rot=80);
print(final)

grphscat= drug_efficacy_Predict.drop(columns=['Grade'])
grphscat.dropna(inplace=True)

chrt1 = grphscat.groupby('drugName', as_index=False).agg({"rating":"mean","vader_compound_polarity_score":"mean","Spending":"count"})

chrt1.reset_index()

chrt1.reset_index(drop=True, inplace=True)

# Sort medication treating a condition
chrt1 = chrt1.sort_values(by=["Spending"], ascending=False) 

chrt1["Spending10"] = (chrt1["Spending"]**2 )

bkPlot_HH = chrt1[:10].plot_bokeh.scatter('rating','vader_compound_polarity_score', 
                                     figsize=(900,720),
                                     category='drugName', 
                                     colormap='Viridis', 
                                     line_color='cyl', line_width=5,
                                     fontsize_legend=8, 
                                     legend="top_left",  
                                     title='Drug Rating and Vader Compound Polarity Score, per Drug & Spending', 
                                     size=35, alpha=3)

bkPlot_HH.grid.grid_line_color = None
bkPlot_HH.axis.minor_tick_line_color = None

bkPlot_HH.output_file("/Users/abelroman/Desktop/python_team3/prices45.html")

drug_efficacy_Predict['effective_drug'] == drug_efficacy_Predict['effective_drug'].apply((lambda x: 1 if x == 'TRUE' else 0))
                                                                                         
drug_efficacy_Predict['Prediction'] = drug_efficacy_Predict['rating'] + drug_efficacy_Predict['effective_drug'] + drug_efficacy_Predict['Grade'] * drug_efficacy_Predict['vader_compound_polarity_score']                                                                    

drug_efficacy_Predict = drug_efficacy_Predict.groupby(['drugName'])['Prediction'].mean().sort_values(ascending=False)

print('Recomended medication for', symp, ":", drug_efficacy_Predict[0:1])






