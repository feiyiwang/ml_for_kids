import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix,roc_auc_score, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
np.random.seed(42)

model_mapping = {'SVM': SVC, 'Logistic Regression': LogisticRegression, 'Random Forest': RandomForestClassifier}


def load_data(file):
    '''
    detailed info of the dataset can be found via 
    https://archive.ics.uci.edu/ml/datasets/heart%2BDisease
    '''
    data = pd.read_csv(file, index_col=[0])
    data = data[['Age', 'Sex', 'ChestPain', 'Slope', 'ExAng', 'Thal', 'AHD']]
    data.Thal = data.Thal.fillna('normal') # 2 N/As
    data = data.rename(columns={
        # 'RestBP':'Resting_Blood_Pressure',
        'ChestPain':'Chest_Pain',
        # 'Chol':'Cholesterol',
        # 'Fbs':'Fasting_Blood_Sugar',
        # 'MaxHR':'Maximum_Heart_Rate',
        'ExAng':'Has_Angina',
        'Thal':'Thalassemia', # 地中海贫血
        'AHD':'Has_Heart_Disease' # Adenovirus Hemorrhagic Disease 腺病毒出血性疾病
    })
    print('Your data for heart disease analysis is well loaded! 😀')
    print('Number of rows:', len(data))
    print('Number of columns:', len(data.columns))
    return data


def dummy(data, dummy_list):
    encoding = pd.get_dummies(data[dummy_list])
    data_final = pd.concat([data, encoding],1)
#     data_final = data_final.drop(['Thalassemia', 'Has_Heart_Disease'], axis = 1)
    return data_final


def draw_basic_plot(data, column, colors):
    data[column].value_counts().plot(kind='bar',figsize=(10,6),color=colors)
    plt.title("Distribution of "+column)
    plt.show()


def draw_bar_plot(data, column='Sex', tags=['Female', 'Male'], colors=['pink', 'blue']):
    pd.crosstab(data['Has_Heart_Disease_Yes'],data[column]).plot(kind='bar',figsize=(20,10),color=colors)
    plt.title("Frequency of Heart Disease vs "+column, size=28)
    plt.xlabel("0 = Has_Heart_Disease_Yes, 1 = Has_Heart_Disease_NO", size=18)
    plt.ylabel("Number of people with heart disease", size=18)
    plt.legend(tags, prop={'size':16})
    plt.xticks(rotation=0, size=14)
    plt.yticks(rotation=0, size=14)


def draw_heatmap(data):
    data = data.drop(['Thalassemia', 'Has_Heart_Disease'], axis = 1)
    cor_mat=data.corr()
    fig,ax=plt.subplots(figsize=(15,10))
    sns.heatmap(cor_mat,annot=True,linewidths=0.5,fmt=".3f")


def split_data(data, test_proportion):
    X = data.drop('Has_Heart_Disease', axis=1)
    Y = data.Has_Heart_Disease
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_proportion, random_state=1)
    print('You have successfully splitted your data! 🎉')
    print('Now you have', len(X_train), 'samples in the training dataset.')
    print('And you have', len(X_test), 'samples in the test dataset.')
    return X_train,X_test,y_train,y_test


def scale(data):
    scaled_data = data
    scaler=MinMaxScaler()
    features=['Age', 'Sex', 'Has_Angina',
              'Chest_Pain_asymptomatic', 'Chest_Pain_nonanginal',
              'Chest_Pain_nontypical', 'Chest_Pain_typical',
              'Thalassemia_fixed', 'Thalassemia_normal', 'Thalassemia_reversable',
              'Has_Heart_Disease_Yes']
    scaled_data[features] = scaler.fit_transform(data[features])
    scaled_data = scaled_data[features].rename(columns={'Has_Heart_Disease_Yes':'Has_Heart_Disease'})
    return scaled_data


def select_features(X_train, y_train):
    mutual_info = mutual_info_regression(X_train, y_train)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = X_train.columns
    mutual_info = mutual_info.sort_values(ascending=False)
    print(mutual_info)


def evaluate(Y_test,Y_pred):
    acc=accuracy_score(Y_test,Y_pred)
    rcl=recall_score(Y_test,Y_pred)
    f1=f1_score(Y_test,Y_pred)
    auc_score=roc_auc_score(Y_test,Y_pred)
    prec_score=precision_score(Y_test,Y_pred)
    
    metric_dict={'accuracy': round(acc*100,2),
               'recall': round(rcl*100,2),
               'F1 score': round(f1*100,2),
               'auc score': round(auc_score*100,2),
               'precision': round(prec_score*100,2)
                }
    
    return print(metric_dict)


def model(datasets, model_name):
    X_train,X_test,y_train,y_test = datasets
    model=model_mapping[model_name]()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print("Accuracy on Training set: ",round(model.score(X_train,y_train)*100,2))
    model_score = round(model.score(X_test,y_test)*100,2)
    print("Accuracy on Testing set: ", model_score)
    print()
    evaluate(y_test, y_pred)



def make_meshgrid(x, y, h=.05):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def visualize(datasets, model_name):
    X, _, y, _ = datasets
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    model = model_mapping[model_name]()
    model = model.fit(X_pca, y)

    fig, ax = plt.subplots(figsize=(10, 8))
    # title for the plots
    # Set-up grid for plotting.
    X0, X1 = X_pca[:, 0], X_pca[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, model, xx, yy, cmap='Set3', alpha=0.2)
    scatter = ax.scatter(X0, X1, c=y, cmap='Set3', s=60, edgecolors='white')
    handles, _ = scatter.legend_elements(prop='colors')
    legend = ax.legend(handles, ['Has_Heart_Disease_No','Has_Heart_Disease_Yes'])
    ax.add_artist(legend)
    ax.set_ylabel('Second principal component', size=14)
    ax.set_xlabel('First principal component', size=14)
    ax.set_title('Decision Boundary of '+model_name, size=20)
    plt.show()