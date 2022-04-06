import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from PIL import Image
matplotlib.use('Agg')

# set Title

st.title('Data Science Employment')
image = Image.open('streamlit.png')
st.image(image, use_column_width=True)


def main():
    activities = ['EDA', 'Visualization', 'model', 'About me']
    option = st.sidebar.selectbox('Selection Option:', activities)

# EDA PART
    if option == 'EDA':
        st.subheader("Exploratory Data Analysis")

        data = st.file_uploader("Upload Dataset:", type=['csv', 'xlsx', 'txt', 'json'])
        st.success('Data Successfully loaded')
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

            if st.checkbox("Display shape"):
                st.write(df.shape)

            if st.checkbox('Display columns'):
                st.write(df.columns)

            if st.checkbox('Select multiple columns'):
                select_columns = st.multiselect('Select preferred columns:', df.columns)
                df1 = df[select_columns]
                st.dataframe(df1)

            if st.checkbox("Display summary"):
                st.write(df.describe().T)

            if st.checkbox('Display Null values'):
                st.dataframe(df.isnull().sum())

            if st.checkbox('Display teh data types'):
                st.write(df.dtypes)

            if st.checkbox('Display Correlation of data various columns '):
                st.write(df.corr())


# DEALING WITH THE VISUALIZATION PART
    elif option == 'Visualization':
        st.subheader('Data Visualization')
        data = st.file_uploader("Upload Dataset:", type=['csv', 'xlsx', 'txt', 'json'])
        if data is not None:
            st.success('Data Successfully loaded')
            df = pd.read_csv(data)
            st.dataframe(df.head())

            if st.checkbox('Selection Multiple columns to plot'):
                selected_columns = st.multiselect("Select your preferred columns", df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1)

            if st.checkbox('Display Heatmap'):
                st.write(sns.heatmap(df.corr(), vmax=1, square=True, annot=True, cmap='viridis'))
                st.pyplot()

            if st.checkbox('Display Pairplot'):
                st.write(sns.pairplot(df1, diag_kind='kde'))
                st.pyplot()

            if st.checkbox('Display pie chart'):
                all_columns = df.columns.to_list()
                pie_columns = st.selectbox('select column to display', all_columns)
                pieChart = df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pieChart)
                st.pyplot()

    elif option == 'model':
        st.subheader('Model Building')
        data = st.file_uploader("Upload Dataset:", type=['csv', 'xlsx', 'txt', 'json'])
        if data is not None:
            st.success('Data Successfully loaded')
            df = pd.read_csv(data)
            st.dataframe(df.head())

            if st.checkbox('Select Multiple columns'):
                new_data = st.multiselect('Select your preferred columns, NB: Let your target variable be the last column', df.columns)
                df1 = df[new_data]
                st.dataframe(df1)

                # Dividing my data into X and y
                X = df1.iloc[:, 0:-1]
                y = df1.iloc[:, -1]
            seed = st.sidebar.slider('Seed', 1, 200)

            classifier_name = st.sidebar.selectbox('Select your preferred classifier', ('KNN', 'SVM', 'Logistic'))

            def add_parameter(name_of_clf):
                params = dict()
                if name_of_clf == 'SVM':
                    C = st.sidebar.slider('C', 0.01, 15)
                    params['C'] = C

                if name_of_clf == 'KNN':
                    K = st.sidebar.slider('K', 1, 15)
                    params['K'] = K

                return params

            # calling the function
            params = add_parameter(classifier_name)

            # defining a function of our classifier

            def get_classifier(name_of_clf, params):
                clf = None
                if name_of_clf == 'SVM':
                    clf = SVC(C=params['C'])

                elif name_of_clf == 'KNN':
                    clf = KNeighborsClassifier(n_neighbors=params['K'])

                elif name_of_clf == 'Logistic':
                    clf = LogisticRegression()

                else:
                    st.warning('Select your choice of algorithm')

                return clf

            clf = get_classifier(classifier_name, params)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

            clf.fit(X_train, y_train)

            y_pred  = clf.predict(X_test)
            st.write('Prediction', y_pred)

            accuracy = accuracy_score(y_test, y_pred)

            st.write("Model: {} Accuracy:{}".format(classifier_name, accuracy))

    elif option == 'About me':
        st.markdown('This is an interactive web page for our ML project, feel free to use it.')
        st.markdown('Created By Guanze. You can contact me by Guanze.ec@gmail.com')
        st.balloons()


if __name__ == "__main__":
    main()


