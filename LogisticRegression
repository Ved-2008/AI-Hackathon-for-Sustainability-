import pandas as pd #for csv data
import matplotlib.pyplot as plt # to plot data
import seaborn as sns # for scatterplot and graph

from sklearn.model_selection import train_test_split # train the data
from sklearn.linear_model import LogisticRegression # algorithm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def load_data(filepath, feature_cols, target_col):
    '''Loads the data'''
    df = pd.read_csv(filepath)
    X = df[feature_cols]
    y = df[target_col]
    return X, y

def preprocess_data(X, y, test_size=0.2, random_state=42):
    '''Splits dataset into testing and training set'''
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    # Scales data so that that mean=0, standard deviation=1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    '''Compare predictions of test data to the actual labels and plot matrix'''
    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Leak", "Leak"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

def plot_feature_distribution(X, y, feature_cols):
    '''Create pairplot (grid of scatterplot and histogram)'''
    df = X.copy()
    df['Leak'] = y
    sns.pairplot(df, hue="Leak", vars=feature_cols)
    plt.suptitle("Feature Distribution by Leak Status", y=1.02)
    plt.show()


def main():
    filepath = "C:/Users/vedik/Downloads/location_aware_gis_leakage_dataset.csv"
    feature_cols = ['Pressure', 'Flow_Rate', 'Temperature', 'Vibration', 'RPM', 'Operational_Hours']
    target_col = 'Leakage_Flag' #target column

    X, y = load_data(filepath, feature_cols, target_col)
    plot_feature_distribution(X, y, feature_cols)

    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__=="__main__":
    main()