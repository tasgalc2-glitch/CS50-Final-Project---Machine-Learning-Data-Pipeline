import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import joblib
from sklearn import set_config
import sys

set_config(transform_output="pandas")

class DataPipeline:
    """
    Handles data preprocessing steps for a machine learning workflow.

    Loads a dataset, cleans missing values, separates features and target,
    splits data into training/testing sets, and scales features.
    """
    def __init__(self, filename, target_column):
        self.filename = filename
        self.target_column = target_column
        self.scaler = StandardScaler()
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self) -> None:
        """
        Loads a CSV file into a pandas DataFrame.
        """
        self.df = pd.read_csv(self.filename)

    def clean_data(self) -> None:
        """
        Removes rows containing missing values from the dataset.
        """
        if self.df is None:
            raise ValueError("Data has not been loaded yet. ")
        
        self.df = self.df.dropna()

    def split_features_target(self) -> None:
        """
        Separates the dataset into feature variables (X) and target variable (y).
        """
        if self.df is None:
            raise ValueError("Data has not been loaded yet. ")

        self.X = self.df.drop(columns=self.target_column)
        self.y = self.df[self.target_column]

    def split_data(self) -> None:
        """
        Splits features and target data into training and testing sets.
        """
        if self.X is None or self.y is None:
            raise ValueError("Features and target have not been separated yet. ")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=42)

    def scale_features(self) -> None:
        """
        Standardizes feature values using StandardScaler.
        """
        if self.X_train is None or self.X_test is None:
            raise ValueError("Training and testing data have not been created yet. ")

        self.X_train = self.scaler.fit_transform(self.X_train)

        self.X_test = self.scaler.transform(self.X_test)


class ModelTrainer:
    """
    Handles training and evaluation of a machine learning classification model.
    """
    def __init__(self):
        self.model = LogisticRegression()
        self.accuracy = None

    def train_model(self, X_train: pd.DataFrame | np.ndarray, y_train: pd.Series | np.ndarray) -> None:
        """
        Trains the model using training features and labels.
        """
        self.model.fit(X_train, y_train)


    def evaluate(self, X_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray) -> float:
        """
        Evaluates model performance using accuracy score.

        Returns:
            Model accuracy.
        """    
        predictions = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_true=y_test, y_pred=predictions)

        return self.accuracy

    def generate_report(self, X_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray):
        """
        Generates a classification report containing precision,
        recall, and F1-score.
        """
        predictions = self.model.predict(X_test)
        return classification_report(y_test, predictions)

    def generate_confusion_matrix(self, X_test, y_test):
        """
        Creates a confusion matrix showing prediction results.
        """
        predictions = self.model.predict(X_test)
        return confusion_matrix(y_test, predictions)


class MLSystem:
    """
    Represents a complete trained machine learning system containing
    a model and preprocessing scaler.
    """
    def __init__(self, model, scaler: StandardScaler):
        self.model = model
        self.scaler = scaler

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Scales new data and generates predictions using the trained model.
        """
        scaled_data = self.scaler.transform(data)

        prediction = self.model.predict(scaled_data)

        return prediction

    def save(self, filename: str) -> None:
        """
        Saves the trained machine learning system to a file.
        """
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename: str) -> "MLSystem":
        """
        Loads a previously saved machine learning system.
        """
        return joblib.load(filename)



        


def main():
    while True:
        choice = input(
        "1. Train a new model\n"
        "2. Load an existing model\n"
        "3. Exit\n\n"
        "Choice: "
        )

        
        if choice == "1":
            input_filename_1 = input("Type the name of the file you want to load that ends in .csv:\n")
            input_target_column_1 = input("Type the name of the target column:\n")
            pipeline = DataPipeline(input_filename_1, input_target_column_1)
            
            try:
                pipeline.load_data()
            except FileNotFoundError:
                print("CSV file not found. ")
                continue
            pipeline.clean_data()
            pipeline.split_features_target()
            pipeline.split_data()
            pipeline.scale_features()
            
            trainer = ModelTrainer()
            
            trainer.train_model(X_train=pipeline.X_train, y_train=pipeline.y_train)
            accuracy = trainer.evaluate(pipeline.X_test, pipeline.y_test)
            
            print(trainer.generate_report(
                X_test=pipeline.X_test,
                y_test=pipeline.y_test))
            
            print(f"Accuracy: {accuracy:.2%}")
            
            system = MLSystem(model=trainer.model, scaler=pipeline.scaler)

            save_input_1 = input("To save, type the name you would like to give (without ending in .pkl):\n")
            
            system.save(f"{save_input_1}.pkl")

            print(f"Model successfully saved as {save_input_1}.pkl ")

        elif choice == "2":
            input_filename_2 = input("Please type the name of the file you would like to load (without ending in .pkl)\n")
            try:
                system = MLSystem.load(f"{input_filename_2}.pkl")
            except FileNotFoundError:
                print("Model file not found. ")
                return
            print("Model loaded successfully! ")
            print("Enter patient data separated by spaces:")
            data = sys.stdin.readline().split()

            age = float(data[0])
            sex = int(data[1])
            cp = int(data[2])
            trestbps = float(data[3])
            chol = float(data[4])
            fbs = int(data[5])
            restecg = int(data[6])
            thalach = float(data[7])
            exang = int(data[8])
            oldpeak = float(data[9])
            slope = int(data[10])
            ca = int(data[11])
            thal = int(data[12])

            patient_data = {"age": age,
                        "sex": sex,
                        "cp": cp,
                        "trestbps": trestbps,
                        "chol": chol,
                        "fbs": fbs,
                        "restecg": restecg,
                        "thalach": thalach, 
                        "exang": exang,
                        "oldpeak": oldpeak,
                        "slope": slope, 
                        "ca": ca, 
                        "thal": thal}

            
            patient = pd.DataFrame([patient_data])
            prediction = system.predict(patient)
            if prediction[0] == 0:
                print("Prediction: No heart disease detected. ")
            else:
                print("Prediction: Heart disease detected. ")


        elif choice == "3":
            print("Goodbye. ")
            break

        else:
            print("Please select a valid choice. ")

    
if __name__ == "__main__":
     main()

    




