import numpy as np
from datetime import datetime
import argparse
import pickle, os
from sklearn_crfsuite import CRF
import mlflow
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

class train_ner():
    def __init__(self):
        pass

    def load_dataset(self, file_path):
        # Load dataset dari .pkl
        try:
            path = os.path.join("dataset", file_path)
            with open(path, "rb") as f:
                data = pickle.load(f)
            X_train = data["X_train"]
            Y_train = data["Y_train"]
            X_test = data["X_test"]
            Y_test = data["Y_test"]
            print("Load Dataset completed successfully.")
        except Exception as e:
            print(f"Error during load dataset: {e}")
            return False
        return X_train, Y_train, X_test, Y_test
    
    def training_model(self, file_path, exp_name):
        X_train, Y_train, X_test, Y_test = self.load_dataset(file_path)
        try:
            with mlflow.start_run():
                mlflow.sklearn.autolog()

                model = LogisticRegression(max_iter=200)
                model.fit(X_train, Y_train)

                y_pred = model.predict(X_test)

                report = classification_report(Y_test, y_pred, digits=3)
                print(report)

            with open("classification_report.txt", "w") as f:
                f.write(report)

            mlflow.sklearn.log_model(model, "model_lr")
            with open("saved_model/lr_model.pkl", "wb") as f:
                pickle.dump(model, f)

            print("Training Model completed successfully.")
        except Exception as e:
            print(f"Error during training model: {e}")
            return False
        return True           

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="ner_dataset_ML.pkl")
    args = parser.parse_args()
    train = train_ner()
    file_path = args.file_path
    exp_name = "NER_MachineLearning"
    train.training_model(file_path, exp_name)
