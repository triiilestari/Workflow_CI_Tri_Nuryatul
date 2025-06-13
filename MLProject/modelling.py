import numpy as np
import tensorflow as tf
from datetime import datetime
import argparse
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense, Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import mlflow

class train_ner():
    def __init__(self):
        self.maxlen = 110
        self.max_words = 36000

    def load_dataset(self, file_path):
        # Load dataset dari .pkl
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        X_train = data["X_train"]
        Y_train = data["Y_train"]
        X_test = data["X_test"]
        Y_test = data["Y_test"]
        return X_train, Y_train, X_test, Y_test
    
    
    def training_model_lstm(self, file_path, exp_name, embedding_dim, batch_size, epochs):
        # Load Dataset
        X_train, Y_train, X_test, Y_test = self.load_dataset(file_path)

        # Hyperparameter
        vocab_size = int(np.max(np.concatenate([X_train, X_test]))) + 1
        tag_size = int(np.max(Y_train))+1
        lstm_units = 64
        learning_rate = 0.001

        early_stopping_cb = EarlyStopping(
            monitor="loss",
            patience=2,
            restore_best_weights=True,
            verbose=1
            )

        checkpoint_cb = ModelCheckpoint(
            "best_model_lstm.h5",
            monitor="accuracy",
            save_best_only=True,
            verbose=1
            )

        # Logging manual ke MLflow
        with mlflow.start_run():
            # Model
            model = Sequential([
                Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
                Bidirectional(LSTM(units=lstm_units, return_sequences=True)),
                TimeDistributed(Dense(tag_size, activation="softmax"))
            ])

            model.compile(optimizer=Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

            # Train
            history = model.fit(
                X_train, Y_train,
                validation_data=(X_test, Y_test),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[early_stopping_cb, checkpoint_cb]
            )

            stopped_epoch = early_stopping_cb.stopped_epoch
            # Manual metric logging
            val_loss, val_acc = model.evaluate(X_test, Y_test)
            mlflow.log_metric("val_loss", val_loss)
            mlflow.log_metric("val_accuracy", val_acc)

            # Simpan model
            mlflow.keras.log_model(model, "model_lstm")
        
    def training_model_sequence(self, file_path, exp_name, embedding_dim, batch_size, epochs):
        # Load Dataset
        X_train, Y_train, X_test, Y_test = self.load_dataset(file_path)
        # Hyperparameter
        vocab_size = int(np.max(np.concatenate([X_train, X_test]))) + 1# Asumsi sudah tokenized integer
        tag_size = int(np.max(Y_train))+1
        # Hyperparameter
        filters = 64
        kernel_size = 3
        learning_rate = 0.001

        checkpoint_cb = ModelCheckpoint(
        "best_model_sequence.h5",
        monitor="accuracy",
        save_best_only=True,
        verbose=1
        )

        early_stopping_cb = EarlyStopping(
        monitor="loss",
        patience=2,
        restore_best_weights=True,
        verbose=1
        )
        
        # Logging manual ke MLflow
        with mlflow.start_run():
            # Build model
            model = Sequential([
                Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
                Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu'),
                TimeDistributed(Dense(tag_size, activation='softmax'))
            ])

            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
            # Train
            history = model.fit(
                X_train, Y_train,
                validation_data=(X_test, Y_test),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[early_stopping_cb, checkpoint_cb]
            )

            stopped_epoch = early_stopping_cb.stopped_epoch

            # Evaluate
            val_loss, val_acc = model.evaluate(X_test, Y_test)
            mlflow.log_metric("val_loss", val_loss)
            mlflow.log_metric("val_accuracy", val_acc)
            model.save("ner_model_sequence.h5")
            mlflow.log_artifact("ner_model_sequence.h5")
            # Log artifacts
            mlflow.keras.log_model(model, "model_sequence")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--file_path", type=str, default="ner_dataset_split.pkl")
    args = parser.parse_args()
    embedding_dim = args.embedding_dim
    batch_size = args.batch_size
    epochs = args.epochs
    train = train_ner()
    file_path = args.file_path
    exp_name = "NER_DeepLearning"
    train.training_model_sequence(file_path, exp_name, embedding_dim, batch_size, epochs)
