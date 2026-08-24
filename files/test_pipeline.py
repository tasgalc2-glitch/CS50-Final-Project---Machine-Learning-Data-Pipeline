import pandas as pd
from pipeline import DataPipeline, ModelTrainer, MLSystem
from sklearn.linear_model import LogisticRegression

def test_clean_data():
    pipeline = DataPipeline("hearts.csv", "target")

    pipeline.load_data()
    pipeline.clean_data()

    assert pipeline.df.isna().sum().sum() == 0

def test_split_features_target():
    pipeline = DataPipeline("hearts.csv", "target")

    pipeline.load_data()
    pipeline.clean_data()
    pipeline.split_features_target()

    assert pipeline.X is not None
    assert pipeline.y is not None

    assert "target" not in pipeline.X.columns
    assert pipeline.y.name == "target"   

def test_train_model():
    pipeline = DataPipeline("hearts.csv", "target")

    pipeline.load_data()
    pipeline.clean_data()
    pipeline.split_features_target()
    pipeline.split_data()
    pipeline.scale_features()

    trainer = ModelTrainer()

    trainer.train_model(X_train=pipeline.X_train, y_train=pipeline.y_train)

    assert isinstance(trainer.model, LogisticRegression)

def test_predict():
    pipeline = DataPipeline("hearts.csv", "target")

    pipeline.load_data()
    pipeline.clean_data()
    pipeline.split_features_target()
    pipeline.split_data()
    pipeline.scale_features()

    trainer = ModelTrainer()

    trainer.train_model(X_train=pipeline.X_train, y_train=pipeline.y_train)

    system = MLSystem(model=trainer.model, scaler=pipeline.scaler)

    prediction = system.predict(pipeline.X_test.iloc[[0]])

    assert prediction is not None
    assert prediction[0] in [0, 1]