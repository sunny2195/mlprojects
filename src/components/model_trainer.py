import os   
import sys  
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor)
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,models= models)

            best_model_score = max(model_report.values())
            best_model_name = [name for name, score in model_report.items() if score == best_model_score][0]
            best_model = models[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            mse = mean_squared_error(y_test, predicted)
            rmse = np.sqrt(mse)

            return r2_square, rmse
        except Exception as e:
            raise CustomException(e, sys)

