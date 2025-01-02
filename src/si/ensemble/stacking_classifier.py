import numpy as np
from si.metrics.accuracy import accuracy
from si.data.dataset import Dataset
from si.base.model import Model


class StackingClassifier(Model):
    def __init__(self, models:list, final_model,**kwargs):
        """
        Initialize the Stacking Classifier ensemble model

        Parameters
        ----------
        models : list
            Array-like of base models to be combined in the ensemble.
            Each model should be an instance of a Model class.
        final_model :
            Model to be used as the meta-model and create the final predictions.
            The model must be an instance of a Model class
        """
        
        # parameters
        super().__init__(**kwargs)
        self.models = models
        self.final_model = final_model

        # attributes
        self.new_dataset = None
    
    def _fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Fit the StackingClassifier ensemble model to the given training data.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the model to (training dataset)

        Returns
        -------
        self : StackingClassifier
            The fitted model
        """
        # Fit the base models
        for model in self.models:
            model.fit(dataset)

        # Genarate the base models predictions
        base_predictions = [model.predict(dataset) for model in self.models]
        base_predictions = np.array(base_predictions).T

        # Create a new dataset with the base models predictions
        self.new_dataset = Dataset(X=base_predictions, y=dataset.y, features = [f"{model}" for model in self.models] ,label= dataset.label)
        
        # Fit the final model (meta-model)
        self.final_model.fit(self.new_dataset)

        return self
    
    def _predict(self, dataset:Dataset) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        dataset : Dataset
            The dataset to make predictions on

        Returns
        -------
        np.ndarray
            The predicted class labels for the samples in X
        """
        # Base models predictions
        base_predictions = [model.predict(dataset) for model in self.models]
        base_predictions = np.array(base_predictions).T

        # Create a new dataset with the base models predictions
        new_dataset = Dataset(X=base_predictions, y=dataset.y, features = [f"{model}" for model in self.models] ,label=dataset.label)
        
        # Make predictions with the final model (meta-model)
        return self.final_model.predict(new_dataset)
    
    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        dataset : Dataset
            The test data.
        predictions: np.ndarray
            Predictions

        Returns
        -------
        score : float
            Mean accuracy
        """
        return accuracy(dataset.y, predictions)