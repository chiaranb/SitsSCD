import numpy as np
from capymoa.base import Classifier
from capymoa.instance import LabeledInstance, Instance
from capymoa.stream import Schema
from river.base import Classifier as RiverBaseClassifier

class RiverClassifier(Classifier):
    """
    A CapyMOA wrapper for River classifiers AND pipelines.
    
    This class allows you to use any instantiated River classifier
    or compose.Pipeline object within the CapyMOA framework.
    """
    
    def __init__(self, schema: Schema, river_model_instance: RiverBaseClassifier):
        """
        Initialize the wrapper.
        
        :param schema: The CapyMOA schema for the data stream.
        :param river_model_instance: An *instantiated* River classifier
                                     or pipeline (e.g., LogisticRegression() 
                                     or compose.Pipeline(...)).
        """
        super().__init__(schema=schema, random_seed=1) 
        
        # Store the provided instance
        self.river_model: RiverBaseClassifier = river_model_instance
        
        # Store schema info for fast conversions
        self.feature_names = self.schema.get_numeric_attributes()
        self.class_labels = self.schema.get_label_values()
        self.num_classes = len(self.class_labels)
        
        # Create a mapping from class label (str/int) to class index (int)
        self._class_to_index_map = {
            int(label) if str(label).isdigit() else label: i
            for i, label in enumerate(self.class_labels)
        }
        
        
    def _capy_to_river_x(self, instance: Instance) -> dict:
        """Converts a CapyMOA numpy array instance to a River dict."""
        return {name: val for name, val in zip(self.feature_names, instance.x)}

    def train(self, instance: LabeledInstance):
        x_river = self._capy_to_river_x(instance)
        # Convert label to int if possible
        label_value = self.class_labels[instance.y_index]
        try:
            y_river = int(label_value)
        except ValueError:
            y_river = label_value
        self.river_model.learn_one(x_river, y_river)

    def predict_proba(self, instance: Instance) -> np.ndarray:
        x_river = self._capy_to_river_x(instance)
        try:
            proba_dict = self.river_model.predict_proba_one(x_river)
        except Exception:
            proba_dict = {}

        prob_array = np.zeros(self.num_classes)
        if not proba_dict:
            prob_array.fill(1.0 / self.num_classes)
            return prob_array

        for label, prob in proba_dict.items():
            # Convert back if River uses int labels
            key = int(label) if str(label).isdigit() else label
            if key in self._class_to_index_map:
                idx = self._class_to_index_map[key]
                prob_array[idx] = prob

        total_prob = np.sum(prob_array)
        if total_prob > 0:
            prob_array /= total_prob
        else:
            prob_array.fill(1.0 / self.num_classes)
        return prob_array

    def predict(self, instance: Instance) -> int:
        """
        Get the predicted class index from the wrapped River model.
        """
        prob_array = self.predict_proba(instance)
        return np.argmax(prob_array)