import pandas as pd
import numpy as np
import joblib
from typing import Dict, Tuple

class SignalGenerator:
    """
    Generate trading signals based on model predictions and probability thresholds.
    """
    
    def __init__(self, model, probability_threshold: float = 0.60):
        """
        Initialize signal generator.
        
        Args:
            model: Trained model with predict_proba method
            probability_threshold (float): Minimum probability to generate a signal
        """
        self.model = model
        self.probability_threshold = probability_threshold
    
    def generate_signal(self, features: pd.DataFrame) -> Dict:
        """
        Generate a trading signal from feature data.
        
        Args:
            features (pd.DataFrame): Single row or multiple rows of features
        
        Returns:
            Dict: Signal information with keys:
                - 'action': 'CALL', 'PUT', or 'NO_TRADE'
                - 'probability': Model's confidence
                - 'features_used': Number of features
        """
        # Get prediction probability
        probabilities = self.model.predict_proba(features)
        
        # For binary classification: probabilities[:, 1] is P(UP)
        prob_up = probabilities[:, 1]
        prob_down = probabilities[:, 0]
        
        signals = []
        
        for i in range(len(features)):
            p_up = prob_up[i]
            p_down = prob_down[i]
            
            # Determine signal
            if p_up >= self.probability_threshold:
                action = 'CALL'  # Predict price will go UP
                confidence = p_up
            elif p_down >= self.probability_threshold:
                action = 'PUT'   # Predict price will go DOWN
                confidence = p_down
            else:
                action = 'NO_TRADE'
                confidence = max(p_up, p_down)
            
            signal = {
                'action': action,
                'probability': confidence,
                'prob_up': p_up,
                'prob_down': p_down,
                'features_used': features.shape[1]
            }
            
            signals.append(signal)
        
        # Return single signal if input was single row
        return signals[0] if len(signals) == 1 else signals
    
    @staticmethod
    def load_model(model_path: str):
        """Load a saved model."""
        return joblib.load(model_path)
