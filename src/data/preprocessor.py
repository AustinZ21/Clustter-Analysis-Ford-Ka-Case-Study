import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Dict

class DataPreprocessor:
    """Handles data standardization and preparation"""
    
    def __init__(self):
        """Initialize preprocessor with standard scaler"""
        self.scaler = StandardScaler()
        self.demo_vars = ['Age', 'MaritalStatus', 'Gender', 'NumberChildren', 
                         'IncomeCategory', 'FirstTimePurchase']
        self.psyc_vars = [f'Q{i}' for i in range(1, 63)]
    
    def preprocess_all_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Preprocess all datasets"""
        processed_data = data.copy()
        
        # Standardize demographic data
        processed_data['demo_std'] = self.standardize_data(
            data['demographic'], 
            self.demo_vars
        )
        
        # Standardize psychographic data
        processed_data['psyc_std'] = self.standardize_data(
            data['psychographic'], 
            self.psyc_vars
        )
        
        # Standardize combined data
        numeric_cols = self.demo_vars + self.psyc_vars
        processed_data['combined_std'] = self.standardize_data(
            data['combined'], 
            numeric_cols
        )
        
        return processed_data
    
    def standardize_data(self, data: pd.DataFrame, 
                        columns: List[str]) -> pd.DataFrame:
        """
        Standardize specified columns in the dataset
        Args:
            data: Input DataFrame
            columns: List of columns to standardize
        Returns:
            Standardized DataFrame
        """
        return pd.DataFrame(
            self.scaler.fit_transform(data[columns]),
            columns=columns,
            index=data.index
        )
    
    @property
    def demographic_variables(self) -> List[str]:
        """Get list of demographic variables"""
        return self.demo_vars
    
    @property
    def psychographic_variables(self) -> List[str]:
        """Get list of psychographic variables"""
        return self.psyc_vars 