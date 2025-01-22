"""Data loading and preparation module for Ford Ka analysis.

This module implements best practices for data handling including:
- Robust data validation
- Comprehensive error handling
- Data quality checks
- Efficient data loading
- Type validation and conversion
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from dataclasses import dataclass
import yaml
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data loading and validation."""
    
    demographic_cols: List[str]
    psychographic_cols: List[str]
    required_files: List[str]
    categorical_cols: List[str]
    numeric_cols: List[str]
    id_col: str
    target_col: Optional[str] = None

class DataLoader:
    """Handles data loading and preparation for analysis."""
    
    def __init__(self, data_dir: Union[str, Path], config_path: Optional[str] = None):
        """Initialize data loader.
        
        Args:
            data_dir: Path to directory containing data files
            config_path: Optional path to configuration file
        """
        self.data_dir = Path(data_dir)
        self.config = self._load_config(config_path)
        self.label_encoders = {}
        
    def _load_config(self, config_path: Optional[str]) -> DataConfig:
        """Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            DataConfig object
        """
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return DataConfig(**config_dict)
        
        # Default configuration
        return DataConfig(
            demographic_cols=['Age', 'MaritalStatus', 'Gender', 'NumberChildren', 
                            'IncomeCategory', 'FirstTimePurchase'],
            psychographic_cols=[f'Q{i}' for i in range(1, 63)],
            required_files=['FordKaDemographicData.csv', 
                          'FordKaPsychographicData.csv',
                          'FordKaSegmentData.csv'],
            categorical_cols=['MaritalStatus', 'Gender', 'IncomeCategory'],
            numeric_cols=['Age', 'NumberChildren'] + [f'Q{i}' for i in range(1, 63)],
            id_col='ID'
        )
    
    def validate_data_directory(self) -> None:
        """Validate presence and format of required data files."""
        missing_files = []
        for file_name in self.config.required_files:
            if not (self.data_dir / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing required files: {', '.join(missing_files)}"
            )
    
    def load_and_validate_file(self, file_path: Path) -> pd.DataFrame:
        """Load and validate a single data file.
        
        Args:
            file_path: Path to data file
            
        Returns:
            Validated DataFrame
        """
        try:
            df = pd.read_csv(file_path, index_col=0)
            logger.info(f"Loaded {file_path.name}: {df.shape}")
            
            # Validate ID column
            if self.config.id_col not in df.columns:
                raise ValueError(f"Missing ID column: {self.config.id_col}")
            
            # Check for duplicates
            duplicates = df[self.config.id_col].duplicated()
            if duplicates.any():
                logger.warning(f"Found {duplicates.sum()} duplicate IDs in {file_path.name}")
                df = df[~duplicates]
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {str(e)}")
            raise
    
    def preprocess_demographic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess demographic data.
        
        Args:
            df: Raw demographic DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        processed = df.copy()
        
        # Handle missing values
        for col in self.config.demographic_cols:
            if col in self.config.numeric_cols:
                processed[col] = processed[col].fillna(processed[col].median())
            else:
                processed[col] = processed[col].fillna(processed[col].mode()[0])
        
        # Encode categorical variables
        for col in self.config.categorical_cols:
            if col in processed.columns:
                le = LabelEncoder()
                processed[col] = le.fit_transform(processed[col].astype(str))
                self.label_encoders[col] = le
        
        return processed
    
    def preprocess_psychographic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess psychographic data.
        
        Args:
            df: Raw psychographic DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        processed = df.copy()
        
        # Handle missing values
        for col in self.config.psychographic_cols:
            processed[col] = processed[col].fillna(processed[col].median())
        
        # Validate response ranges
        for col in self.config.psychographic_cols:
            invalid_responses = ~processed[col].between(1, 5)
            if invalid_responses.any():
                logger.warning(
                    f"Found {invalid_responses.sum()} invalid responses in {col}"
                )
                processed.loc[invalid_responses, col] = processed[col].median()
        
        return processed
    
    def load_and_prepare_data(self) -> Dict[str, pd.DataFrame]:
        """Load and prepare all data for analysis.
        
        Returns:
            Dictionary containing prepared DataFrames
        """
        # Validate data directory
        self.validate_data_directory()
        
        # Load raw data
        demographic_data = self.load_and_validate_file(
            self.data_dir / 'FordKaDemographicData.csv'
        )
        psychographic_data = self.load_and_validate_file(
            self.data_dir / 'FordKaPsychographicData.csv'
        )
        segment_data = self.load_and_validate_file(
            self.data_dir / 'FordKaSegmentData.csv'
        )
        
        # Preprocess data
        demographic_processed = self.preprocess_demographic_data(demographic_data)
        psychographic_processed = self.preprocess_psychographic_data(psychographic_data)
        
        # Merge data
        full_data = pd.merge(
            demographic_processed,
            psychographic_processed,
            on=self.config.id_col,
            validate='1:1'
        )
        
        if 'PreferenceGroup' in segment_data.columns:
            full_data = pd.merge(
                full_data,
                segment_data[['ID', 'PreferenceGroup']],
                on='ID',
                validate='1:1'
            )
        
        # Create group-specific datasets
        data_dict = {
            'full_data': full_data,
            'demographic_data': demographic_processed,
            'psychographic_data': psychographic_processed,
            'segment_data': segment_data
        }
        
        if 'PreferenceGroup' in full_data.columns:
            for group in full_data['PreferenceGroup'].unique():
                data_dict[f'group{group}'] = full_data[
                    full_data['PreferenceGroup'] == group
                ].copy()
        
        logger.info("Data loading and preparation completed successfully")
        return data_dict
    
    def get_data_quality_report(self) -> Dict:
        """Generate data quality report.
        
        Returns:
            Dictionary containing quality metrics
        """
        report = {}
        
        for file_name in self.config.required_files:
            df = pd.read_csv(self.data_dir / file_name)
            
            report[file_name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'duplicates': df.duplicated().sum(),
                'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
            }
            
            # Add column-specific metrics
            report[file_name]['column_metrics'] = {}
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    report[file_name]['column_metrics'][col] = {
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max()
                    }
                else:
                    report[file_name]['column_metrics'][col] = {
                        'unique_values': df[col].nunique(),
                        'most_common': df[col].value_counts().head(3).to_dict()
                    }
        
        return report 