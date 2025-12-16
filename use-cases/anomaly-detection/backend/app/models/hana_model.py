"""
SAP HANA ML model implementation for pharmaceutical anomaly detection.

This module implements the Isolation Forest using SAP HANA ML backend.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Any, List
from models.base import AnomalyDetectionModel
from config.settings import (
    N_ESTIMATORS, MAX_SAMPLES, RANDOM_STATE, CONTAMINATION_RATE, HANA_AVAILABLE
)

if HANA_AVAILABLE:
    from hana_ml import ConnectionContext
    from hana_ml.dataframe import create_dataframe_from_pandas
    from hana_ml.algorithms.pal.preprocessing import IsolationForest as HanaIsolationForest


class HanaAnomalyModel(AnomalyDetectionModel):
    """SAP HANA ML Isolation Forest implementation."""
    
    def train(self, X_train: pd.DataFrame, train_data: pd.DataFrame = None) -> Tuple[Any, Any, Any, List[str]]:
        """
        Train Isolation Forest using SAP HANA ML.
        
        Args:
            X_train: Training features
            train_data: Full training dataset (optional)
            
        Returns:
            tuple: (model, connection_context, hdf_train, feature_names)
        """
        print(f"\nTraining Isolation Forest with HANA ML...")
        
        # Setup HANA connection
        try:
            hana_address = os.getenv('hana_address', "<your_hana_address>")
            hana_port = int(os.getenv('hana_port', 443))
            hana_user = os.getenv('hana_user', "<your_hana_user>")
            hana_password = os.getenv('hana_password', "<your_hana_password>")
            hana_encrypt = os.getenv('hana_encrypt', 'true').lower() == 'true'
            hana_schema = os.getenv('HANA_SCHEMA', 'AICOE')
            
            cc = ConnectionContext(
                address=hana_address,
                port=hana_port,
                user=hana_user,
                password=hana_password,
                encrypt=hana_encrypt,
                current_schema=hana_schema
            )
            print(f"HANA Connection successful: {cc.hana_version()}")
            
        except Exception as e:
            print(f"HANA connection failed: {e}")
            raise Exception("Cannot proceed with HANA ML without valid connection")
        
        # Prepare data for HANA
        hana_data = X_train.copy()
        hana_data.insert(0, 'ID', range(len(hana_data)))  # Add ID column required by HANA PAL
        
        # Upload to HANA
        table_name = "PHARMA_ANOMALY_TRAIN"
        try:
            cc.drop_table(table_name)
        except:
            pass
        
        hdf_train = create_dataframe_from_pandas(
            connection_context=cc,
            pandas_df=hana_data,
            table_name=table_name,
            force=True,
            replace=True,
            primary_key='ID'
        )
        
        # Configure and train Isolation Forest
        iforest = HanaIsolationForest(
            n_estimators=N_ESTIMATORS,
            max_samples=MAX_SAMPLES,
            random_state=RANDOM_STATE,
            # thread_ratio=-1  # Let HANA decide
        )
        
        feature_names = [col for col in hana_data.columns if col != 'ID']
        iforest.fit(data=hdf_train, key='ID', features=feature_names)
        
        print(f"Model training completed with {N_ESTIMATORS} trees")
        
        return iforest, cc, hdf_train, feature_names
    
    def predict(self, model_data: Tuple[Any, Any, Any, List[str]], X_test: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions using HANA ML model.
        
        Args:
            model_data: Tuple containing (model, connection_context, hdf_train, feature_names)
            X_test: Test features
            
        Returns:
            tuple: (anomaly_scores, anomaly_labels)
        """
        model, cc, hdf_train, feature_names = model_data
        test_data = kwargs.get('test_data')
        
        print(f"\nGenerating predictions with HANA ML...")
        
        try:
            # Prepare test data for HANA
            hana_test_data = X_test.copy()
            hana_test_data.insert(0, 'ID', range(len(hana_test_data)))  # Add ID column
            
            # Upload test data to HANA
            test_table_name = "PHARMA_ANOMALY_TEST"
            try:
                cc.drop_table(test_table_name)
            except:
                pass
            
            hdf_test = create_dataframe_from_pandas(
                connection_context=cc,
                pandas_df=hana_test_data,
                table_name=test_table_name,
                force=True,
                replace=True,
                primary_key='ID'
            )
            
            # Generate predictions using the trained model
            if CONTAMINATION_RATE == 'auto':
                # HANA ML doesn't have 'auto' mode, so we'll use anomaly scores without contamination
                print(f"Predicting anomalies using HANA ML auto mode (no contamination parameter)...")
                
                results_hdf = model.predict(
                    data=hdf_test,
                    key='ID',
                    features=feature_names
                    # No contamination parameter - HANA will return raw scores
                )
            else:
                print(f"Predicting anomalies using contamination = {CONTAMINATION_RATE:.4f}...")
                
                results_hdf = model.predict(
                    data=hdf_test,
                    key='ID',
                    features=feature_names,
                    contamination=float(CONTAMINATION_RATE)
                )
            
            print("HANA ML prediction completed.")
            
            # Collect results into pandas DataFrame
            if results_hdf:
                df_results = results_hdf.collect()
                print(f"Retrieved {len(df_results)} prediction results")
                
                # Extract anomaly scores and labels
                # HANA ML Isolation Forest scoring:
                # SCORE: 0-1 range, higher scores = more anomalous
                # LABEL: -1 = anomaly, 1 = normal
                anomaly_scores = df_results['SCORE'].values
                
                if CONTAMINATION_RATE == 'auto':
                    # For auto mode in HANA, we use a threshold approach
                    # Since HANA scores are 0-1 with higher = more anomalous,
                    # we use 0.5 as threshold (similar to scikit-learn's 'auto' mode logic)
                    print("HANA ML auto mode: using 0.5 as anomaly threshold")
                    auto_threshold = 0.5
                    anomaly_labels = (anomaly_scores >= auto_threshold).astype(int)
                    actual_contamination_rate = anomaly_labels.mean()
                    
                    print(f"Score range: [{anomaly_scores.min():.3f}, {anomaly_scores.max():.3f}]")
                    print(f"Auto-detected contamination rate: {actual_contamination_rate:.2%}")
                    print(f"Anomaly threshold: score ≥ {auto_threshold:.3f}")
                    print(f"Detected {anomaly_labels.sum()} anomalies out of {len(X_test)} samples")
                    
                else:
                    # Use HANA's provided labels when contamination rate is specified
                    # HANA returns LABEL column: -1 = anomaly, 1 = normal
                    if 'LABEL' in df_results.columns:
                        anomaly_labels = (df_results['LABEL'] == -1).astype(int)  # Convert to 0/1
                    else:
                        # Fallback: use score-based thresholding
                        threshold = np.percentile(anomaly_scores, (1 - float(CONTAMINATION_RATE)) * 100)
                        anomaly_labels = (anomaly_scores >= threshold).astype(int)
                    
                    print(f"Score range: [{anomaly_scores.min():.3f}, {anomaly_scores.max():.3f}]")
                    print(f"Detected {anomaly_labels.sum()} anomalies out of {len(X_test)} samples ({anomaly_labels.mean():.2%})")
                    if 'LABEL' in df_results.columns:
                        print(f"Using HANA ML labels (LABEL column)")
                    else:
                        print(f"Anomaly threshold: score ≥ {np.percentile(anomaly_scores, (1-float(CONTAMINATION_RATE))*100):.3f}")
                
                # Cleanup test table
                cc.drop_table(test_table_name)
                
                return anomaly_scores, anomaly_labels
            else:
                raise Exception("HANA ML prediction returned no results")
                
        except Exception as e:
            print(f"Error during HANA ML prediction: {e}")
            # Cleanup on error
            try:
                cc.drop_table(test_table_name)
            except:
                pass
            raise
    
    def get_model_type(self) -> str:
        """Get the model type identifier."""
        return "hana"