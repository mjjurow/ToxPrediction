import json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import joblib
import pickle
from flask import escape
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from google.cloud import logging
from google.cloud import storage

# Initialize Cloud Logging
client = logging.Client()
logger = client.logger("tox-prediction-function")

# Load models and feature lists when the Lambda function starts
model_filenames = {
    'NR.AhR': 'NR.AhR_svm_model.pkl',
    'NR.AR': 'NR.AR_svm_model.pkl',
    'NR.AR.LBD': 'NR.AR.LBD_svm_model.pkl',
    'NR.Aromatase': 'NR.Aromatase_svm_model.pkl',
    'NR.ER': 'NR.ER_svm_model.pkl',
    'NR.ER.LBD': 'NR.ER.LBD_svm_model.pkl',
    'NR.PPAR.gamma': 'NR.PPAR.gamma_svm_model.pkl',
    'SR.ARE': 'SR.ARE_svm_model.pkl',
    'SR.ATAD5': 'SR.ATAD5_svm_model.pkl',
    'SR.HSE': 'SR.HSE_svm_model.pkl',
    'SR.MMP': 'SR.MMP_svm_model.pkl',
    'SR.p53': 'SR.p53_svm_model.pkl'
}

def load_models_and_features():
    loaded_models = {}
    client = storage.Client()
    bucket = client.bucket('toxicityprediction')

    for endpoint, filename in model_filenames.items():
        blob = bucket.blob(f'models/{filename}')  # Adjust the path if you used a directory within the bucket
        temp_model_path = f'/tmp/{filename}'  # Save to /tmp directory
        blob.download_to_filename(temp_model_path)  # Downloads the file to the writable /tmp filesystem
        try:
            model = joblib.load(temp_model_path)
            loaded_models[endpoint] = model
            logger.log_text(f"Model loaded successfully for endpoint: {endpoint}", severity='INFO')
        except Exception as e:
            logger.log_text(f"Failed to load model for endpoint {endpoint}: {str(e)}", severity='ERROR')
            raise e

    # Similar adjustment for feature_lists if it's also being downloaded
    try:
        feature_lists_blob = bucket.blob('feature_lists.json')
        temp_feature_lists_path = '/tmp/feature_lists.json'
        feature_lists_blob.download_to_filename(temp_feature_lists_path)
        with open(temp_feature_lists_path, 'r') as file:
            loaded_feature_lists = json.load(file)
        logger.log_text("Feature lists loaded successfully", severity='INFO')
    except Exception as e:
        logger.log_text(f"Failed to load feature lists: {str(e)}", severity='ERROR')
        raise e

    return loaded_models, loaded_feature_lists

def getMolDescriptorsFromSmiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    res = {}
    for nm, fn in Descriptors._descList:
        try:
            res[nm] = fn(mol)
        except Exception as e:
            logger.log_text(f"Error calculating descriptor {nm} for SMILES {smiles}: {str(e)}", severity='ERROR')
            res[nm] = None
    return res

def get_descriptors_for_endpoints(smiles, loaded_feature_lists):
    all_descriptors = getMolDescriptorsFromSmiles(smiles)
    if all_descriptors is None:
        logger.log_text(f"Could not generate molecule from SMILES: {smiles}", severity='WARNING')
        return None

    descriptors_for_endpoints = {}
    for endpoint, features in loaded_feature_lists.items():
        descriptors_for_endpoints[endpoint] = {feature: all_descriptors[feature] for feature in features if feature in all_descriptors}
    return descriptors_for_endpoints

def predict_endpoints(smiles, loaded_feature_lists, loaded_models):
    descriptors_by_endpoint = get_descriptors_for_endpoints(smiles, loaded_feature_lists)
    if descriptors_by_endpoint is None:
        return pd.DataFrame(columns=['Endpoint', 'Prediction'])

    predictions = []
    for endpoint, descriptors in descriptors_by_endpoint.items():
        model = loaded_models.get(endpoint)
        if model:
            descriptor_values = [descriptors[feature] for feature in loaded_feature_lists[endpoint] if feature in descriptors]
            descriptor_values_2d = np.array(descriptor_values).reshape(1, -1)
            prediction = model.predict(descriptor_values_2d)
            predictions.append((endpoint, prediction[0]))

    return pd.DataFrame(predictions, columns=['Endpoint', 'Prediction'])

def tox_prediction(request):
    # Ensure models and features are loaded
    loaded_models, loaded_feature_lists = load_models_and_features()

    # Preflight request for CORS
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    # Main request handling
    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    request_json = request.get_json(silent=True)
    input_SMILES_string = request_json.get('smiles') if request_json else None

    if not input_SMILES_string:
        return ('No SMILES string provided in the input', 400, headers)
    
    try:
        endpoint_predictions = predict_endpoints(input_SMILES_string, loaded_feature_lists, loaded_models)
        predictions_json = endpoint_predictions.to_json(orient='records')
        return (predictions_json, 200, headers)
    except Exception as e:
        logger.log_text(f"Error processing request: {str(e)}", severity='ERROR')
        return (json.dumps({'error': f'Error processing request: {str(e)}'}), 500, headers)
