
from flask import Flask, jsonify, request
import pandas as pd
import json
import os
from canopy_data import load_canopy_data, cluster_cities, get_base_recommendations
from health_data import (
    load_tree_data, train_species_classifier, evaluate_species_classifier,
    train_phenolics_regressor, evaluate_phenolics_regressor,
    load_borough_tree_data, train_age_group_classifier, evaluate_age_group_classifier,
    train_xgboost_age_group_classifier, evaluate_xgboost_age_group_classifier,
    get_shap_explanation, create_semantic_search_model, query_tree_info
)

app = Flask(__name__)

# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
TREE_DATA_CSV = os.path.join(DATA_DIR, 'Tree_Data.csv')
BOROUGH_DATA_CSV = os.path.join(DATA_DIR, 'Borough_tree_list_2021.csv')
SHAPEFILE_PATH = 'CityBoundaries.shp' # Replace with your actual path if needed
TIF_PATH = 'tcc_1km_resolution.tif'   # Replace with your actual path if needed

# --- Data Loading and Model Training (Run on startup) ---
canopy_stats = None
cluster_info = None
tree_df, tree_label_encoders = load_tree_data(TREE_DATA_CSV)
species_model, species_test_X, species_test_y = train_species_classifier(tree_df)
phenolics_model, phenolics_test_X, phenolics_test_y = train_phenolics_regressor(tree_df)
borough_df, borough_label_encoders, age_group_label_encoder = load_borough_tree_data(BOROUGH_DATA_CSV)
age_group_rf_model, age_group_test_X, age_group_test_y = train_age_group_classifier(
    borough_df, ['borough', 'tree_name', 'spread_m', 'height_m', 'diameter_at_breast_height_cm'], 'age_group'
)
age_group_xgb_model, age_group_xgb_test_X, age_group_xgb_test_y = train_xgboost_age_group_classifier(
    borough_df, ['borough', 'tree_name', 'spread_m', 'height_m', 'diameter_at_breast_height_cm'], 'age_group'
)
semantic_model = create_semantic_search_model()

# Load canopy data and cluster cities on startup (if shapefile and tif are available)
if os.path.exists(SHAPEFILE_PATH) and os.path.exists(TIF_PATH):
    canopy_data = load_canopy_data(SHAPEFILE_PATH, TIF_PATH)
    if canopy_data is not None:
        canopy_stats, cluster_info = cluster_cities(canopy_data.copy())
    else:
        print("Warning: Could not load or process canopy data.")
else:
    print("Warning: Shapefile or TIF file for canopy analysis not found.")
    canopy_stats = None
    cluster_info = None

# --- API Endpoints ---

@app.route('/api/canopy/stats', methods=['GET'])
def get_canopy_statistics():
    """Retrieves tree canopy statistics."""
    if canopy_stats is not None:
        return jsonify(json.loads(canopy_stats.to_json(default_handler=str)))
    return jsonify({"error": "Canopy statistics not available."}), 404

@app.route('/api/canopy/clusters', methods=['GET'])
def get_canopy_clusters():
    """Retrieves city clusters based on tree canopy characteristics."""
    if cluster_info is not None:
        return jsonify(cluster_info)
    return jsonify({"error": "Cluster information not available."}), 404

@app.route('/api/policy/clusters', methods=['GET'])
def get_cluster_recommendations():
    """Retrieves policy recommendations for each city cluster."""
    if cluster_info is not None and canopy_stats is not None:
        all_recommendations = {}
        for cluster_num, cluster_label in enumerate(cluster_info['cluster_labels']):
            cluster_cities = canopy_stats[canopy_stats['cluster'] == cluster_num]
            if not cluster_cities.empty:
                cluster_avg = {
                    'mean_tcc': cluster_cities['mean_tcc'].mean(),
                    'area_km2': cluster_cities['area_km2'].mean(),
                    'cluster_type': cluster_label
                }
                base_recs = get_base_recommendations(cluster_avg, cluster_label, is_cluster=True)
                all_recommendations[cluster_label] = base_recs
        return jsonify(all_recommendations)
    return jsonify({"error": "Canopy data or cluster information not available."}), 404

@app.route('/api/policy/cities/<city_name>', methods=['GET'])
def get_city_recommendations(city_name):
    """Retrieves policy recommendations for a specific city."""
    if canopy_stats is not None:
        city_data = canopy_stats[canopy_stats['city'].str.lower() == city_name.lower()]
        if not city_data.empty:
            city_data = city_data.iloc[0]
            cluster = city_data['cluster']
            cluster_type = cluster_info['cluster_labels'][cluster] if cluster_info else "Unknown"
            base_recommendations = get_base_recommendations(city_data, cluster_type)
            return jsonify(base_recommendations)
        return jsonify({"error": f"City '{city_name}' not found."}), 404
    return jsonify({"error": "Canopy data not available."}), 404

@app.route('/api/data/species/distribution', methods=['GET'])
def get_species_distribution():
    """Retrieves the distribution of tree species."""
    if tree_df is not None:
        species_counts = tree_df['Species'].value_counts().to_dict()
        return jsonify({'species_counts': species_counts})
    return jsonify({"error": "Tree data not available."}), 404

@app.route('/api/ml/species_prediction/report', methods=['GET'])
def get_species_prediction_report():
    """Retrieves the classification report for species prediction."""
    if species_model is not None and species_test_X is not None and species_test_y is not None:
        report, _ = evaluate_species_classifier(species_model, species_test_X, species_test_y)
        return jsonify(report)
    return jsonify({"error": "Species prediction model not available."}), 404

@app.route('/api/ml/species_prediction/feature_importance', methods=['GET'])
def get_species_feature_importance():
    """Retrieves feature importances for species prediction."""
    if species_model is not None and species_test_X is not None and species_test_y is not None:
        _, feature_importances = evaluate_species_classifier(species_model, species_test_X, species_test_y)
        return jsonify(json.loads(feature_importances.to_json()))
    return jsonify({"error": "Species prediction model not available."}), 404

@app.route('/api/ml/phenolics_prediction/metrics', methods=['GET'])
def get_phenolics_prediction_metrics():
    """Retrieves regression metrics for Phenolics prediction."""
    if phenolics_model is not None and phenolics_test_X is not None and phenolics_test_y is not None:
        metrics, _ = evaluate_phenolics_regressor(phenolics_model, phenolics_test_X, phenolics_test_y)
        return jsonify(metrics)
    return jsonify({"error": "Phenolics prediction model not available."}), 404

@app.route('/api/ml/phenolics_prediction/feature_importance', methods=['GET'])
def get_phenolics_feature_importance():
    """Retrieves feature importances for Phenolics prediction."""
    if phenolics_model is not None and phenolics_test_X is not None and phenolics_test_y is not None:
        _, feature_importances = evaluate_phenolics_regressor(phenolics_model, phenolics_test_X, phenolics_test_y)
        return jsonify(json.loads(feature_importances.to_json()))
    return jsonify({"error": "Phenolics prediction model not available."}), 404

@app.route('/api/ml/age_group_prediction/report', methods=['GET'])
def get_age_group_prediction_report():
    """Retrieves the classification report for age group prediction."""
    if age_group_rf_model is not None and age_group_test_X is not None and age_group_test_y is not None and age_group_label_encoder is not None:
        report, _ = evaluate_age_group_classifier(age_group_rf_model, age_group_test_X, age_group_test_y, age_group_label_encoder)
        return jsonify(report)
    return jsonify({"error": "Age group prediction model not available."}), 404

@app.route('/api/ml/age_group_prediction/confusion_matrix', methods=['GET'])
def get_age_group_confusion_matrix():
    """Retrieves the confusion matrix for age group prediction."""
    if age_group_rf_model is not None and age_group_test_X is not None and age_group_test_y is not None and age_group_label_encoder is not None:
        _, confusion = evaluate_age_group_classifier(age_group_rf_model, age_group_test_X, age_group_test_y, age_group_label_encoder)
        # Convert NumPy array to list for JSON serialization
        confusion_list = [row.tolist() for row in confusion]
        return jsonify({'confusion_matrix': confusion_list, 'labels': list(age_group_label_encoder.classes_)})
    return jsonify({"error": "Age group prediction model not available."}), 404

@app.route('/api/ml/age_group_prediction/xgboost_report', methods=['GET'])
def get_age_group_xgboost_report():
    """Retrieves the classification report for XGBoost age group prediction."""
    if age_group_xgb_model is not None and age_group_xgb_test_X is not None and age_group_label_encoder is not None:
        report = evaluate_xgboost_age_group_classifier(age_group_xgb_model, age_group_xgb_test_X, age_group_xgb_test_y, age_group_label_encoder)
        return jsonify(report)
    return jsonify({"error": "XGBoost age group prediction model not available."}), 404

@app.route('/api/ml/shap/species', methods=['GET'])
def get_species_shap_values():
    """Retrieves SHAP values for species prediction."""
    if species_model is not None and species_test_X is not None:
        shap_values = get_shap_explanation(species_model, species_test_X)
        if shap_values is not None:
            return jsonify({'shap_values': shap_values.tolist(), 'feature_names': species_test_X.columns.tolist()})
        else:
            return jsonify({"error": "Could not generate SHAP values."}), 500
    return jsonify({"error": "Species prediction model not available."}), 404

@app.route('/api/ml/shap/age_group', methods=['GET'])
def get_age_group_shap_values():
    """Retrieves SHAP values for age group prediction (XGBoost)."""
    if age_group_xgb_model is not None and age_group_xgb_test_X is not None:
        shap_values = get_shap_explanation(age_group_xgb_model, age_group_xgb_test_X)
        if shap_values is not None:
            return jsonify({'shap_values': shap_values.tolist(), 'feature_names': age_group_xgb_test_X.columns.tolist()})
        else:
            return jsonify({"error": "Could not generate SHAP values."}), 500
    return jsonify({"error": "Age group prediction model not available."}), 404

@app.route('/api/search/trees', methods=['POST'])
def search_tree_info():
    """Performs a semantic search on tree information."""
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No query provided."}), 400

    if semantic_model is not None and tree_df is not None and borough_df is not None:
        results = query_tree_info(semantic_model, tree_df, borough_df, query)
        return jsonify({'results': results})
    return jsonify({"error": "Search model or data not available."}), 500

if __name__ == '__main__':
    app.run(debug=True)