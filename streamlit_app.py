# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib # For loading models
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer # Needed for MultiLabelBinarizerTransformer
from sklearn.impute import SimpleImputer # Needed if your pipeline uses it and you load the pipeline with it
from sklearn.preprocessing import OneHotEncoder # Needed if your pipeline uses it and you load the pipeline with it
from sklearn.feature_extraction.text import TfidfVectorizer # Needed for text_transformer
from sklearn.decomposition import TruncatedSVD # Needed for text_transformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# --- Custom Transformers Definition (MUST BE IDENTICAL TO TRAINING SCRIPT) ---
# These classes are needed for joblib.load to correctly unpickle the pipeline.

# IMPORTANT: MultiLabelBinarizerTransformer must have the 'top_n' argument
# if it was used during training.
class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, top_n=None): # Add top_n parameter
        self.mlb = MultiLabelBinarizer()
        self.top_n = top_n
        self.keep_classes_ = None # To store the top_n classes

    def fit(self, X, y=None):
        list_of_lists = [
            [e.strip() for e in str(val).split(',') if e.strip() and e != 'Unknown']
            for val in X
        ]
        self.mlb.fit(list_of_lists)

        if self.top_n is not None and self.top_n < len(self.mlb.classes_):
            # For fit, you need a way to determine the 'top' classes.
            # This often involves counting occurrences in the training data.
            # Since we don't have training data here, we'll need to load these
            # top_n classes if they were determined during training and saved.
            # For now, let's assume `mlb.classes_` from `joblib.load`
            # already represents the filtered classes if `top_n` was applied
            # during the original `fit` on the training data.
            # Or, you'd need to save and load `self.keep_classes_` if you
            # implemented a proper top_n selection during training.
            # For a deployment scenario, it's safer to just let the `mlb.classes_`
            # attribute of the loaded MLB be whatever it was during training.
            # If the original MLB was fitted with `top_n`, its `classes_`
            # attribute will already reflect that.
            pass
        self.keep_classes_ = self.mlb.classes_ # Keep all fitted classes from the loaded MLB
        return self

    def transform(self, X):
        list_of_lists = [
            [e.strip() for e in str(val).split(',') if e.strip() and e != 'Unknown']
            for val in X
        ]
        # Transform using the fitted MLB, then filter columns if top_n was applied
        transformed_data = self.mlb.transform(list_of_lists)
        # If mlb.classes_ was pruned during fit (not directly supported by MLB),
        # then the transformed_data will already match the pruned classes.
        # If your original MLB in the pipeline explicitly filtered features,
        # you need to ensure this logic is mirrored or the loaded MLB
        # correctly represents the pruned feature set.
        # For typical MLB usage, `transform` will produce columns for all classes
        # seen during fit.
        return transformed_data

    def get_feature_names_out(self, input_features=None):
        # This should return the classes that the MLB object actually learned to output
        return self.mlb.classes_


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.median_delay = pd.Timedelta(days=365) # Default median delay

    def fit(self, X, y=None):
        # In a real deployment, you'd ideally load the median_delay calculated
        # during training. For simplicity in this app, we'll re-calculate
        # based on the input data *if* it's not None, but it's better
        # to save/load this value from training.
        # For now, we'll rely on the `median_delay` being set when the
        # full pipeline is loaded by joblib.
        return self

    def transform(self, X):
        X_df = X.copy()
        X_df['original_release_date'] = pd.to_datetime(X_df['original_release_date'], errors='coerce')
        X_df['streaming_release_date'] = pd.to_datetime(X_df['streaming_release_date'], errors='coerce')

        # Use the median_delay attribute that should be set during pipeline loading
        # (i.e., when the preprocessor containing this transformer is loaded from joblib).
        # If the loaded pipeline already has a fitted DateFeatureExtractor,
        # `self.median_delay` will be the one from training.

        mask_orig_missing_stream_present = X_df['original_release_date'].isnull() & X_df['streaming_release_date'].notnull()
        X_df.loc[mask_orig_missing_stream_present, 'original_release_date'] = \
            X_df.loc[mask_orig_missing_stream_present, 'streaming_release_date'] - self.median_delay

        mask_stream_missing_orig_present = X_df['streaming_release_date'].isnull() & X_df['original_release_date'].notnull()
        X_df.loc[mask_stream_missing_orig_present, 'streaming_release_date'] = \
            X_df.loc[mask_stream_missing_orig_present, 'original_release_date'] + self.median_delay

        X_df['original_release_year'] = X_df['original_release_date'].dt.year
        X_df['original_release_month'] = X_df['original_release_date'].dt.month
        X_df['original_release_dayofweek'] = X_df['original_release_date'].dt.dayofweek
        X_df['streaming_release_year'] = X_df['streaming_release_date'].dt.year
        X_df['streaming_release_month'] = X_df['streaming_release_date'].dt.month
        X_df['streaming_release_dayofweek'] = X_df['streaming_release_date'].dt.dayofweek

        X_df['release_date_diff_days'] = (X_df['streaming_release_date'] - X_df['original_release_date']).dt.days
        X_df['release_date_diff_days'].fillna(0, inplace=True) # Fill NaNs for the diff before converting to numeric

        # Convert to numeric and fill NaNs based on the specific column's median (from current input batch)
        # In a robust deployment, you'd load the medians from the training set.
        # For demonstration, this imputation is fine for missing values after extraction.
        for col in ['original_release_year', 'original_release_month', 'original_release_dayofweek',
                    'streaming_release_year', 'streaming_release_month', 'streaming_release_dayofweek',
                    'release_date_diff_days']:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
            # Use a default fill_value if the median is NaN (e.g., if a column is all NaNs)
            X_df[col].fillna(X_df[col].median() if X_df[col].median() is not np.nan else 0, inplace=True)


        return X_df[['original_release_year', 'original_release_month', 'original_release_dayofweek',
                     'streaming_release_year', 'streaming_release_month', 'streaming_release_dayofweek',
                     'release_date_diff_days']].values

    def get_feature_names_out(self, input_features=None):
        return ['original_release_year', 'original_release_month', 'original_release_dayofweek',
                'streaming_release_year', 'streaming_release_month', 'streaming_release_dayofweek',
                'release_date_diff_days']

# --- Helper Function for Text Preprocessing ---
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Load Pre-trained Model and Input Columns ---
@st.cache_resource # Cache the loaded model to avoid reloading on every rerun
def load_model():
    try:
        # It's better to save the *entire pipeline* if your preprocessor is part of it.
        # If 'xgboost_regression_model.pkl' is the final regressor *after* preprocessing,
        # you'll need to load the preprocessor separately.
        # Assuming xgboost_regression_model.pkl IS THE FULL PIPELINE (preprocessor + regressor)
        model = joblib.load('xgboost_regression_model.pkl')
        
        # If 'model_input_columns.pkl' explicitly saved the column names
        # *before* the ColumnTransformer, then load that.
        # Otherwise, the pipeline itself (if it's a full pipeline) will handle
        # column selection, but we still need the original feature names
        # for manual input and CSV template.

        # Let's assume 'model_input_columns.pkl' contains the list of *raw* columns
        # that need to be fed into the pipeline.
        input_cols_from_pkl = joblib.load('model_input_columns.pkl')

        st.success("Pre-trained model loaded successfully!")
        return model, input_cols_from_pkl
    except FileNotFoundError:
        st.error("Error: Model files not found. Please ensure 'xgboost_regression_model.pkl' and 'model_input_columns.pkl' are in the same directory as this app.")
        return None, []
    except Exception as e:
        st.error(f"Error loading model: {e}. Ensure custom transformers are defined correctly and files are not corrupted.")
        st.error(f"Detailed error: {e}") # Provide more detail
        return None, []

best_model, all_transformer_input_cols = load_model()

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Movie Rating Predictor")

st.title("🎬 Movie Tomatometer Rating Predictor")
st.markdown("Load a pre-trained model to predict Tomatometer Ratings for new movies.")

if best_model is None:
    st.warning("Model not loaded. Please ensure 'xgboost_regression_model.pkl' and 'model_input_columns.pkl' are in the same directory as this app.")
    st.stop() # Stop execution if model isn't loaded

st.sidebar.header("Model Status")
st.sidebar.success("Model ready for predictions!")

st.markdown("---")

# --- Prediction Interface ---
st.subheader("Predict Tomatometer Rating for a New Movie")

input_method = st.radio("Choose input method:", ("Enter Manually", "Upload CSV for Batch Prediction"))

# Define all expected columns for input, to ensure consistency
EXPECTED_INPUT_COLUMNS = [
    'movie_title', # Not used by model, but good for display/identification
    'movie_info',
    'critics_consensus',
    'genres',
    'directors',
    'actors',
    'authors',
    'production_company',
    'runtime',
    'audience_count',
    'tomatometer_count',
    'audience_rating',
    'tomatometer_top_critics_count',
    'content_rating',
    'original_release_date',
    'streaming_release_date'
]

if input_method == "Enter Manually":
    st.markdown("#### Enter Movie Details:")
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        movie_title = st.text_input("Movie Title", "Example Movie")
        movie_info = st.text_area("Movie Info (Plot Summary)", "A thrilling adventure of a hero saving the world from an alien invasion.")
        critics_consensus = st.text_area("Critics Consensus", "Action-packed and visually stunning, but with a predictable plot.")
        genres = st.text_input("Genres (comma-separated)", "Action, Sci-Fi, Adventure")
        directors = st.text_input("Directors (comma-separated)", "Director X")
        actors = st.text_input("Actors (comma-separated)", "Actor A, Actor B")
    with col_input2:
        authors = st.text_input("Authors (comma-separated)", "Author Y")
        production_company = st.text_input("Production Company", "Big Studio")
        content_rating = st.selectbox("Content Rating", ["G", "PG", "PG-13", "R", "NC-17", "Unrated", "Unknown"], index=5)
        runtime = st.number_input("Runtime (minutes)", min_value=1, value=120)
        audience_count = st.number_input("Audience Count", min_value=0, value=100000)
        tomatometer_count = st.number_input("Tomatometer Count (Number of Critic Reviews)", min_value=0, value=150)
        # CHANGED: audience_rating input to 0-100 scale
        audience_rating = st.number_input("Audience Rating (0-100)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
        tomatometer_top_critics_count = st.number_input("Tomatometer Top Critics Count", min_value=0, value=50)
        original_release_date = st.date_input("Original Release Date", pd.to_datetime("2020-01-01"))
        streaming_release_date = st.date_input("Streaming Release Date", pd.to_datetime("2020-06-01"))

    new_movie_data = pd.DataFrame([{
        'movie_title': movie_title,
        'movie_info': movie_info,
        'critics_consensus': critics_consensus,
        'genres': genres,
        'directors': directors,
        'actors': actors,
        'authors': authors,
        'production_company': production_company,
        'content_rating': content_rating,
        'runtime': float(runtime),
        'audience_count': float(audience_count),
        'tomatometer_count': float(tomatometer_count),
        'audience_rating': float(audience_rating), # This value is now 0-100
        'tomatometer_top_critics_count': float(tomatometer_top_critics_count),
        'original_release_date': original_release_date,
        'streaming_release_date': streaming_release_date
    }])

    # Preprocess combined_text_description for the new input
    new_movie_data['combined_text_description'] = new_movie_data['movie_info'].fillna('') + ' ' + new_movie_data['critics_consensus'].fillna('')
    new_movie_data['combined_text_description'] = new_movie_data['combined_text_description'].apply(preprocess_text)

    # Filter new_movie_data to match the columns expected by the preprocessor
    missing_cols = [col for col in all_transformer_input_cols if col not in new_movie_data.columns]
    if missing_cols:
        st.error(f"Error: The input data is missing expected columns: {missing_cols}. Please ensure your manual entry includes all necessary fields.")
        st.stop()

    final_input_for_model = new_movie_data[all_transformer_input_cols].copy()


    if st.button("Predict Rating"):
        try:
            prediction = best_model.predict(final_input_for_model)
            st.success(f"Predicted Tomatometer Rating: **{prediction[0]:.2f}**")
            st.info("Note: Ratings are typically on a scale of 0-100.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning(f"Please ensure all input fields are correctly filled and match the training data format. Error details: {e}")
            st.write("Debug: Input DataFrame columns:", final_input_for_model.columns.tolist())
            st.write("Debug: Input DataFrame head:", final_input_for_model.head())

else: # Upload CSV for Batch Prediction
    st.markdown("#### Upload a CSV file for batch prediction:")
    st.info(f"Your CSV must contain the following columns: {', '.join(col for col in EXPECTED_INPUT_COLUMNS if col != 'movie_title')} and 'movie_title'. Ensure 'audience_rating' is on a 0-100 scale.") # Added info about audience_rating scale
    uploaded_batch_file = st.file_uploader("Upload CSV for prediction", type="csv")

    if uploaded_batch_file is not None:
        batch_df = pd.read_csv(uploaded_batch_file)
        st.write("First 5 rows of your batch data:")
        st.dataframe(batch_df.head())

        # Ensure 'combined_text_description' is present
        batch_df['combined_text_description'] = batch_df['movie_info'].fillna('') + ' ' + batch_df['critics_consensus'].fillna('')
        batch_df['combined_text_description'] = batch_df['combined_text_description'].apply(preprocess_text)

        # Ensure all expected columns are present in the uploaded batch_df
        missing_cols_batch = [col for col in all_transformer_input_cols if col not in batch_df.columns]
        if missing_cols_batch:
            st.error(f"Error: The uploaded batch data is missing expected columns: {missing_cols_batch}. Please ensure your CSV includes all necessary fields.")
            st.stop()
        
        # Filter batch_df to match the columns expected by the preprocessor
        final_input_for_batch_model = batch_df[all_transformer_input_cols].copy()

        if st.button("Run Batch Prediction"):
            try:
                with st.spinner("Running predictions..."):
                    batch_predictions = best_model.predict(final_input_for_batch_model)
                
                batch_df['predicted_tomatometer_rating'] = batch_predictions
                st.success("Batch predictions complete!")
                st.write("Predictions added to your data:")
                st.dataframe(batch_df[['movie_title', 'predicted_tomatometer_rating']].head())

                csv_output = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv_output,
                    file_name="movie_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"An error occurred during batch prediction: {e}")
                st.warning(f"Please ensure your uploaded CSV has all the necessary columns for prediction and matches the training data format. Error details: {e}")
                st.write("Debug: Batch Input DataFrame columns:", final_input_for_batch_model.columns.tolist())
                st.write("Debug: Batch Input DataFrame head:", final_input_for_batch_model.head())