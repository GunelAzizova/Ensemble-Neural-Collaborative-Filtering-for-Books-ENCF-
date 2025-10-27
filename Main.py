import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
book = pd.read_csv('/content/Books.csv')
user = pd.read_csv('/content/Users.csv')
rating = pd.read_csv('/content/Ratings.csv')
print(f"Books: {book.shape}, Users: {user.shape}, Ratings: {rating.shape}")

print("\n" + "="*60)
print("BUILDING POPULARITY-BASED RECOMMENDATION")
print("="*60)

rating_with_name = rating.merge(book, on='ISBN')

num_rating_df = rating_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'Num_rating'}, inplace=True)

avg_rating_df = rating_with_name.groupby('Book-Title').mean(numeric_only=True)['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'Avg_rating'}, inplace=True)

popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')

pbr_df = popular_df[popular_df['Num_rating'] >= 300].sort_values('Avg_rating', ascending=False).head(100)
pbr_df = pbr_df.merge(book, on='Book-Title').drop_duplicates('Book-Title')[
    ['Book-Title', 'Book-Author', 'Publisher', 'Image-URL-M', 'Num_rating', 'Avg_rating']]

print(f"Top 100 popular books created: {pbr_df.shape}")

print("\n" + "="*60)
print("PREPARING DATA FOR COLLABORATIVE FILTERING")
print("="*60)

b = rating_with_name.groupby('User-ID').count()['Book-Rating'] > 250
users_with_ratings = b[b].index
print(f"Users with 250+ ratings: {len(users_with_ratings)}")

filtered_rating = rating_with_name[rating_with_name['User-ID'].isin(users_with_ratings)]

c = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = c[c].index
print(f"Books with 50+ ratings: {len(famous_books)}")

final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
print(f"Final ratings shape: {final_ratings.shape}")

pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)
print(f"Pivot table shape: {pt.shape}")

print("\n" + "="*60)
print("CREATING TRAIN-TEST SPLIT")
print("="*60)

train_data, test_data = train_test_split(final_ratings, test_size=0.2, random_state=42)
print(f"Train size: {train_data.shape}, Test size: {test_data.shape}")

train_pt = train_data.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
train_pt.fillna(0, inplace=True)

print("\n" + "="*60)
print("MODEL 1: COSINE SIMILARITY")
print("="*60)

cosine_sim = cosine_similarity(train_pt)
print(f"Cosine similarity matrix shape: {cosine_sim.shape}")

print("\n" + "="*60)
print("MODEL 2: K-NEAREST NEIGHBORS")
print("="*60)

train_sparse = csr_matrix(train_pt.values)

knn_model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
knn_model.fit(train_sparse)
print("KNN model trained")

print("\n" + "="*60)
print("MODEL 3: MATRIX FACTORIZATION (SVD)")
print("="*60)

svd = TruncatedSVD(n_components=50, n_iter=10, random_state=42)
svd_matrix = svd.fit_transform(train_pt)
print(f"SVD matrix shape: {svd_matrix.shape}")

print("\n" + "="*60)
print("PREPARING DATA FOR NEURAL NETWORKS")
print("="*60)

user_ids = final_ratings['User-ID'].unique()
book_titles = final_ratings['Book-Title'].unique()

user_to_index = {user: idx for idx, user in enumerate(user_ids)}
book_to_index = {book: idx for idx, book in enumerate(book_titles)}
index_to_book = {idx: book for book, idx in book_to_index.items()}

final_ratings['user_idx'] = final_ratings['User-ID'].map(user_to_index)
final_ratings['book_idx'] = final_ratings['Book-Title'].map(book_to_index)

train_data['user_idx'] = train_data['User-ID'].map(user_to_index)
train_data['book_idx'] = train_data['Book-Title'].map(book_to_index)
test_data['user_idx'] = test_data['User-ID'].map(user_to_index)
test_data['book_idx'] = test_data['Book-Title'].map(book_to_index)

train_data = train_data.dropna(subset=['user_idx', 'book_idx'])
test_data = test_data.dropna(subset=['user_idx', 'book_idx'])

X_train_user = train_data['user_idx'].values
X_train_book = train_data['book_idx'].values
y_train = train_data['Book-Rating'].values

X_test_user = test_data['user_idx'].values
X_test_book = test_data['book_idx'].values
y_test = test_data['Book-Rating'].values

num_users = len(user_ids)
num_books = len(book_titles)

print(f"Number of users: {num_users}")
print(f"Number of books: {num_books}")
print(f"Training samples: {len(X_train_user)}")
print(f"Testing samples: {len(X_test_user)}")

print("\n" + "="*60)
print("MODEL 4: NEURAL COLLABORATIVE FILTERING (NCF)")
print("="*60)

user_input = layers.Input(shape=(1,), name='user_input')
book_input = layers.Input(shape=(1,), name='book_input')

user_embedding = layers.Embedding(num_users, 50, name='user_embedding')(user_input)
book_embedding = layers.Embedding(num_books, 50, name='book_embedding')(book_input)

user_vec = layers.Flatten()(user_embedding)
book_vec = layers.Flatten()(book_embedding)

concat = layers.Concatenate()([user_vec, book_vec])
dense1 = layers.Dense(128, activation='relu')(concat)
dropout1 = layers.Dropout(0.3)(dense1)
dense2 = layers.Dense(64, activation='relu')(dropout1)
dropout2 = layers.Dropout(0.3)(dense2)
dense3 = layers.Dense(32, activation='relu')(dropout2)
output = layers.Dense(1, activation='linear')(dense3)

ncf_model = keras.Model(inputs=[user_input, book_input], outputs=output)
ncf_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("NCF Model Architecture:")
ncf_model.summary()

print("\nTraining NCF model...")
history_ncf = ncf_model.fit(
    [X_train_user, X_train_book],
    y_train,
    batch_size=256,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

ncf_predictions = ncf_model.predict([X_test_user, X_test_book])
ncf_mse = mean_squared_error(y_test, ncf_predictions)
ncf_mae = mean_absolute_error(y_test, ncf_predictions)
print(f"\nNCF Test MSE: {ncf_mse:.4f}")
print(f"NCF Test MAE: {ncf_mae:.4f}")

print("\n" + "="*60)
print("MODEL 5: DEEP MATRIX FACTORIZATION")
print("="*60)

user_input_dmf = layers.Input(shape=(1,))
book_input_dmf = layers.Input(shape=(1,))

user_embedding_dmf = layers.Embedding(num_users, 100)(user_input_dmf)
user_vec_dmf = layers.Flatten()(user_embedding_dmf)
user_dense_dmf = layers.Dense(64, activation='relu')(user_vec_dmf)
user_dense_dmf = layers.Dropout(0.3)(user_dense_dmf)

book_embedding_dmf = layers.Embedding(num_books, 100)(book_input_dmf)
book_vec_dmf = layers.Flatten()(book_embedding_dmf)
book_dense_dmf = layers.Dense(64, activation='relu')(book_vec_dmf)
book_dense_dmf = layers.Dropout(0.3)(book_dense_dmf)

dot_product = layers.Dot(axes=1)([user_dense_dmf, book_dense_dmf])
output_dmf = layers.Dense(1, activation='linear')(dot_product)

dmf_model = keras.Model(inputs=[user_input_dmf, book_input_dmf], outputs=output_dmf)
dmf_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("Deep MF Model Architecture:")
dmf_model.summary()

print("\nTraining Deep MF model...")
history_dmf = dmf_model.fit(
    [X_train_user, X_train_book],
    y_train,
    batch_size=256,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

dmf_predictions = dmf_model.predict([X_test_user, X_test_book])
dmf_mse = mean_squared_error(y_test, dmf_predictions)
dmf_mae = mean_absolute_error(y_test, dmf_predictions)
print(f"\nDeep MF Test MSE: {dmf_mse:.4f}")
print(f"Deep MF Test MAE: {dmf_mae:.4f}")

print("\n" + "="*60)
print("MODEL 6: AUTOENCODER")
print("="*60)

input_dim = train_pt.shape[1]
encoding_dim = 128

input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(256, activation='relu')(input_layer)
encoded = layers.Dropout(0.3)(encoded)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

decoded = layers.Dense(256, activation='relu')(encoded)
decoded = layers.Dropout(0.3)(decoded)
decoded = layers.Dense(input_dim, activation='linear')(decoded)

autoencoder = keras.Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

print("Autoencoder Architecture:")
autoencoder.summary()

print("\nTraining Autoencoder...")
history_ae = autoencoder.fit(
    train_pt.values,
    train_pt.values,
    batch_size=32,
    epochs=20,
    validation_split=0.1,
    verbose=1
)

ae_predictions = autoencoder.predict(train_pt.values)
ae_mse = mean_squared_error(train_pt.values.flatten(), ae_predictions.flatten())
print(f"\nAutoencoder MSE: {ae_mse:.4f}")

print("\n" + "="*60)
print("ENSEMBLE MODELS - GRADIENT BOOSTING")
print("="*60)

train_features = train_data[['user_idx', 'book_idx']].values
test_features = test_data[['user_idx', 'book_idx']].values

print("Training Gradient Boosting Regressor...")
gbm_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
gbm_model.fit(train_features, y_train)
gbm_predictions = gbm_model.predict(test_features)
gbm_mse = mean_squared_error(y_test, gbm_predictions)
gbm_mae = mean_absolute_error(y_test, gbm_predictions)
print(f"GBM Test MSE: {gbm_mse:.4f}")
print(f"GBM Test MAE: {gbm_mae:.4f}")

print("\nTraining Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(train_features, y_train)
rf_predictions = rf_model.predict(test_features)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
print(f"RF Test MSE: {rf_mse:.4f}")
print(f"RF Test MAE: {rf_mae:.4f}")

print("\n" + "="*60)
print("META-LEARNING ENSEMBLE")
print("="*60)

all_predictions = np.column_stack([
    ncf_predictions.flatten(),
    dmf_predictions.flatten(),
    gbm_predictions,
    rf_predictions
])

meta_model = Ridge(alpha=1.0)
meta_model.fit(all_predictions, y_test)
meta_predictions = meta_model.predict(all_predictions)
meta_mse = mean_squared_error(y_test, meta_predictions)
meta_mae = mean_absolute_error(y_test, meta_predictions)

print(f"Meta-Learning Ensemble MSE: {meta_mse:.4f}")
print(f"Meta-Learning Ensemble MAE: {meta_mae:.4f}")

print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

results = {
    'NCF': {'MSE': ncf_mse, 'MAE': ncf_mae},
    'Deep MF': {'MSE': dmf_mse, 'MAE': dmf_mae},
    'Autoencoder': {'MSE': ae_mse, 'MAE': 'N/A'},
    'Gradient Boosting': {'MSE': gbm_mse, 'MAE': gbm_mae},
    'Random Forest': {'MSE': rf_mse, 'MAE': rf_mae},
    'Meta-Learning': {'MSE': meta_mse, 'MAE': meta_mae}
}

for model, metrics in results.items():
    print(f"\n{model}:")
    for metric, value in metrics.items():
        if value != 'N/A':
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

print("\n" + "="*60)
print("CREATING RECOMMENDATION FUNCTIONS")
print("="*60)

def recommend_ncf(user_id, n=8):
    if user_id not in user_to_index:
        return []
    
    user_idx = user_to_index[user_id]
    user_array = np.array([user_idx] * num_books)
    book_array = np.arange(num_books)
    
    predictions = ncf_model.predict([user_array, book_array], verbose=0)
    
    top_indices = predictions.flatten().argsort()[::-1][:n]
    
    recommendations = []
    for idx in top_indices:
        book_title = index_to_book[idx]
        temp_df = book[book['Book-Title'] == book_title]
        if not temp_df.empty:
            recommendations.append({
                'title': temp_df['Book-Title'].values[0],
                'author': temp_df['Book-Author'].values[0],
                'image': temp_df['Image-URL-M'].values[0],
                'score': float(predictions[idx][0])
            })
    return recommendations

def recommend_ensemble(user_id, n=8):
    if user_id not in user_to_index:
        return []
    
    user_idx = user_to_index[user_id]
    user_array = np.array([user_idx] * num_books)
    book_array = np.arange(num_books)
    
    ncf_pred = ncf_model.predict([user_array, book_array], verbose=0).flatten()
    dmf_pred = dmf_model.predict([user_array, book_array], verbose=0).flatten()
    
    features = np.column_stack([user_array, book_array])
    gbm_pred = gbm_model.predict(features)
    rf_pred = rf_model.predict(features)
    
    ensemble_pred = (ncf_pred + dmf_pred + gbm_pred + rf_pred) / 4
    
    top_indices = ensemble_pred.argsort()[::-1][:n]
    
    recommendations = []
    for idx in top_indices:
        book_title = index_to_book[idx]
        temp_df = book[book['Book-Title'] == book_title]
        if not temp_df.empty:
            recommendations.append({
                'title': temp_df['Book-Title'].values[0],
                'author': temp_df['Book-Author'].values[0],
                'image': temp_df['Image-URL-M'].values[0],
                'score': float(ensemble_pred[idx])
            })
    return recommendations

def recommend_cosine(book_name, n=8):
    try:
        index = np.where(pt.index == book_name)[0][0]
        similar_items = sorted(list(enumerate(cosine_sim[index])), reverse=True, key=lambda x: x[1])[1:n+1]
        
        recommendations = []
        for i in similar_items:
            temp_df = book[book['Book-Title'] == pt.index[i[0]]]
            if not temp_df.empty:
                recommendations.append({
                    'title': temp_df['Book-Title'].values[0],
                    'author': temp_df['Book-Author'].values[0],
                    'image': temp_df['Image-URL-M'].values[0],
                    'similarity': i[1]
                })
        return recommendations
    except:
        return []

print("\n" + "="*60)
print("TESTING RECOMMENDATIONS")
print("="*60)

test_user = list(user_to_index.keys())[0]
print(f"\nRecommendations for User ID: {test_user}")

print("\n--- Neural Collaborative Filtering ---")
recs_ncf = recommend_ncf(test_user, n=5)
for i, rec in enumerate(recs_ncf, 1):
    print(f"{i}. {rec['title']} by {rec['author']} (score: {rec['score']:.4f})")

print("\n--- Ensemble Model ---")
recs_ensemble = recommend_ensemble(test_user, n=5)
for i, rec in enumerate(recs_ensemble, 1):
    print(f"{i}. {rec['title']} by {rec['author']} (score: {rec['score']:.4f})")

test_book = "1984"
print(f"\n\nSimilar books to: {test_book}")
print("\n--- Cosine Similarity ---")
recs_cos = recommend_cosine(test_book, n=5)
for i, rec in enumerate(recs_cos, 1):
    print(f"{i}. {rec['title']} by {rec['author']} (sim: {rec['similarity']:.4f})")

print("\n" + "="*60)
print("SAVING ALL MODELS")
print("="*60)

pickle.dump(pbr_df, open('PopularBookRecommendation.pkl', 'wb'))
pickle.dump(pt, open('pt.pkl', 'wb'))
pickle.dump(book, open('book.pkl', 'wb'))
pickle.dump(cosine_sim, open('cosine_similarity.pkl', 'wb'))
pickle.dump(knn_model, open('knn_model.pkl', 'wb'))
pickle.dump(svd, open('svd_model.pkl', 'wb'))
pickle.dump(svd_matrix, open('svd_matrix.pkl', 'wb'))
pickle.dump(gbm_model, open('gbm_model.pkl', 'wb'))
pickle.dump(rf_model, open('rf_model.pkl', 'wb'))
pickle.dump(meta_model, open('meta_model.pkl', 'wb'))
pickle.dump(user_to_index, open('user_to_index.pkl', 'wb'))
pickle.dump(book_to_index, open('book_to_index.pkl', 'wb'))
pickle.dump(index_to_book, open('index_to_book.pkl', 'wb'))

ncf_model.save('ncf_model.h5')
dmf_model.save('dmf_model.h5')
autoencoder.save('autoencoder_model.h5')

print("All models saved successfully!")
print("\nFiles created:")
print("- PopularBookRecommendation.pkl")
print("- pt.pkl")
print("- book.pkl")
print("- cosine_similarity.pkl")
print("- knn_model.pkl")
print("- svd_model.pkl")
print("- svd_matrix.pkl")
print("- gbm_model.pkl")
print("- rf_model.pkl")
print("- meta_model.pkl")
print("- user_to_index.pkl")
print("- book_to_index.pkl")
print("- index_to_book.pkl")
print("- ncf_model.h5")
print("- dmf_model.h5")
print("- autoencoder_model.h5")

print("\n" + "="*60)
print("RECOMMENDATION SYSTEM WITH NEURAL NETWORKS COMPLETE!")
print("="*60)
