import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC



st.set_page_config(page_title="🌾 Crop Recommendation System", layout="wide")
st.title("🌱 Intelligent Crop Recommendation System")

st.markdown("""
This app helps predict the **best crop to grow** based on soil and weather conditions  
using machine learning models trained on the `Crop_recommendation.csv` dataset.
""")



df = pd.read_csv("Crop_recommendation.csv")
st.success("✅ Dataset Loaded Successfully!")

if st.checkbox("Show Dataset Preview"):
    st.dataframe(df.head())

st.write("### Dataset Information")
st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")



le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

if st.checkbox("Show Label Distribution"):
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(y='label', data=df, order=df['label'].value_counts().index)
    plt.title("Crop Distribution")
    st.pyplot(fig)


X = df.drop(['label', 'label_encoded'], axis=1)
y = df['label_encoded']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)



st.header("🤖 Model Training and Accuracy Comparison")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(kernel='rbf', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc


result_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
st.dataframe(result_df)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x='Model', y='Accuracy', data=result_df)
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
st.pyplot(fig)

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
st.success(f"🏆 Best Model: **{best_model_name}** (Accuracy: {results[best_model_name]:.2f})")


st.header("🌿 Crop Prediction Options")
option = st.radio("Choose a prediction method:", ("🔢 Manual Input", "📂 Upload CSV File"))


if option == "🔢 Manual Input":
    st.subheader("Enter Agro-Climatic Parameters")
    
    N = st.number_input("Nitrogen (N)", 0, 200, 90)
    P = st.number_input("Phosphorus (P)", 0, 200, 42)
    K = st.number_input("Potassium (K)", 0, 200, 43)
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
    ph = st.number_input("pH Value", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 400.0, 200.0)

    if st.button("🌾 Predict Crop"):
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)
        prediction = best_model.predict(input_scaled)
        crop_name = le.inverse_transform(prediction)[0]
        st.success(f"🌱 The recommended crop to grow is: **{crop_name.upper()}**")


elif option == "📂 Upload CSV File":
    uploaded_file = st.file_uploader("Upload your test CSV file", type=['csv'])
    
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
        st.write("✅ Test File Uploaded Successfully!")
        st.dataframe(test_df.head())

        try:
            test_scaled = scaler.transform(test_df)
            preds = best_model.predict(test_scaled)
            test_df['Predicted_Crop'] = le.inverse_transform(preds)
            
            st.success("✅ Predictions Completed!")
            st.dataframe(test_df.head())

            
            csv_download = test_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", data=csv_download, file_name="crop_predictions.csv", mime='text/csv')

        except Exception as e:
            st.error("⚠️ Error: Ensure your CSV has the same columns as the training dataset.")


st.markdown("---")
st.caption("🚀 Built by killer  | Machine Learning for Smart Agriculture 🌾")
