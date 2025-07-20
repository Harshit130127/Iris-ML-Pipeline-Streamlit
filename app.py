import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# load the model
model = pickle.load(open("iris_model.pkl", "rb"))
scaler = pickle.load(open("iris_scaler.pkl", "rb"))
slider_df = pd.read_csv("iris_slider_data.csv")
target_names = ["setosa", "versicolor", "virginica"]

features = ["sepal_length","sepal_width","petal_length","petal_width","petal_area"]  # Define the features to be used in the model
labels = {
    "sepal_length":"Sepal Length (cm)",
    "sepal_width":"Sepal Width (cm)",
    "petal_length":"Petal Length (cm)",
    "petal_width":"Petal Width (cm)",
    "petal_area":"Petal Area (cm²)"
}  # Define the labels for the features

st.title("Iris Flower Classification")
st.markdown("Adjust the sliders to enter measurements, then click **Predict**.")   # Display the title and instructions

# Input sliders with units
inputs = {}
for f in features:   # Loop through each feature
    st.markdown(f"**{labels[f]}**")
    inputs[f] = st.slider(
        "", float(slider_df[f].min()), float(slider_df[f].max()), float(slider_df[f].mean()), key=f
    )

if st.button("Predict"):   
    # Scale & predict
    x = np.array([[inputs[f] for f in features]])
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]
    probs = model.predict_proba(x_scaled)[0]

    # Display prediction
    st.subheader(f"**Predicted Species:** {target_names[pred].title()}")

    # Probability chart
    prob_df = pd.Series(probs, index=target_names, name="Probability")
    st.bar_chart(prob_df)

    # Feature importances
    fi = model.feature_importances_
    df_fi = pd.DataFrame({"Feature":features, "Importance":fi}).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(6,3))
    sns.barplot(x="Importance", y="Feature", data=df_fi, ax=ax, palette="viridis")
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Annotate bars
    for idx, row in df_fi.iterrows():
        ax.text(row.Importance + 0.01, idx, f"{row.Importance:.1f}", va='center')

    # Highlight & label top feature
    top = df_fi.iloc[-1]
    ax.patches[-1].set_edgecolor('red')
    ax.patches[-1].set_linewidth(2)
    ax.text(
        top.Importance, len(df_fi)-1 + 0.1,
        f"Most important:\n{top.Feature.replace('_',' ').title()}",
        color="red", fontsize=10, va='bottom'
    )

    ax.set_title("Feature Importances")
    plt.tight_layout()
    st.pyplot(fig)
