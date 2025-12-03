import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Crime Clustering Dashboard",
    layout="wide"
)

st.title("📊 Crime Clustering Dashboard")



@st.cache_resource
def load_artifacts():

    return {
        # SCALERS
        "geo_scaler": joblib.load("geo_scaler.pkl"),
        "time_scaler": joblib.load("time_scaler.pkl"),
        "crime_scaler": joblib.load("crime_scaler.pkl"),

        # MODELS
        "kmeans_geo": joblib.load("kmeans_geo_model.joblib"),
        "kmeans_time": joblib.load("kmeans_time_model.joblib"),
        "kmeans_cyclical": joblib.load("kmeans_time_cyclical_model.joblib"),
        'kmeans_crime': joblib.load("kmeans_crime_model.joblib"),

        # DATA
        "geo_df": pd.read_pickle("geo_df.pkl"),   
        "time_df": joblib.load("time_df.pkl"),    
        "new_crime_df": joblib.load("new_crime_df.pkl"),
        "time_cyclical_df": joblib.load("time_cyclical_data.pkl")
    }

art = load_artifacts()



with st.sidebar:
    st.header("🔍 Select Model")
    selected_model = st.selectbox(
        "Choose a clustering model:",
        ["Geo Based Clustering", "Time Based Clustering"]
    )

    st.markdown("---")
    show_silhouette = st.checkbox("Compute Silhouette Score", value=True)
    sample_frac = st.slider("Sampling for Visualization", 0.05, 1.0, 0.2, step=0.05)
    show_centroids = st.checkbox("Show Centroids (if available)", value=True)



# Geo Based Clustering
if selected_model == "Geo Based Clustering":

    st.subheader("📍 Geo-Based Clustering Results")

    df = art["geo_df"].copy()
    model = art["kmeans_geo"]
    scaler = art["geo_scaler"]
    feature_cols = ['Latitude', 'Longitude', 'Beat', 'District', 'Ward', 'Community Area']

    X = df[feature_cols].dropna().reset_index(drop=True)
    X_scaled = scaler.transform(X)

    labels = model.predict(X_scaled)
    df["cluster"] = labels

    n_clusters = len(np.unique(labels))

    # Silhouette
    sil_text = "Not Computed"
    if show_silhouette and n_clusters > 1:
        sample_size = min(len(X_scaled), 20000)
        idx = np.random.choice(len(X_scaled), size=sample_size, replace=False)
        sil_text = f"{silhouette_score(X_scaled[idx], labels[idx]):.4f}"

    col1, col2, col3 = st.columns(3)
    col1.metric("Data Points", len(X))
    col2.metric("Clusters", n_clusters)
    col3.metric("Silhouette Score", sil_text)

    sampled_df = df.sample(frac=sample_frac, random_state=42)

    # Scatter plot
    fig = px.scatter(
        sampled_df,
        x="Longitude", y="Latitude",
        color=sampled_df["cluster"].astype(str),
        title="Geo Cluster Visualization"
    )
    
    # Add centroids
    if show_centroids:
        centroids = scaler.inverse_transform(model.cluster_centers_)
        fig.add_scatter(
            x=centroids[:,1], y=centroids[:,0],
            mode="markers+text",
            text=[f"C{i}" for i in range(len(centroids))],
            marker=dict(size=15, symbol="x")
        )
        
    st.plotly_chart(fig, use_container_width=True)

    # PCA
    st.subheader("PCA View")

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])
    pca_df["cluster"] = labels  

    # Sample after adding cluster
    sampled_pca_df = pca_df.sample(frac=sample_frac, random_state=42)

    fig_pca = px.scatter(
        sampled_pca_df,
        x="PC1",
        y="PC2",
        color=sampled_pca_df["cluster"].astype(str),
        title="PCA Projection of Geo Clusters"
    )

    st.plotly_chart(fig_pca, use_container_width=True)


    st.subheader("Cluster Distribution")
    st.bar_chart(df["cluster"].value_counts())


elif selected_model == "Time Based Clustering":

    st.subheader("⏱ Time-Based Clustering Results")

    df = art["time_cyclical_df"].copy()  # 🔹 Use cyclical transformed data
    model = art["kmeans_cyclical"]       # 🔹 Correct key name
    scaler = art["time_scaler"]

    # Use the SAME columns used during training
    feature_cols = scaler.feature_names_in_

    X = df[feature_cols].dropna().reset_index(drop=True)
    X_scaled = scaler.transform(X)

    labels = model.predict(X_scaled)
    df["cluster"] = labels

    n_clusters = len(np.unique(labels))

    # Silhouette Score
    sil_text = "Not Computed"
    if show_silhouette and n_clusters > 1:
        sample_size = min(len(X_scaled), 20000)
        idx = np.random.choice(len(X_scaled), sample_size, replace=False)
        sil_text = f"{silhouette_score(X_scaled[idx], labels[idx]):.4f}"

    col1, col2, col3 = st.columns(3)
    col1.metric("Data Points", len(X))
    col2.metric("Clusters", n_clusters)
    col3.metric("Silhouette Score", sil_text)

    sampled_df = df.sample(frac=sample_frac, random_state=42)

    # Plot based on raw readable columns, not cyclical ones
    if "Hour" in df.columns and "day_of_week" in df.columns:
        fig = px.scatter(
            sampled_df,
            x="day_of_week", y="Hour",
            color=sampled_df["cluster"].astype(str),
            title="Time Cluster Visualization (Readable Scale)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # PCA
    st.subheader("PCA View")
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])
    pca_df["cluster"] = labels

    fig_pca = px.scatter(
        pca_df.sample(frac=sample_frac),
        x="PC1", y="PC2",
        color=pca_df["cluster"].astype(str),
        title="PCA Projection of Time-Based Clusters"
    )

    st.plotly_chart(fig_pca, use_container_width=True)

    st.subheader("Cluster Distribution")
    st.bar_chart(df["cluster"].value_counts())


# elif selected_model == "Crime Severity Clustering":

#     st.subheader("🚨 Crime Severity Clustering Results")

#     df = art["new_crime_df"].copy()  
#     model = art["kmeans_crime"]      
#     scaler = art["crime_scaler"]

#     feature_cols = ['crime_category','Severity_Score','Arrest','Domestic']

#     X = df[feature_cols].dropna().reset_index(drop=True)

#     X_scaled = scaler.transform(X)

#     labels = model.predict(X_scaled)
#     df["cluster"] = labels

#     n_clusters = len(np.unique(labels))

#     sil_text = "Not Computed"
#     if show_silhouette and n_clusters > 1:
#         sample_size = min(len(X_scaled), 20000)
#         idx = np.random.choice(len(X_scaled), size=sample_size, replace=False)
#         sil_text = f"{silhouette_score(X_scaled[idx], labels[idx]):.4f}"

#     # Summary Metrics
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Data Points", len(X))
#     col2.metric("Clusters", n_clusters)
#     col3.metric("Silhouette Score", sil_text)

#     # Sample for visualization
#     sampled_df = df.sample(frac=sample_frac, random_state=42)

#     # -----------------------------
#     # 🔹 Scatter Plot: Severity vs Arrest
#     # -----------------------------
#     fig = px.scatter(
#         sampled_df,
#         x="Severity_Score",
#         y="Arrest",
#         color=sampled_df["cluster"].astype(str),
#         title="Crime Clustering Scatter Plot (Severity vs Arrest)",
#         hover_data=["Domestic"]
#     )

#     # Add centroids if requested
#     if show_centroids:
#         centroids = scaler.inverse_transform(model.cluster_centers_)
#         fig.add_scatter(
#             x=centroids[:, 0],
#             y=centroids[:, 1],
#             mode="markers+text",
#             text=[f"C{i}" for i in range(len(centroids))],
#             marker=dict(size=15, symbol="x")
#         )

#     st.plotly_chart(fig, use_container_width=True)


#     st.subheader("PCA Projection")

#     pca = PCA(n_components=2)
#     pca_data = pca.fit_transform(X_scaled)

#     pca_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])
#     pca_df["cluster"] = labels

#     sampled_pca = pca_df.sample(frac=sample_frac, random_state=42)

#     fig_pca = px.scatter(
#         sampled_pca,
#         x="PC1",
#         y="PC2",
#         color=sampled_pca["cluster"].astype(str),
#         title="PCA Visualization of Crime Clusters"
#     )

#     st.plotly_chart(fig_pca, use_container_width=True)

#     # -----------------------------
#     # 🔹 Cluster Distribution Chart
#     # -----------------------------
#     st.subheader("Cluster Distribution")
#     st.bar_chart(df["cluster"].value_counts())
