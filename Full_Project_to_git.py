import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from huggingface_hub import InferenceClient
from sklearn.metrics import silhouette_score
from dotenv import load_dotenv
from huggingface_hub.errors import HfHubHTTPError

# =========================================================
# COMPLETE YOURSELF FUNCTIONS
# =========================================================



def get_hf_token():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    return hf_token

def build_elbow_graph_by_k(X: np.ndarray, k_min: int, k_max: int, method: str) -> pd.DataFrame:
    # Fixed fake elbow curve (nice smooth drop)
    inertias , silhouette = [],[]
    k_values = range(k_min, k_max + 1)

    if method == 'wcss':
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            result = kmeans.inertia_
            inertias.append({'k': k,'wcss':result})
        return pd.DataFrame(inertias)

    else:
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)

            if len(np.unique(labels)) >= 2:
                score = silhouette_score(X, labels)
            else:
                score = -1
            silhouette.append({'k': k,'silhouette':score})

        return pd.DataFrame(silhouette)


def fit_kmeans_labels(x: np.ndarray, k: int) -> np.ndarray:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_of_clusters = kmeans.fit_predict(x)
    return labels_of_clusters


@st.cache_resource
def ask_llama_for_name_desc(cluster_summary: str) -> tuple[str, str]:
    # Fixed fake "LLM" answers based on cluster id found in text
    hf_token = get_hf_token()
    if hf_token is None:
        st.error("Missing HuggingFace token")
        st.stop()


    client = InferenceClient(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        token=hf_token
    )

    messages = [{"role": "system","content": "You are a data analyst who explains clusters based on statistical summaries."},
                {"role": "user","content": f"""Given the following cluster summary, return ONLY a valid JSON in this format:
                {{
                "name": "<short cluster name>",
                "description": "<short description of 7-10 words>"
                }}
    
                Cluster:
                {cluster_summary}
                """}]
    try:
        response = client.chat_completion(
        messages=messages,
        max_tokens=150,
        temperature=0.5,
        top_p=0.9,
    )
    except HfHubHTTPError as e:
        if "402" in str(e):
            st.error("HuggingFace credits finished")
            st.stop()
    content = response.choices[0].message["content"]

    try:
        #serach for json in response
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("No JSON found")

        data = json.loads(match.group())

        cluster_name = data.get("name", "").strip()
        description = data.get("description", "").strip()

    except Exception:
        # fallback if no valid json
        cluster_name = "cluster"
        description = content[:80]

    if not cluster_name:
        cluster_name = "cluster"
    if not description:
        description = "No description available"

    return cluster_name, description

@st.cache_resource
def ask_llama_for_optimal_k(wcss_df: pd.DataFrame) -> tuple[str, str]:
    hf_token = get_hf_token()
    if hf_token is None:
        st.error("Missing HuggingFace token")
        st.stop()

    client = InferenceClient(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        token=hf_token)

    messages = [
        {"role": "system", "content": "You are a data analyst who choose the optimal k by the wcss dataframe you get."},
        {"role": "user",
         "content": f"by this data frame pick the optimal number of clusters you think the user should choose for the best clustering return in massage only integer \n{wcss_df.to_string()}"}
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=150,
            temperature=0.5,
            top_p=0.9,
        )
    except HfHubHTTPError as e:
        if "402" in str(e):
            st.error("HuggingFace credits finished")
            st.stop()

    answer = response.choices[0].message["content"]
    match = re.search(r"\d+", answer)

    if match is None:
        st.error("Response llama Error")
        st.stop()

    optimal_k = int(match.group())
    return optimal_k

# =========================================================
# PROVIDED HELPERS (not the focus)
# =========================================================

def preprocess(df_features: pd.DataFrame) -> np.ndarray:
    df_features = df_features.dropna(how="all").dropna(axis=1, how="all")

    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df_features.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ],
        remainder="drop",
    )

    return preprocessor.fit_transform(df_features)


def build_cluster_summary(df_features: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    tmp = df_features.copy()

    numeric_cols = tmp.select_dtypes(include=[np.number]).columns
    cat_cols = tmp.select_dtypes(exclude=[np.number]).columns

    tmp["_cluster"] = labels

    out = {}

    for c in sorted(tmp["_cluster"].unique()):
        part = tmp[tmp["_cluster"] == c]
        lines = [f"Cluster {c}: n={len(part)}"]

        if len(numeric_cols) > 0:
            means = part[numeric_cols].mean(numeric_only=True).sort_values(ascending=False).head(8)
            means_text = ", ".join([f"{col}={means[col]:.2f}" for col in means.index])
            lines.append(f"Numeric averages (top): {means_text}")

        if len(cat_cols) > 0:
            bits = []
            for col in list(cat_cols)[:6]:
                vc = part[col].astype(str).value_counts(dropna=False).head(3)
                top_vals = "; ".join([f"{idx}({cnt})" for idx, cnt in vc.items()])
                bits.append(f"{col}: {top_vals}")
            lines.append("Categorical top values: " + " | ".join(bits))

        out[int(c)] = "\n".join(lines)

    return out


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(page_title="Segment Studio", layout="wide")
st.title("Segment Studio")

st.session_state.setdefault("general_df", None)
st.session_state.setdefault("wcss_or_silhouette", None)
st.session_state.setdefault("wcss_df", None)
st.session_state.setdefault("silhouette_df", None)
st.session_state.setdefault("choose_k", None)
st.session_state.setdefault("chosen_k", None)
st.session_state.setdefault("labels", None)
st.session_state.setdefault("cluster_counts_table", None)
st.session_state.setdefault("cluster_labels_table", None)
st.session_state.setdefault("cluster_name_map", None)

st.session_state.setdefault("elbow_run_done", False)  #  "Run WCSS/SILHOUETTE"
st.session_state.setdefault("clusters_created", False)  #  "Create clusters"
st.session_state.setdefault("names_generated", False)  #  "Generate names/descriptions"


# Step 1: Upload
uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to continue")
    st.stop()

try:
    st.session_state["general_df"] = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

df = st.session_state["general_df"]
st.success("File uploaded successfully ✅")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

df_features = df.dropna(how="all").copy()
if df_features.shape[0] < 3:
    st.error("Need at least 3 rows")
    st.stop()

X = preprocess(df_features)

# Step 2: WCSS \Silhouette elbow
options = ["choose", "wcss", "silhouette"]

current_value = st.session_state.get("wcss_or_silhouette", "choose")

if current_value not in options:
    current_value = "choose"

method_choice = st.selectbox(
    "Elbow method chosen",
    options=options,
    index=options.index(current_value)
)

st.session_state["wcss_or_silhouette"] = method_choice

if st.session_state['wcss_or_silhouette'] == "choose":
    st.warning("You have to choose method style")
    st.stop()

st.info(st.session_state['wcss_or_silhouette'])

st.subheader("Step 2: WCSS (Elbow)" if st.session_state['wcss_or_silhouette'] == 'wcss' else "Step 2: SILHOUETTE (Elbow)" )

n_samples = X.shape[0]
max_allowed_k = min(20, n_samples - 1)

col1, col2 = st.columns(2)
with col1:
    k_min = st.slider("Min k", 2, max_allowed_k, 2, 1,)
with col2:
    k_max = st.slider("Max k", 2, max_allowed_k, value=min(15, max_allowed_k), step= 1)

if k_min >= k_max:
    st.warning("Min k must be smaller than Max k")
    st.stop()

result_df = build_elbow_graph_by_k(X, k_min, k_max,st.session_state['wcss_or_silhouette'])
if st.session_state['wcss_or_silhouette'] == 'wcss':
    st.session_state["wcss_df"] = result_df
else:
    st.session_state["silhouette_df"] = result_df

run_clicked = st.button("Run WCSS" if st.session_state['wcss_or_silhouette'] == 'wcss' else "Run SILHOUETTE")
if run_clicked:
    st.session_state["elbow_run_done"] = True

if not st.session_state["elbow_run_done"]:
    st.info("Click for continue")
    st.stop()

else:
    if st.session_state['wcss_or_silhouette'] == 'wcss':
        wcss_df = st.session_state["wcss_df"]
        st.dataframe(wcss_df, use_container_width=True)
        fig = plt.figure()
        plt.plot(wcss_df["k"], wcss_df["wcss"], marker="o")
        plt.xlabel("k")
        plt.xlim(2, max_allowed_k)
        plt.ylabel("WCSS (inertia)")
        plt.title("Elbow Plot")
        st.pyplot(fig, clear_figure=True)

    else:
        silhouette_df = st.session_state["silhouette_df"]
        print(f"\n\n\n\n\n",silhouette_df)
        st.dataframe(silhouette_df, use_container_width=True)
        fig = plt.figure()
        plt.plot(silhouette_df["k"], silhouette_df["silhouette"], marker="o")
        plt.xlabel("k")
        plt.xlim(2, max_allowed_k)
        plt.ylabel("Silhouette")
        plt.title("Elbow Plot")
        st.pyplot(fig, clear_figure=True)

# Step 3: "K Choosing method"
k_define_options = ["choose", "auto", "manual"]
current_define = st.session_state.get("choose_k", "choose")

st.subheader("Step 3: Choose k and create clusters")

if current_define not in k_define_options:
    current_define = "choose"

choice_define = st.selectbox(
    "K Choosing method",
    options=k_define_options,
    index=k_define_options.index(current_define)
)

st.session_state["choose_k"] = choice_define


st.info(st.session_state['choose_k'])
if st.session_state['choose_k'] == "choose":
    st.warning("K Choosing method required")
    st.stop()


if st.session_state["choose_k"] == 'manual':
    st.session_state["chosen_k"] = st.slider("Select k",min_value=k_min,max_value=k_max,value=k_min,step=1)

else:
    if st.session_state['wcss_or_silhouette'] == 'silhouette':
        idx = st.session_state["silhouette_df"]['silhouette'].idxmax()
        st.session_state["chosen_k"] = st.session_state["silhouette_df"].loc[idx, 'k']
    else:
        st.session_state["chosen_k"] = ask_llama_for_optimal_k(st.session_state["wcss_df"])

st.info(f"K Value:   {st.session_state['chosen_k']}")

# Step 4: Cluster + show counts with empty name/desc
if st.button("Create clusters"):

    labels = fit_kmeans_labels(X, st.session_state["chosen_k"])

    st.session_state["labels"] = labels
    st.session_state["cluster_labels_table"] = None
    st.session_state["cluster_name_map"] = None

    counts = pd.Series(labels).value_counts().sort_index()
    counts_table = pd.DataFrame({
            "cluster_id": counts.index.astype(int),
            "count": counts.values.astype(int),
            "name": [""] * len(counts),
            "description": [""] * len(counts),
        })

    st.session_state["cluster_counts_table"] = counts_table
    st.session_state["clusters_created"] = True

if st.session_state["cluster_counts_table"] is not None:
    st.write("Cluster counts (name/description empty for now)")
    st.dataframe(st.session_state["cluster_counts_table"], use_container_width=True)
else:
    st.warning("Click 'Create clusters' to generate the clusters")
if not st.session_state["clusters_created"]:
    st.stop()

# Step 5: Button to call LLM and fill name/description
st.subheader("Step 4: Generate group name + description (LLaMA)")

if st.button("Generate names/descriptions with LLaMA"):
    if st.session_state["labels"] is None or st.session_state["cluster_counts_table"] is None:
        st.warning("Please create clusters first (Step 3) before generating names/descriptions.")
    else:
        labels = st.session_state["labels"]
        counts_table = st.session_state["cluster_counts_table"]

        summaries = build_cluster_summary(df_features, labels)

        rows = []
        name_map = {}
        with st.spinner("Calling LLaMA for each cluster..."):
            for cluster_id in counts_table["cluster_id"].tolist():
                summary = summaries[int(cluster_id)]
                name,desc = ask_llama_for_name_desc(summary)

                if not name:
                    name = f"cluster_{int(cluster_id)}"
                if desc is None:
                    desc = ""

                name_map[int(cluster_id)] = name
                rows.append({
                    "cluster_id": int(cluster_id),
                    "count": int(counts_table.loc[counts_table["cluster_id"] == cluster_id, "count"].iloc[0]),
                    "name": name,
                    "description": desc,
                })

        labeled_table = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
        st.session_state["cluster_labels_table"] = labeled_table
        st.session_state["cluster_name_map"] = name_map
        st.session_state["names_generated"] = True

if st.session_state["cluster_labels_table"] is not None:
    st.write("Cluster labels (with name + description)")
    st.dataframe(st.session_state["cluster_labels_table"], use_container_width=True)
else:
    st.info("Click 'Generate names/descriptions with LLaMA' to fill Step 4")

if not st.session_state["names_generated"]:
    st.stop()
# Step 6: Export
st.subheader("Step 5: Export clustered CSV")

if st.session_state["labels"] is None:
    st.info("Create clusters (Step 3) to enable the export.")
else:
    labels = st.session_state["labels"]
    name_map = st.session_state["cluster_name_map"] or {}
    cluster_names = [name_map.get(int(c), f"cluster_{int(c)}") for c in labels]

    df_out = df.copy()
    df_out["cluster_name"] = cluster_names

    original_name = getattr(uploaded, "name", "input.csv")
    out_name = original_name[:-4] + "_clustered.csv" if original_name.lower().endswith(".csv") else original_name + "_clustered.csv"

    st.download_button(
        "Download clustered CSV",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name=out_name,
        mime="text/csv",
    )