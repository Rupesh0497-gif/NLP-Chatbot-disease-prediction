import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
dataset = pd.read_csv("dataset.csv")
desc_df = pd.read_csv("symptom_Description.csv")
precaution_df = pd.read_csv("symptom_precaution.csv")
severity_df = pd.read_csv("Symptom-severity.csv")

# Preprocess dataset
symptom_columns = dataset.columns[1:]
dataset["Symptoms"] = dataset[symptom_columns].apply(
    lambda x: [i.strip().lower() for i in x.dropna().astype(str)], axis=1
)
dataset = dataset[["Disease", "Symptoms"]]

# Unique symptoms
symptom_list = sorted(set(symptom for symptoms in dataset["Symptoms"] for symptom in symptoms))

# One-hot encoding
for symptom in symptom_list:
    dataset[symptom] = dataset["Symptoms"].apply(lambda x: 1 if symptom in x else 0)

X = dataset.drop(columns=["Disease", "Symptoms"])
y = dataset["Disease"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Helper maps
desc_map = dict(zip(desc_df["Disease"].str.lower(), desc_df["Description"]))
precaution_map = precaution_df.set_index("Disease").apply(lambda x: x.dropna().tolist(), axis=1).to_dict()
severity_map = dict(zip(severity_df["Symptom"].str.lower(), severity_df["weight"]))

# Disease matcher
def find_unique_matching_diseases(symptom_input):
    user_symptoms = [s.strip().lower() for s in symptom_input.split(",")]
    valid_symptoms = [s for s in user_symptoms if s in symptom_list]

    if not valid_symptoms:
        return []

    disease_matches = {}
    for _, row in dataset.iterrows():
        disease = row["Disease"]
        matched = list(set(valid_symptoms) & set(row["Symptoms"]))
        if matched:
            if disease not in disease_matches:
                disease_matches[disease] = set()
            disease_matches[disease].update(matched)

    results = []
    for disease, matched_set in disease_matches.items():
        matched = list(matched_set)
        avg_severity = sum([severity_map.get(sym, 0) for sym in matched]) / len(matched)
        results.append({
            "Disease": disease,
            "Matched Symptoms": matched,
            "Avg Severity": avg_severity,
            "Description": desc_map.get(disease.lower(), "Description not available."),
            "Precautions": precaution_map.get(disease, ["No precautions found."])
        })

    return results

# Streamlit UI
st.title("🩺 Symptom-Based Disease Info System")
st.write("Enter symptoms (comma separated) like `shivering, chills, joint_pain`")

if st.checkbox("Show valid symptoms"):
    st.write(", ".join(symptom_list))

user_input = st.text_input("Enter your symptoms:")

if user_input.strip() == "":
    st.info("ℹ️ Please enter symptoms above to see results.")
else:
    if st.button("Show Matching Diseases"):
        results = find_unique_matching_diseases(user_input)

        if results:
            st.success(f"✅ Found {len(results)} matching disease(s).")
            for res in results:
                st.header(f"🦠 {res['Disease']}")
                st.info(f"📖 **Description:** {res['Description']}")
                st.warning(f"⚖️ **Avg Severity of Matched Symptoms ({len(res['Matched Symptoms'])}):** {res['Avg Severity']:.2f}")
                st.write("🩺 **Matched Symptoms:**", ", ".join(res["Matched Symptoms"]))
                st.write("🛡️ **Precautions:**")
                for p in res["Precautions"]:
                    st.write(f"- {p}")
                st.markdown("---")

            # Show model accuracy after valid match
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.header("📈 Model Evaluation")
            st.metric(label="🎯 Model Accuracy", value=f"{accuracy:.2%}")
        else:
            st.error("❌ No diseases matched your symptoms.")
