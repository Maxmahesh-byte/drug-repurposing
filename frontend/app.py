import streamlit as st
import requests

st.title("🧬 Cancer Drug Repurposing – GNN MLOps Pipeline")
st.markdown("**Using real GDSC data**")

drug_name = st.selectbox("Select Drug (from GDSC)", 
                        ["Methotrexate", "Paclitaxel", "Cisplatin", "Doxorubicin", "Imatinib"])
target_cancer = st.text_input("Target Cancer Type", "Lung Cancer")

if st.button("🚀 Predict Repurposing"):
    with st.spinner("Running GNN on heterogeneous KG..."):
        response = requests.post(
            "http://localhost:8000/predict",
            json={"drug_id": drug_name, "target_disease": target_cancer}
        )
        if response.status_code == 200:
            result = response.json()
            st.success(f"**Repurposing Score: {result['repurposing_score']:.2f}**")
            st.info(result["explanation"])
        else:
            st.error("API error")