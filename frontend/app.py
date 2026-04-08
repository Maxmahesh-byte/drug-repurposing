import streamlit as st
import requests

st.title("🧬 Cancer Drug Repurposing – GNN MLOps Pipeline")
st.markdown("**Using real GDSC data**")

drug_name = st.selectbox("Select Drug (from GDSC)", 
                        ["Methotrexate", "Paclitaxel", "Cisplatin", "Doxorubicin", "Imatinib"])
target_cancer = st.text_input("Target Cancer Type", "Lung Cancer")

if st.button("🚀 Predict Repurposing"):
    with st.spinner("Running GNN on heterogeneous KG..."):
        try:
            # Important: Use service name 'fastapi' instead of localhost
            response = requests.post(
                "http://fastapi:8000/predict",
                json={"drug_id": drug_name, "target_disease": target_cancer},
                timeout=10
            )
            if response.status_code == 200:
                result = response.json()
                st.success(f"**Repurposing Score: {result['repurposing_score']:.2f}**")
                st.info(result["explanation"])
            else:
                st.error(f"API returned error: {response.status_code}")
        except Exception as e:
            st.error(f"Connection error: {e}")