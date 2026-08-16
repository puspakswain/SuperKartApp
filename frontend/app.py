
import os, requests, pandas as pd, streamlit as st
BACKEND_URL=os.getenv("BACKEND_URL","http://backend:7860").rstrip("/")
PREDICT_URL=BACKEND_URL+"/v1/predict"; BATCH_URL=BACKEND_URL+"/v1/predictbatch"
st.set_page_config(page_title="SuperKart Sales Forecast",layout="wide")
st.title("SuperKart Sales Forecast"); st.caption(f"Backend: {BACKEND_URL}")
col1,col2=st.columns(2)
with col1:
    product_weight=st.number_input("Product Weight",value=12.66); sugar=st.selectbox("Product Sugar Content",["Low Sugar","Regular","No Sugar"]); allocated_area=st.number_input("Product Allocated Area",value=0.027,min_value=0.0,format="%.3f"); mrp=st.number_input("Product MRP",value=117.08); store_size=st.selectbox("Store Size",["Small","Medium","High"])
with col2:
    city_tier=st.selectbox("Store Location City Type",["Tier 1","Tier 2","Tier 3"]); store_type=st.selectbox("Store Type",["Supermarket Type1","Supermarket Type2","Supermarket Type3","Departmental Store","Food Mart"]); product_id_char=st.selectbox("Product ID Prefix",["FD","DR","NC"]); store_age=st.number_input("Store Age (Years)",value=16,min_value=0,step=1); product_category=st.selectbox("Product Type Category",["Perishables","Non Perishables"])
payload={"Product_Weight":product_weight,"Product_Sugar_Content":sugar,"Product_Allocated_Area":allocated_area,"Product_MRP":mrp,"Store_Size":store_size,"Store_Location_City_Type":city_tier,"Store_Type":store_type,"Product_Id_char":product_id_char,"Store_Age_Years":store_age,"Product_Type_Category":product_category}
if st.button("Predict Sales"):
    try:
        r=requests.post(PREDICT_URL,json=payload,timeout=30); r.raise_for_status(); st.success(f"Predicted sales: {r.json()['predicted_sales']:.2f}")
    except Exception as exc: st.error(f"Prediction failed: {exc}")
st.subheader("Batch Prediction"); uploaded=st.file_uploader("Upload CSV with the 10 model features",type=["csv"])
if uploaded is not None and st.button("Run Batch Prediction"):
    try:
        r=requests.post(BATCH_URL,files={"file":uploaded.getvalue()},timeout=60); r.raise_for_status(); result=r.json(); st.dataframe(pd.DataFrame({"row_index":list(result.keys()),"predicted_sales":list(result.values())}),use_container_width=True)
    except Exception as exc: st.error(f"Batch prediction failed: {exc}")
