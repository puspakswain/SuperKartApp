
from flask import Flask, request, jsonify
import pandas as pd
import joblib
app = Flask(__name__)
model = joblib.load("superkart_model.joblib")
FEATURE_COLUMNS = ["Product_Weight","Product_Sugar_Content","Product_Allocated_Area","Product_MRP","Store_Size","Store_Location_City_Type","Store_Type","Product_Id_char","Store_Age_Years","Product_Type_Category"]
@app.get("/health")
def health(): return jsonify({"status":"ok"})
@app.post("/v1/predict")
def predict():
    try:
        payload=request.get_json(force=True); missing=[c for c in FEATURE_COLUMNS if c not in payload]
        if missing: return jsonify({"error":f"Missing required fields: {missing}"}),400
        pred=float(model.predict(pd.DataFrame([payload])[FEATURE_COLUMNS])[0]); return jsonify({"predicted_sales":pred})
    except Exception as exc: return jsonify({"error":str(exc)}),500
@app.post("/v1/predictbatch")
def predict_batch():
    try:
        if "file" not in request.files: return jsonify({"error":"Upload a CSV using the 'file' field."}),400
        batch_df=pd.read_csv(request.files["file"]); missing=[c for c in FEATURE_COLUMNS if c not in batch_df.columns]
        if missing: return jsonify({"error":f"Missing required columns: {missing}"}),400
        preds=model.predict(batch_df[FEATURE_COLUMNS]); return jsonify({str(i):float(p) for i,p in enumerate(preds)})
    except Exception as exc: return jsonify({"error":str(exc)}),500
if __name__=="__main__": app.run(host="0.0.0.0",port=7860)
