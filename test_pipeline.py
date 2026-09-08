import requests, json
from ml.nlp.clinical_parser import parse_clinical_text

def test_disease(name, notes):
    print(f"\n==========================================")
    print(f"TESTING DISEASE: {name.upper()}")
    print(f"Notes: {notes}")
    res = parse_clinical_text(notes)
    print(f"Extracted Vitals Count: {len(res['vitals'])}")
    print(f"Vitals: {list(res['vitals'].keys())}")
    
    resp = requests.post(f"http://127.0.0.1:8000/api/predict/{name}", json={"features": res["vitals"]})
    print(f"HTTP Status: {resp.status_code}")
    data = resp.json()
    print(f"Prediction: {data.get('prediction')} | Probability: {data.get('probability')}% | Status: {data.get('status')}")
    
    exp = data.get("explanation")
    if exp:
        print(f"Explainer Type: {exp.get('explainer_type')}")
        print(f"Primary Driver: {exp.get('primary_driver')}")
        print("Top Contributing Factors:")
        for f in exp.get("top_5_contributing", []):
            print(f"   • {f['display_name']} ({f['recorded_value']}): SHAP {f['shap_value']:+.3f} [{f['direction']}]")
        print("Clinical Rationale:")
        for s in exp.get("clinical_explanation", []):
            print(f"   - {s}")
    else:
        print("WARNING: No explanation returned!")

if __name__ == "__main__":
    # Test across 4 diverse diseases
    test_disease("diabetes", "54-year-old female with BP 142/88, fasting blood sugar 172 mg/dL, BMI 31.8, insulin 145, 2 prior pregnancies, and maternal history of diabetes.")
    test_disease("heart", "62-year-old male with resting BP 155/92, total cholesterol 245 mg/dL, max heart rate 165 bpm, ST depression 1.8, 1 major vessel, smoker.")
    test_disease("bp", "58-year-old male with BP 165/100, pulse 88, BMI 29.5, cholesterol 230, sedentary smoker.")
    test_disease("stroke", "68-year-old married woman, glucose 180 mg/dL, BMI 30.2, hypertension diagnosed, former smoker.")
