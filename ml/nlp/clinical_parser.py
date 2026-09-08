import re
from typing import Dict, Any, List, Optional

class ClinicalNLPParser:
    """
    Advanced Clinical NLP Parser for extracting vitals, biomarkers,
    symptoms, and risk triage from unstructured clinical notes.
    """

    SYMPTOM_MAP = {
        "diabetes": [
            "frequent urination", "polyuria", "excessive thirst", "polydipsia", "unexplained weight loss",
            "extreme hunger", "polyphagia", "blurred vision", "slow healing sores", "tingling hands",
            "tingling feet", "neuropathy", "sweet breath", "high sugar", "hyperglycemia"
        ],
        "heart": [
            "chest pain", "angina", "shortness of breath", "dyspnea", "palpitations", "racing heartbeat",
            "dizziness", "lightheadedness", "left arm pain", "jaw pain", "cold sweat", "high cholesterol",
            "hypertension", "cyanosis", "swollen ankles", "edema"
        ],
        "stroke": [
            "facial drooping", "arm weakness", "slurred speech", "sudden numbness", "vision loss in one eye",
            "sudden severe headache", "loss of balance", "ataxia", "confusion", "aphasia", "hemiplegia"
        ],
        "kidney": [
            "decreased urine", "foamy urine", "proteinuria", "blood in urine", "hematuria", "puffy eyes",
            "metallic taste", "nausea", "loss of appetite", "flank pain", "fatigue", "elevated creatinine"
        ],
        "liver": [
            "yellowing skin", "jaundice", "yellow eyes", "dark urine", "pale stool", "abdominal swelling",
            "ascites", "itchy skin", "pruritus", "elevated bilirubin", "hepatic tenderness"
        ],
        "anemia": [
            "pale skin", "pallor", "chronic fatigue", "cold hands", "cold feet", "brittle nails",
            "dizziness", "weakness", "headache", "low hemoglobin", "shortness of breath on exertion"
        ],
        "obesity": [
            "excessive weight", "high bmi", "joint pain", "sleep apnea", "snoring", "breathlessness",
            "metabolic syndrome", "sedentary"
        ]
    }

    CRITICAL_TRIGGERS = [
        "crushing chest pain", "facial drooping", "slurred speech", "sudden numbness", "loss of consciousness",
        "systolic bp over 180", "unresponsive", "severe dyspnea", "cyanosis", "massive bleeding", "hemiplegia"
    ]

    URGENT_TRIGGERS = [
        "chest pain", "shortness of breath", "glucose over 250", "high fever", "irregular pulse",
        "severe headache", "jaundice", "blood in urine", "edema", "sudden vision change"
    ]

    def __init__(self):
        pass

    def extract_vitals(self, text: str) -> Dict[str, Any]:
        """
        Extract numerical biometric parameters from free text.
        """
        lower = text.lower()
        vitals: Dict[str, Any] = {}

        # 1. Age
        age_match = re.search(r'\b(?:age|aged)\s*[:=]?\s*(\d{1,3})\b', lower)
        if not age_match:
            age_match = re.search(r'\b(\d{1,3})\s*[- ]*(?:years?[- ]*old|yo|y/o)\b', lower)
        if not age_match:
            age_match = re.search(r'\b(\d{1,3})\s*[-]?year[-]?old\b', lower)
        if age_match:
            try:
                v = int(age_match.group(1))
                if 1 <= v <= 120:
                    vitals["Age"] = v
                    vitals["age"] = v
            except:
                pass

        # 2. Gender / Sex
        if re.search(r'\b(female|woman|girl|she|her)\b', lower):
            vitals["Gender"] = "Female"
            vitals["sex"] = 0
            vitals["gender"] = 0
        elif re.search(r'\b(male|man|boy|he|his)\b', lower):
            vitals["Gender"] = "Male"
            vitals["sex"] = 1
            vitals["gender"] = 1

        # 3. Blood Pressure
        bp_match = re.search(r'\b(?:bp|blood pressure|pressure)\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})\b', lower)
        if bp_match:
            sys_bp = int(bp_match.group(1))
            dia_bp = int(bp_match.group(2))
            vitals["BloodPressure"] = dia_bp
            vitals["trestbps"] = sys_bp
            vitals["bp"] = dia_bp
            vitals["systolic_bp"] = sys_bp
            vitals["diastolic_bp"] = dia_bp
            vitals["systolic"] = sys_bp
            vitals["diastolic"] = dia_bp
        else:
            single_bp = re.search(r'\b(?:bp|blood pressure)\s*[:=]?\s*(\d{2,3})\s*(?:mmhg)?\b', lower)
            if single_bp:
                v = int(single_bp.group(1))
                vitals["BloodPressure"] = v
                vitals["trestbps"] = v
                vitals["bp"] = v
                vitals["systolic"] = v

        # 4. Glucose / Blood Sugar
        gluc_match = re.search(r'\b(?:glucose|blood sugar|sugar|fbs|fasting glucose)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:mg/dl|mg/l)?\b', lower)
        if gluc_match:
            val = float(gluc_match.group(1))
            vitals["Glucose"] = val
            vitals["avg_glucose_level"] = val
            vitals["fbs"] = 1 if val > 120 else 0

        # 5. Cholesterol
        chol_match = re.search(r'\b(?:cholesterol|total cholesterol|chol)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:mg/dl)?\b', lower)
        if chol_match:
            vitals["chol"] = float(chol_match.group(1))
            vitals["Cholesterol"] = float(chol_match.group(1))
            vitals["cholesterol"] = float(chol_match.group(1))

        # 6. Heart Rate / Pulse (thalach)
        hr_match = re.search(r'\b(?:heart rate|pulse|pulse rate|hr|bpm|max hr)\s*[:=]?\s*(\d{2,3})\s*(?:bpm)?\b', lower)
        if hr_match:
            hr_val = int(hr_match.group(1))
            vitals["thalach"] = hr_val
            vitals["HeartRate"] = hr_val
            vitals["pulse"] = hr_val

        # 7. BMI
        bmi_match = re.search(r'\b(?:bmi|body mass index)\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)\b', lower)
        if bmi_match:
            vitals["BMI"] = float(bmi_match.group(1))
            vitals["bmi"] = float(bmi_match.group(1))

        # 8. Weight & Height
        wt_match = re.search(r'\b(?:weight|wt)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:kg|kgs|kilograms)?\b', lower)
        if wt_match:
            vitals["Weight"] = float(wt_match.group(1))
        
        ht_match = re.search(r'\b(?:height|ht)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:cm|centimeters)?\b', lower)
        if ht_match:
            vitals["Height"] = float(ht_match.group(1))

        if "BMI" not in vitals and "Weight" in vitals and "Height" in vitals:
            ht_m = vitals["Height"] / 100.0
            if ht_m > 0:
                calc_bmi = round(vitals["Weight"] / (ht_m * ht_m), 1)
                vitals["BMI"] = calc_bmi
                vitals["bmi"] = calc_bmi

        # 9. Hemoglobin
        hb_match = re.search(r'\b(?:hemoglobin|hb|hgb)\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)\s*(?:g/dl)?\b', lower)
        if hb_match:
            vitals["Hemoglobin"] = float(hb_match.group(1))

        # 10. Bilirubin
        bili_match = re.search(r'\b(?:bilirubin|total bilirubin|tb)\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)\s*(?:mg/dl)?\b', lower)
        if bili_match:
            vitals["Total_Bilirubin"] = float(bili_match.group(1))

        # 11. Insulin
        ins_match = re.search(r'\b(?:insulin|fasting insulin)\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)\s*(?:mu/l)?\b', lower)
        if ins_match:
            vitals["Insulin"] = float(ins_match.group(1))

        # 12. Skin Thickness
        st_match = re.search(r'\b(?:skin thickness|skinfold)\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)\s*(?:mm)?\b', lower)
        if st_match:
            vitals["SkinThickness"] = float(st_match.group(1))

        # 13. Pregnancies
        preg_match = re.search(r'\b(?:pregnancies|gravida|pregnancy count)\s*[:=]?\s*(\d{1,2})\b', lower)
        if preg_match:
            vitals["Pregnancies"] = int(preg_match.group(1))

        # 14. Smoking Status
        if re.search(r'\b(?:smoker|smokes|current smoker|daily smoker)\b', lower):
            vitals["smoking_status"] = 3
        elif re.search(r'\b(?:former smoker|quit smoking|ex-smoker)\b', lower):
            vitals["smoking_status"] = 2
        elif re.search(r'\b(?:never smoked|non-smoker|non smoker)\b', lower):
            vitals["smoking_status"] = 0

        # 15. Hypertension
        if re.search(r'\b(?:hypertension|htn|high bp diagnosed)\b', lower):
            vitals["hypertension"] = 1
        
        # 16. Chest Pain Type
        if re.search(r'\b(?:typical angina|severe chest pressure|crushing chest pain)\b', lower):
            vitals["cp"] = 0
        elif re.search(r'\b(?:atypical angina|sharp chest pain)\b', lower):
            vitals["cp"] = 1
        elif re.search(r'\b(?:non-anginal|non anginal pain)\b', lower):
            vitals["cp"] = 2
        elif re.search(r'\b(?:asymptomatic|no chest pain)\b', lower):
            vitals["cp"] = 3

        # 17. Diabetes Pedigree Function
        ped_match = re.search(r'\b(?:pedigree|diabetes pedigree|dpf)\s*[:=]?\s*(\d(?:\.\d+)?)\b', lower)
        if ped_match:
            vitals["DiabetesPedigreeFunction"] = float(ped_match.group(1))
        elif re.search(r'\b(?:family history of diabetes|diabetic parent|diabetic mother|diabetic father)\b', lower):
            vitals["DiabetesPedigreeFunction"] = 0.75

        # 18. Heart Disease history
        if re.search(r'\b(?:coronary artery disease|cad|history of heart disease|heart disease|prior mi|myocardial infarction)\b', lower):
            vitals["heart_disease"] = 1
        elif re.search(r'\b(?:no heart disease|no cardiac history)\b', lower):
            vitals["heart_disease"] = 0

        # 19. Marital Status
        if re.search(r'\b(?:married|has spouse|husband|wife)\b', lower):
            vitals["ever_married"] = 1
        elif re.search(r'\b(?:never married|single|unmarried)\b', lower):
            vitals["ever_married"] = 0

        # 20. Residence Type (1 = Urban, 0 = Rural)
        if re.search(r'\b(?:urban|city|metropolitan)\b', lower):
            vitals["Residence_type"] = 1
        elif re.search(r'\b(?:rural|village|countryside)\b', lower):
            vitals["Residence_type"] = 0

        # 21. Alcohol Consumption
        if re.search(r'\b(?:drinks alcohol|alcohol consumption|alcoholic|drinks daily|social drinker|alcohol:?\s*yes)\b', lower):
            vitals["alcohol"] = 1
        elif re.search(r'\b(?:no alcohol|non-drinker|teetotaler|alcohol:?\s*no)\b', lower):
            vitals["alcohol"] = 0

        # 22. Physical Activity
        if re.search(r'\b(?:sedentary|inactive|no exercise|rarely exercises)\b', lower):
            vitals["active"] = 0
        elif re.search(r'\b(?:active|exercises regularly|physically active|daily workouts|athletic)\b', lower):
            vitals["active"] = 1

        # 23. Major Vessels / ST Depression / Exercise Angina (Heart)
        oldpeak_match = re.search(r'\b(?:oldpeak|st depression)\s*[:=]?\s*(\d(?:\.\d+)?)\b', lower)
        if oldpeak_match:
            vitals["oldpeak"] = float(oldpeak_match.group(1))

        ca_match = re.search(r'\b(?:major vessels|vessels|ca)\s*[:=]?\s*(\d)\b', lower)
        if ca_match:
            vitals["ca"] = int(ca_match.group(1))

        if re.search(r'\b(?:exercise angina|angina on exertion|exang:?\s*yes)\b', lower):
            vitals["exang"] = 1
        elif re.search(r'\b(?:no exercise angina|exang:?\s*no)\b', lower):
            vitals["exang"] = 0

        return vitals

    def extract_symptoms(self, text: str) -> Dict[str, List[str]]:
        """
        Identify clinical symptoms grouped by diagnostic category.
        """
        lower = text.lower()
        detected: Dict[str, List[str]] = {}

        for category, symptoms in self.SYMPTOM_MAP.items():
            cat_symptoms = []
            for s in symptoms:
                pattern = r'\b' + re.escape(s) + r'\b'
                if re.search(pattern, lower):
                    cat_symptoms.append(s)
            if cat_symptoms:
                detected[category] = cat_symptoms

        return detected

    def assess_triage(self, text: str, vitals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate clinical triage urgency and generate alert highlights.
        """
        lower = text.lower()
        alerts = []
        severity = "Routine"
        score = 0  # 0: Routine, 1: Elevated, 2: Urgent, 3: Critical

        # Check Critical Triggers
        for trig in self.CRITICAL_TRIGGERS:
            if trig in lower:
                alerts.append(f"Critical Sign: {trig.title()}")
                score = max(score, 3)

        # Check Urgent Triggers
        for trig in self.URGENT_TRIGGERS:
            if trig in lower and score < 3:
                alerts.append(f"Urgent Finding: {trig.title()}")
                score = max(score, 2)

        # Check Vitals Outliers
        if vitals.get("systolic_bp", 0) >= 180 or vitals.get("trestbps", 0) >= 180:
            alerts.append("Hypertensive Crisis Risk (Systolic BP >= 180 mmHg)")
            score = max(score, 3)
        elif vitals.get("systolic_bp", 0) >= 140 or vitals.get("trestbps", 0) >= 140:
            alerts.append("Stage 2 Hypertension (Systolic BP >= 140 mmHg)")
            score = max(score, 1)

        if vitals.get("Glucose", 0) >= 250 or vitals.get("avg_glucose_level", 0) >= 250:
            alerts.append("Severe Hyperglycemia (Glucose >= 250 mg/dL)")
            score = max(score, 2)

        if vitals.get("thalach", 0) >= 180:
            alerts.append("Tachycardia / Elevated Peak Heart Rate")
            score = max(score, 1)

        if vitals.get("BMI", 0) >= 35:
            alerts.append("Severe Obesity Class II/III (BMI >= 35)")
            score = max(score, 1)

        # Map score to label & theme
        if score >= 3:
            severity = "Critical Alert"
            color = "#ff3366"
            badge_class = "danger"
            action_plan = "Immediate Emergency Physician Consultation & Hospital Triage Required."
        elif score == 2:
            severity = "Urgent Assessment"
            color = "#ff9900"
            badge_class = "warning"
            action_plan = "Priority Diagnostic Workup within 24 Hours Recommended."
        elif score == 1:
            severity = "Elevated Monitoring"
            color = "#00f2ff"
            badge_class = "info"
            action_plan = "Schedule Routine Diagnostic Screening and Biometric Follow-up."
        else:
            severity = "Routine Screening"
            color = "#00e676"
            badge_class = "success"
            action_plan = "Patient metrics appear stable. Standard periodic preventive screening."

        return {
            "severity": severity,
            "score": score,
            "color": color,
            "badge_class": badge_class,
            "alerts": alerts if alerts else ["No acute red-flag anomalies detected."],
            "action_plan": action_plan
        }

    def recommend_modules(self, symptoms: Dict[str, List[str]], vitals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rank recommended diagnostic modules based on extracted symptoms and vitals.
        """
        module_scores: Dict[str, int] = {
            "diabetes": 0, "heart": 0, "stroke": 0,
            "kidney": 0, "liver": 0, "anemia": 0, "obesity": 0, "general_health": 1
        }

        # Symptom weight
        for disease, sym_list in symptoms.items():
            if disease in module_scores:
                module_scores[disease] += len(sym_list) * 3

        # Vital clues
        if "Glucose" in vitals or "avg_glucose_level" in vitals:
            module_scores["diabetes"] += 2
        if "trestbps" in vitals or "chol" in vitals or "thalach" in vitals:
            module_scores["heart"] += 3
        if "hypertension" in vitals or "smoking_status" in vitals:
            module_scores["stroke"] += 2
        if "Hemoglobin" in vitals:
            module_scores["anemia"] += 3
        if "Total_Bilirubin" in vitals:
            module_scores["liver"] += 3
        if "Weight" in vitals or "BMI" in vitals:
            module_scores["obesity"] += 2

        # Sort and build recommendations
        sorted_mods = sorted(module_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for mod, sc in sorted_mods:
            if sc > 0:
                results.append({
                    "id": mod,
                    "relevance_score": sc,
                    "matched_symptoms": symptoms.get(mod, [])
                })

        return results[:4]

    def parse(self, text: str) -> Dict[str, Any]:
        """
        Full end-to-end clinical NLP analysis of unstructured text.
        """
        if not text or not text.strip():
            return {
                "success": False,
                "error": "Empty clinical text provided."
            }

        vitals = self.extract_vitals(text)
        symptoms = self.extract_symptoms(text)
        triage = self.assess_triage(text, vitals)
        recommendations = self.recommend_modules(symptoms, vitals)

        # Generate Clinical Summary
        all_symptoms = [s for lst in symptoms.values() for s in lst]
        summary_lines = []
        if "Age" in vitals and "Gender" in vitals:
            summary_lines.append(f"{vitals['Age']}-year-old {vitals['Gender'].lower()}")
        elif "Age" in vitals:
            summary_lines.append(f"{vitals['Age']}-year-old patient")
        
        if all_symptoms:
            summary_lines.append(f"presenting with {', '.join(all_symptoms[:4])}")
        
        vital_summary = []
        if "BloodPressure" in vitals:
            vital_summary.append(f"BP {vitals.get('trestbps', vitals['BloodPressure'])}/{vitals['BloodPressure']} mmHg")
        if "Glucose" in vitals:
            vital_summary.append(f"Glucose {vitals['Glucose']} mg/dL")
        if "chol" in vitals:
            vital_summary.append(f"Cholesterol {vitals['chol']} mg/dL")
        if "BMI" in vitals:
            vital_summary.append(f"BMI {vitals['BMI']}")

        narrative = " ".join(summary_lines).capitalize() if summary_lines else "Clinical parameters analyzed."
        if vital_summary:
            narrative += f" Key vitals recorded: {', '.join(vital_summary)}."

        return {
            "success": True,
            "vitals": vitals,
            "symptoms": symptoms,
            "total_symptoms_detected": len(all_symptoms),
            "triage": triage,
            "recommended_modules": recommendations,
            "clinical_narrative": narrative,
            "raw_text_length": len(text)
        }

# Global helper function
_parser_instance = ClinicalNLPParser()
def parse_clinical_text(text: str) -> Dict[str, Any]:
    return _parser_instance.parse(text)

def map_vitals_to_feature_value(field_id: str, vitals: dict, old_val=None):
    """
    Intelligently maps extracted clinical NLP vitals to any disease model feature.
    Works across Diabetes, Heart, Stroke, Kidney, Liver, Anemia, BP, Obesity, and General Health.
    """
    f_lower = str(field_id).strip().lower()
    v_lower = {str(k).strip().lower(): v for k, v in vitals.items()}

    # Exact key match
    if field_id in vitals:
        return vitals[field_id]
    if f_lower in v_lower:
        return v_lower[f_lower]

    # Age
    if f_lower == "age":
        return v_lower.get("age", old_val)

    # Sex / Gender
    if f_lower in ["sex", "gender"]:
        return v_lower.get("sex", v_lower.get("gender", old_val))

    # Blood Pressure mappings
    if f_lower in ["trestbps", "systolic", "systolic_bp"]:
        return v_lower.get("systolic", v_lower.get("trestbps", v_lower.get("systolic_bp", v_lower.get("bloodpressure", old_val))))
    if f_lower in ["diastolic", "diastolic_bp"]:
        return v_lower.get("diastolic", v_lower.get("diastolic_bp", v_lower.get("bloodpressure", v_lower.get("bp", old_val))))
    if f_lower in ["bloodpressure", "bp"]:
        return v_lower.get("bloodpressure", v_lower.get("bp", v_lower.get("diastolic_bp", v_lower.get("diastolic", v_lower.get("trestbps", old_val)))))

    # Glucose mappings
    if f_lower in ["glucose", "avg_glucose_level"]:
        val = v_lower.get("glucose", v_lower.get("avg_glucose_level"))
        if val is not None:
            if f_lower == "glucose" and old_val in [1, 2, 3]:
                return 1 if val < 100 else (2 if val < 126 else 3)
            return val
        return old_val

    # Cholesterol mappings
    if f_lower in ["chol", "cholesterol"]:
        val = v_lower.get("chol", v_lower.get("cholesterol"))
        if val is not None:
            if f_lower == "cholesterol" and old_val in [1, 2, 3]:
                return 1 if val < 200 else (2 if val < 240 else 3)
            return val
        return old_val

    # Pulse / Heart Rate mappings
    if f_lower in ["pulse", "thalach", "heartrate"]:
        return v_lower.get("pulse", v_lower.get("thalach", v_lower.get("heartrate", old_val)))

    # BMI / Height / Weight
    if f_lower == "bmi":
        return v_lower.get("bmi", old_val)
    if f_lower == "height":
        return v_lower.get("height", old_val)
    if f_lower == "weight":
        return v_lower.get("weight", old_val)

    # Diabetes specific
    if f_lower == "pregnancies":
        return v_lower.get("pregnancies", old_val)
    if f_lower == "skinthickness":
        return v_lower.get("skinthickness", old_val)
    if f_lower == "insulin":
        return v_lower.get("insulin", old_val)
    if f_lower == "diabetespedigreefunction":
        return v_lower.get("diabetespedigreefunction", old_val)

    # Stroke / Cardiovascular history
    if f_lower == "hypertension":
        return v_lower.get("hypertension", old_val)
    if f_lower == "heart_disease":
        return v_lower.get("heart_disease", old_val)
    if f_lower == "ever_married":
        return v_lower.get("ever_married", old_val)
    if f_lower == "residence_type":
        return v_lower.get("residence_type", old_val)
    if f_lower == "smoking_status":
        return v_lower.get("smoking_status", old_val)
    if f_lower == "smoke":
        smk = v_lower.get("smoking_status", v_lower.get("smoke"))
        if smk is not None:
            return 1 if smk > 0 else 0
        return old_val
    if f_lower == "alcohol":
        return v_lower.get("alcohol", old_val)
    if f_lower == "active":
        return v_lower.get("active", old_val)

    # Heart exam specifics
    if f_lower == "cp":
        return v_lower.get("cp", old_val)
    if f_lower == "fbs":
        return v_lower.get("fbs", old_val)
    if f_lower == "exang":
        return v_lower.get("exang", old_val)
    if f_lower == "oldpeak":
        return v_lower.get("oldpeak", old_val)
    if f_lower == "ca":
        return v_lower.get("ca", old_val)

    # Liver & Anemia
    if f_lower in ["total_bilirubin", "bilirubin"]:
        return v_lower.get("total_bilirubin", v_lower.get("bilirubin", old_val))
    if f_lower in ["hemoglobin", "hb"]:
        return v_lower.get("hemoglobin", v_lower.get("hb", old_val))

    return old_val
