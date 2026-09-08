import re
from typing import Dict, Any, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AegisNLPAssistant:
    """
    Advanced Clinical Conversational AI powered by TF-IDF Semantic Intent Matching,
    Medical Ontology lookup, and Contextual Recommendations.
    """

    KNOWLEDGE_BASE = [
        # 1. Greetings & Meta
        {
            "intent": "greeting",
            "patterns": [
                "hello", "hi", "hey there", "greetings", "good morning", "good evening",
                "who are you", "what can you do", "introduce yourself", "help me"
            ],
            "response": (
                "**Greetings! I am Aegis AI**, your clinical intelligence assistant.\n\n"
                "I can assist you with:\n"
                "- 🩺 **Multi-Disease Assessment Guidance** (Diabetes, Heart, Stroke, Kidney, Liver, etc.)\n"
                "- 📋 **Clinical Note Parsing** & automated vital extraction\n"
                "- 📊 **Biomarker Interpretation** (e.g. ECG, glucose, cholesterol, GFR, bilirubin)\n"
                "- 🚨 **Symptom Triage** & clinical risk scoring\n\n"
                "How may I support your clinical evaluation today?"
            )
        },
        # 2. Heart Disease & Cardiology
        {
            "intent": "heart_disease",
            "patterns": [
                "heart disease", "chest pain", "angina", "cardiovascular risk", "high cholesterol",
                "st depression", "resting ecg", "max heart rate", "thalassemia", "heart attack",
                "shortness of breath on exertion", "cardiac symptoms", "myocardial infarction"
            ],
            "response": (
                "🫀 **Cardiovascular Disease Evaluation**\n\n"
                "Our neural engine evaluates key cardiac markers:\n"
                "- **Resting BP (`trestbps`)**: Normal < 120 mmHg. Stage 1 HTN >= 130 mmHg.\n"
                "- **Serum Cholesterol (`chol`)**: Desirable < 200 mg/dL.\n"
                "- **Maximum Heart Rate (`thalach`)** & **Exercise Angina (`exang`)**\n"
                "- **ST Depression (`oldpeak`)** & **Slope of peak exercise ST segment**\n\n"
                "👉 Navigate to the **[Heart Disease Module](/analysis/heart)** to run real-time inference on patient metrics."
            )
        },
        # 3. Diabetes & Endocrinology
        {
            "intent": "diabetes",
            "patterns": [
                "diabetes", "blood sugar", "fasting glucose", "hba1c", "insulin resistance",
                "polyuria", "excessive thirst", "polydipsia", "skin thickness", "diabetes pedigree",
                "high blood sugar", "type 2 diabetes", "hyperglycemia"
            ],
            "response": (
                "🩸 **Diabetes & Metabolic Health Screening**\n\n"
                "Our predictive model analyzes 8 diagnostic parameters:\n"
                "- **Fasting Glucose**: Normal < 100 mg/dL; Prediabetes: 100–125; Diabetes >= 126 mg/dL.\n"
                "- **Serum Insulin**: Evaluates beta-cell function and insulin resistance.\n"
                "- **BMI & Skinfold Thickness**: Indicators of visceral adiposity.\n"
                "- **Diabetes Pedigree Function**: Quantifies genetic hereditary risk score.\n\n"
                "👉 Test patient data directly in the **[Diabetes Module](/analysis/diabetes)**."
            )
        },
        # 4. Stroke & Cerebrovascular
        {
            "intent": "stroke",
            "patterns": [
                "stroke risk", "brain attack", "cerebrovascular accident", "facial drooping",
                "slurred speech", "sudden weakness", "arm numbness", "tia", "transient ischemic attack",
                "average glucose stroke", "stroke symptoms fast"
            ],
            "response": (
                "🧠 **Stroke Risk & Cerebrovascular Assessment**\n\n"
                "⚠️ **Emergency FAST Protocol**:\n"
                "- **F**ace drooping | **A**rm weakness | **S**peech difficulty | **T**ime to call emergency!\n\n"
                "For longitudinal risk calculation, our **[Stroke Module](/analysis/stroke)** weighs:\n"
                "- Hypertension history, cardiac comorbidity, age\n"
                "- Serum glucose spikes & BMI metrics\n"
                "- Smoking status and lifestyle indicators."
            )
        },
        # 5. Kidney / Renal Pathology
        {
            "intent": "kidney",
            "patterns": [
                "kidney disease", "renal failure", "creatinine", "gfr", "blood in urine",
                "proteinuria", "foamy urine", "swollen feet edema", "flank pain", "chronic kidney disease"
            ],
            "response": (
                "🧪 **Kidney Function & Renal Diagnostics**\n\n"
                "Renal integrity is critical for systemic metabolic equilibrium. Key indicators:\n"
                "- **Blood Pressure & Glomerular Filtration**: Elevated pressure damages delicate nephron capillaries.\n"
                "- **Proteinuria / Albuminuria**: Early markers of renal microvascular damage.\n\n"
                "👉 Evaluate renal markers in the **[Kidney Disease Module](/analysis/kidney)**."
            )
        },
        # 6. Liver / Hepatic Health
        {
            "intent": "liver",
            "patterns": [
                "liver disease", "jaundice", "yellow eyes", "bilirubin", "ast alt enzymes",
                "hepatic screen", "cirrhosis", "fatty liver", "dark urine pale stool"
            ],
            "response": (
                "🧫 **Hepatic Function & Liver Screening**\n\n"
                "The **[Liver Disease Module](/analysis/liver)** screens for biliary obstruction and hepatocellular injury by evaluating:\n"
                "- **Total Bilirubin**: Normal is typically 0.2–1.2 mg/dL. Elevated levels cause jaundice.\n"
                "- **Enzyme Markers** and patient demographic profiles."
            )
        },
        # 7. Anemia & Hematology
        {
            "intent": "anemia",
            "patterns": [
                "anemia", "low hemoglobin", "pale skin", "chronic fatigue", "iron deficiency",
                "dizziness weakness", "oxygen carrying capacity", "rbc red blood cells"
            ],
            "response": (
                "🩸 **Hematological & Anemia Analysis**\n\n"
                "Symptoms of fatigue, pallor, and exertional dyspnea often stem from inadequate oxygenation.\n"
                "- **Standard Hemoglobin Ranges**:\n"
                "  - Adult Males: 13.8 to 17.2 g/dL\n"
                "  - Adult Females: 12.1 to 15.1 g/dL\n\n"
                "👉 Screen hemoglobin levels in the **[Anemia Module](/analysis/anemia)**."
            )
        },
        # 8. Obesity & Body Composition
        {
            "intent": "obesity",
            "patterns": [
                "obesity", "overweight", "bmi index", "body mass index", "weight height ratio",
                "metabolic syndrome", "bariatric risk"
            ],
            "response": (
                "⚖️ **Obesity & Body Mass Analysis**\n\n"
                "- **BMI Classification**:\n"
                "  - Normal: 18.5 – 24.9\n"
                "  - Overweight: 25.0 – 29.9\n"
                "  - Obesity Class I: 30.0 – 34.9\n"
                "  - Severe Obesity (Class II/III): >= 35.0\n\n"
                "👉 Calculate index classifications in the **[Obesity Module](/analysis/obesity)**."
            )
        },
        # 9. Clinical NLP Notes Parser
        {
            "intent": "nlp_parser_help",
            "patterns": [
                "how to parse notes", "extract vitals from text", "nlp parser", "natural language processing",
                "doctor notes", "clinical text extraction", "paste lab report"
            ],
            "response": (
                "📝 **Clinical NLP Note Parser Feature**\n\n"
                "You can paste unstructured doctor notes, discharge summaries, or patient dialogues into our NLP workspace.\n\n"
                "**Example Input**:\n"
                "> *'54 yo male presenting with BP 145/90, glucose 165 mg/dL, complaining of exertion-related chest tightness and fatigue.'*\n\n"
                "⚡ **Our NLP engine extracts**:\n"
                "- `Age`: 54 | `Gender`: Male\n"
                "- `BloodPressure`: 90 | `trestbps`: 145\n"
                "- `Glucose`: 165 | `Symptoms`: Chest tightness, fatigue\n"
                "- **Triage Urgency** & **Auto-Fill Actions** directly into the disease forms!"
            )
        },
        # 10. Reports & Documentation
        {
            "intent": "report_generation",
            "patterns": [
                "how to generate report", "download pdf", "medical report", "export results",
                "clinical documentation", "print report"
            ],
            "response": (
                "📄 **Medical Report Generation**\n\n"
                "To generate a synthesized clinical diagnostic report:\n"
                "1. Open any diagnostic module (e.g. Heart or Diabetes).\n"
                "2. Click **'Analyze Clinical Data'** to run neural inference.\n"
                "3. Click **'Generate Medical Report'** to download a formatted, timestamped clinical PDF synthesis containing risk indices, vital radar breakdowns, and AI physician notes."
            )
        },
        # 11. Blood Pressure & Hypertension
        {
            "intent": "blood_pressure",
            "patterns": [
                "blood pressure", "hypertension", "systolic diastolic", "high bp",
                "bp reading meaning", "normal bp range"
            ],
            "response": (
                "🩺 **Blood Pressure Staging Standards (AHA/ACC)**:\n\n"
                "- **Normal**: Systolic < 120 AND Diastolic < 80 mmHg\n"
                "- **Elevated**: Systolic 120-129 AND Diastolic < 80 mmHg\n"
                "- **Stage 1 HTN**: Systolic 130-139 OR Diastolic 80-89 mmHg\n"
                "- **Stage 2 HTN**: Systolic >= 140 OR Diastolic >= 90 mmHg\n"
                "- **Hypertensive Crisis**: Systolic > 180 and/or Diastolic > 120 mmHg (Immediate medical care needed)."
            )
        }
    ]

    def __init__(self):
        self._corpus = []
        self._intents = []
        for item in self.KNOWLEDGE_BASE:
            text_corpus = " ".join(item["patterns"]) + " " + item["intent"]
            self._corpus.append(text_corpus)
            self._intents.append(item)

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self._corpus)

    def answer_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process user query using TF-IDF cosine similarity, keyword heuristics,
        and clinical fallback triage.
        """
        if not user_query or not user_query.strip():
            return {
                "response": "Please enter a clinical question, symptom, or metric to analyze.",
                "intent": "empty",
                "confidence": 0.0
            }

        # Vector matching
        query_vec = self.vectorizer.transform([user_query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])

        # If similarity is solid (>= 0.15), return matching knowledge
        if best_score >= 0.15:
            matched = self._intents[best_idx]
            return {
                "response": matched["response"],
                "intent": matched["intent"],
                "confidence": round(best_score, 3)
            }

        # Fallback with clinical NLP symptom detection
        from .clinical_parser import parse_clinical_text
        parsed = parse_clinical_text(user_query)

        if parsed.get("symptoms"):
            rec_mods = [f"**[{m['id'].title()} Module](/analysis/{m['id']})**" for m in parsed.get("recommended_modules", [])]
            mods_str = ", ".join(rec_mods) if rec_mods else "our diagnostic suite"
            return {
                "response": (
                    f"🔍 **Clinical Symptoms Detected in Query**:\n\n"
                    f"- Identified: *{', '.join([s for lst in parsed['symptoms'].values() for s in lst])}*\n"
                    f"- Triage Level: **{parsed['triage']['severity']}**\n"
                    f"- Recommended Pathway: {mods_str}\n\n"
                    f"You can paste complete notes or navigate directly to the recommended module for precise risk modeling."
                ),
                "intent": "symptom_detected_fallback",
                "confidence": 0.5,
                "parsed_data": parsed
            }

        # General intelligent medical response
        return {
            "response": (
                "I've logged your query. As the Aegis Clinical Intelligence Assistant, I can assist with:\n"
                "- Interpreting specific biomarkers (e.g. *'What is normal glucose?'*, *'Explain ST depression'*)\n"
                "- Navigating to disease modules (Diabetes, Heart, Stroke, Kidney, Liver, Anemia, Obesity)\n"
                "- Parsing unformatted doctor notes into structured clinical features.\n\n"
                "Please specify your symptom, metric, or desired diagnostic module!"
            ),
            "intent": "general_guidance",
            "confidence": round(best_score, 3)
        }

# Global singleton
_assistant_instance = AegisNLPAssistant()
def get_ai_response(user_query: str) -> Dict[str, Any]:
    return _assistant_instance.answer_query(user_query)
