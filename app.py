# app.py

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load model & scaler
model = joblib.load('model/dropout_ann_model.joblib')
scaler = joblib.load('model/scaler.joblib')

# Try to load metadata for better default values and ranges
try:
    import json
    with open('model/metadata.json', 'r') as f:
        metadata = json.load(f)
        default_values = metadata.get('default_values', {})
        feature_ranges = metadata.get('feature_ranges', {})
except:
    print("Warning: Could not load metadata.json. Using fallback defaults.")
    metadata = None
    feature_ranges = {}
    # Load a sample from the training data to get default values
    try:
        sample_data = pd.read_csv('data/student-dropout.csv')
        default_values = sample_data.mean().to_dict()
    except:
        # If we can't load the data, use some reasonable defaults
        default_values = {}
MARITAL_STATUS_MAP = {
    "Single": 1,
    "Married": 2,
    "Widowed": 3,
    "Divorced": 4,
    "Facto union": 5,
    "Legally separated": 6
}

APPLICATION_MODE_MAP = {
    "1st phase - general contingent": 1,
    "Ordinance No. 612/93": 2,
    "1st phase - special contingent (Azores Island)": 5,
    "Holders of other higher courses": 7,
    "Ordinance No. 854-B/99": 10,
    "International student (bachelor)": 15,
    "1st phase - special contingent (Madeira Island)": 16,
    "2nd phase - general contingent": 17,
    "3rd phase - general contingent": 18,
    "Ordinance No. 533-A/99, item b2) (Different Plan)": 26,
    "Ordinance No. 533-A/99, item b3 (Other Institution)": 27,
    "Over 23 years old": 39,
    "Transfer": 42,
    "Change of course": 43,
    "Technological specialization diploma holders": 44,
    "Change of institution/course": 51,
    "Short cycle diploma holders": 53,
    "Change of institution/course (International)": 57
}

COURSE_MAP = {
    "Biofuel Production Technologies": 33,
    "Animation and Multimedia Design": 171,
    "Social Service (evening attendance)": 8014,
    "Agronomy": 9003,
    "Communication Design": 9070,
    "Veterinary Nursing": 9085,
    "Informatics Engineering": 9119,
    "Equiniculture": 9130,
    "Management": 9147,
    "Social Service": 9238,
    "Tourism": 9254,
    "Nursing": 9500,
    "Oral Hygiene": 9556,
    "Advertising and Marketing Management": 9670,
    "Journalism and Communication": 9773,
    "Basic Education": 9853,
    "Management (evening attendance)": 9991
}

ATTENDANCE_MAP = {
    "Daytime": 1,
    "Evening": 0
}

PREVIOUS_QUALIFICATION_MAP = {
    "Secondary education": 1,
    "Higher education - bachelor's degree": 2,
    "Higher education - degree": 3,
    "Higher education - master's": 4,
    "Higher education - doctorate": 5,
    "Frequency of higher education": 6,
    "12th year of schooling - not completed": 9,
    "11th year of schooling - not completed": 10,
    "Other - 11th year of schooling": 11,
    "10th year of schooling": 12,
    "10th year of schooling - not completed": 14,
    "Basic education 3rd cycle (9th/10th/11th year) or equiv.": 18,
    "Basic education 2nd cycle (6th/7th/8th year) or equiv.": 19,
    "Technological specialization course": 22,
    "Higher education - degree (1st cycle)": 26,
    "Professional higher technical course": 27,
    "Higher education - master (2nd cycle)": 29,
    "Higher education - doctorate (3rd cycle)": 30,
    "Higher education - degree (2nd cycle)": 34,
    "Higher education - doctorate (3rd cycle)": 35,
    "Higher education - master integrated (2nd cycle)": 36,
    "Higher education - master (2nd cycle)": 37,
    "Higher education - diploma": 38,
    "Technological specialization course": 39,
    "Higher education - degree (1st cycle)": 40,
    "Higher education - diploma (1st cycle)": 41,
    "Professional higher technical course": 42,
    "Higher education - master (2nd cycle)": 43,
    "Higher education - doctorate (3rd cycle)": 44
}

NACIONALITY_MAP = {
    "Portuguese": 1,
    "German": 2,
    "Spanish": 6,
    "Italian": 11,
    "Dutch": 13,
    "English": 14,
    "Lithuanian": 17,
    "Angolan": 21,
    "Cape Verdean": 22,
    "Guinean": 24,
    "Mozambican": 25,
    "Santomean": 26,
    "Turkish": 32,
    "Brazilian": 41,
    "Romanian": 62,
    "Moldova (Republic of)": 100,
    "Mexican": 101,
    "Ukrainian": 103,
    "Russian": 105,
    "Cuban": 108,
    "Colombian": 109
}

PARENTS_QUALIFICATION_MAP = {
    "Secondary Education - 12th Year or Eq.": 1,
    "Higher Education - Bachelor's Degree": 2,
    "Higher Education - Degree": 3,
    "Higher Education - Master's": 4,
    "Higher Education - Doctorate": 5,
    "Frequency of Higher Education": 6,
    "12th Year - Not Completed": 9,
    "11th Year - Not Completed": 10,
    "7th Year (Old)": 11,
    "Other - 11th Year": 12,
    "2nd year complementary high school course": 13,
    "10th Year": 14,
    "General commerce course": 18,
    "Basic Education 3rd Cycle or Equiv.": 19,
    "Complementary High School Course": 20,
    "Technical-professional course": 22,
    "Complementary High School Course - not concluded": 25,
    "7th year of schooling": 26,
    "2nd cycle of general high school course": 27,
    "9th Year - Not Completed": 29,
    "8th year of schooling": 30,
    "General Course of Administration and Commerce": 31,
    "Supplementary Accounting and Administration": 33,
    "Unknown": 34,
    "Can't read or write": 35,
    "Can read without having a 4th year": 36,
    "Basic education 1st cycle (4th/5th year) or equiv.": 37,
    "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.": 38,
    "Technological specialization course": 39,
    "Higher education - degree (1st cycle)": 40,
    "Specialized higher studies course": 41,
    "Professional higher technical course": 42,
    "Higher education - master (2nd cycle)": 43,
    "Higher education - doctorate (3rd cycle)": 44
}

PARENTS_OCCUPATION_MAP = {
    "Student": 0,
    "Representatives of the Legislative Power and Executive Bodies": 1,
    "Specialists in Intellectual and Scientific Activities": 2,
    "Intermediate Level Technicians and Professions": 3,
    "Administrative staff": 4,
    "Personal Services, Security and Safety Workers": 5,
    "Farmers and Skilled Workers in Agriculture and Fisheries": 6,
    "Skilled Workers in Industry, Construction and Craftsmen": 7,
    "Installation and Machine Operators and Assembly Workers": 8,
    "Unskilled Workers": 9,
    "Armed Forces Professions": 10,
    "Other Situation": 90,
    "(blank)": 99,
    "Health professionals": 101,
    "Teachers": 102,
    "Specialists in ICT": 103,
    "Intermediate level science and engineering technicians": 104,
    "Intermediate level health technicians": 105,
    "Intermediate level technicians from legal, social, sports, cultural areas": 106,
    "Office workers, secretaries in general and data processing operators": 107,
    "Data, accounting, statistical, financial services and registry-related operators": 108,
    "Other administrative support staff": 109,
    "Personal service workers": 110,
    "Sellers": 111,
    "Personal care workers and the like": 112,
    "Protection and security services personnel": 113,
    "Market-oriented farmers and skilled agricultural and animal production workers": 114,
    "Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence": 115,
    "Skilled construction workers and the like": 116,
    "Skilled workers in metallurgy, metalworking and similar": 117,
    "Skilled workers in electricity and electronics": 118,
    "Workers in food processing, woodworking, clothing": 119,
    "Skilled workers in printing, precision instrument manufacturing": 120,
    "Other skilled workers in industry and crafts": 121,
    "Fixed plant and machine operators": 122,
    "Assembly workers": 123,
    "Vehicle drivers and mobile equipment operators": 124,
    "Unskilled workers in agriculture, animal production, fisheries and forestry": 125,
    "Unskilled workers in extractive industry, construction, manufacturing": 126,
    "Meal preparation assistants": 127,
    "Street vendors and street service providers": 128,
    "Garbage collectors and other elementary workers": 129,
    "Armed Forces Officers": 130,
    "Armed Forces Sergeants": 131,
    "Other Armed Forces personnel": 132,
    "Directors of administrative and commercial services": 133,
    "Hotel, catering, trade and other services directors": 134,
    "Specialists in the physical sciences, mathematics, engineering": 135,
    "Specialists in the life sciences and health professionals": 136,
    "Teaching professionals": 137,
    "Specialists in finance, accounting, administrative organization": 138,
    "Legal, social and cultural specialists": 139,
    "Information and communication technology technicians": 140,
    "Intermediate level technicians and professions of the legal, social services": 141,
    "Technical support staff": 142,
    "Accounting and financial services administrative assistants": 143,
    "Administrative support staff for material registration": 144,
    "Other personal services workers": 145,
    "Workers specialized in personal security and protection services": 146,
    "Market-oriented farmers and skilled workers in animal production": 147,
    "Market-oriented producers and skilled workers in forestry and floriculture": 148,
    "Market-oriented farmers and skilled workers in mixed farms": 149,
    "Bricklayers and the like": 150,
    "Building finishers and the like": 151,
    "Painters, building cleaners and the like": 152,
    "Blacksmiths, toolmakers and the like": 153,
    "Mechanics and repairers of machines and vehicles": 154,
    "Manual workers in handicrafts and the like": 155,
    "Workers of electrical installations": 156,
    "Workers in electronics and telecommunications": 157,
    "Food processing and related workers": 158,
    "Wood, cabinet makers, other funnel trades and the like": 159,
    "Clothing and footwear workers and the like": 160,
    "Workers of other industries and crafts": 161,
    "Mining and quarrying workers": 162,
    "Construction workers": 163,
    "Stationary machine and equipment operators": 164,
    "Manufacturing and assembly workers": 165,
    "Drivers of heavy vehicles and buses": 166,
    "Mobile equipment drivers": 167,
    "Sailors' deck crew and the like": 168,
    "Cleaners": 169,
    "Agricultural, forestry and fishery unskilled workers": 170,
    "Unskilled construction, mining and manufacturing workers": 171,
    "Food preparation assistants": 172,
    "Street vendors (except food) and street service providers": 173,
    "Waste collectors": 174,
    "Other unskilled workers in services and sales": 175,
    "Laborers in cargo handling": 176,
    "Packagers and manual manufacturing laborers": 177,
    "Elementary occupations not elsewhere classified": 178,
    "Armed Forces Officers": 179,
    "Other Armed Forces Sergeants": 180,
    "Other lower-ranking armed forces": 181,
    "Cleaning workers": 182,
    "Vehicle drivers and mobile equipment operators": 183,
    "Unskilled workers in agriculture, animal production, fisheries": 184,
    "Unskilled workers in extractive industry, construction": 185,
    "Street vendors (except food) and street service providers": 186,
    "Directors and executive managers": 187,
    "Manufacturing, mining, construction, and distribution directors": 188,
    "Information and communication technology service directors": 189,
    "Chief executives": 190,
    "Directors of catering services and similar": 191,
    "Sales, marketing and business development directors": 192,
    "Directors and managers of service branches not elsewhere classified": 193,
    "Accountants": 194,
    "Sales representatives": 195
}

GENDER_MAP = {
    "Male": 1,
    "Female": 0
}

YES_NO_MAP = {
    "Yes": 1,
    "No": 0
}

# Define feature names and their types
FEATURE_INFO = [
    ("Marital status", "dropdown", MARITAL_STATUS_MAP),
    ("Application mode", "dropdown", APPLICATION_MODE_MAP),
    ("Application order", "number", None),
    ("Course", "dropdown", COURSE_MAP),
    ("Daytime/evening attendance", "dropdown", ATTENDANCE_MAP),
    ("Previous qualification", "dropdown", PREVIOUS_QUALIFICATION_MAP),
    ("Previous qualification (grade)", "number", None),
    ("Nacionality", "dropdown", NACIONALITY_MAP),
    ("Mother's qualification", "dropdown", PARENTS_QUALIFICATION_MAP),
    ("Father's qualification", "dropdown", PARENTS_QUALIFICATION_MAP),
    ("Mother's occupation", "dropdown", PARENTS_OCCUPATION_MAP),
    ("Father's occupation", "dropdown", PARENTS_OCCUPATION_MAP),
    ("Admission grade", "number", None),
    ("Displaced", "dropdown", YES_NO_MAP),
    ("Educational special needs", "dropdown", YES_NO_MAP),
    ("Debtor", "dropdown", YES_NO_MAP),
    ("Tuition fees up to date", "dropdown", YES_NO_MAP),
    ("Gender", "dropdown", GENDER_MAP),
    ("Scholarship holder", "dropdown", YES_NO_MAP),
    ("Age at enrollment", "number", None),
    ("International", "dropdown", YES_NO_MAP),
    ("Curricular units 1st sem (credited)", "number", None),
    ("Curricular units 1st sem (enrolled)", "number", None),
    ("Curricular units 1st sem (evaluations)", "number", None),
    ("Curricular units 1st sem (approved)", "number", None),
    ("Curricular units 1st sem (grade)", "number", None),
    ("Curricular units 1st sem (without evaluations)", "number", None),
    ("Curricular units 2nd sem (credited)", "number", None),
    ("Curricular units 2nd sem (enrolled)", "number", None),
    ("Curricular units 2nd sem (evaluations)", "number", None),
    ("Curricular units 2nd sem (approved)", "number", None),
    ("Curricular units 2nd sem (grade)", "number", None),
    ("Curricular units 2nd sem (without evaluations)", "number", None),
    ("Unemployment rate", "number", None),
    ("Inflation rate", "number", None),
    ("GDP", "number", None)
]

# Load a sample from the training data to get default values
try:
    sample_data = pd.read_csv('data/student-dropout.csv')
    default_values = sample_data.mean().to_dict()
except:
    # If we can't load the data, use some reasonable defaults
    default_values = {}

def predict(*inputs):
    # Convert inputs to appropriate format
    features = []
    
    for i, (name, input_type, mapping) in enumerate(FEATURE_INFO):
        value = inputs[i]
        
        if value is None or value == "":
            # Use default value if available, otherwise use median/mode
            if name in default_values:
                features.append(default_values[name])
            else:
                # Use reasonable defaults based on the feature type
                if input_type == "dropdown" and mapping:
                    # Use the most common value (first in the list)
                    features.append(list(mapping.values())[0])
                else:
                    # Use 0 for numeric features
                    features.append(0)
        else:
            if input_type == "dropdown" and mapping:
                # Convert dropdown selection to numeric code
                features.append(mapping[value])
            else:
                # Use numeric value directly
                features.append(float(value))
    
    # Convert to numpy array and reshape
    X = np.array(features).reshape(1, -1)
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    pred = model.predict(X_scaled)[0]
    pred_proba = model.predict_proba(X_scaled)[0]
    
    # Create result with probabilities
    labels = ['Dropout', 'Enrolled', 'Graduate']
    result = f"**Predicted Status: {labels[pred]}**\n\n"
    result += "Probabilities:\n"
    for label, prob in zip(labels, pred_proba):
        result += f"- {label}: {prob:.2%}\n"
    
    return result

# Create Gradio inputs
inputs = []
for name, input_type, mapping in FEATURE_INFO:
    if input_type == "dropdown":
        # Add "Not specified" option for optional fields
        choices = ["Not specified"] + list(mapping.keys())
        inputs.append(gr.Dropdown(
            choices=choices,
            label=name,
            value="Not specified",
            info="Select 'Not specified' if unknown"
        ))
    else:
        # For numerical inputs, set appropriate constraints
        if "grade" in name.lower():
            # Grades typically have a range (0-200 based on UCI dataset)
            # Use actual max from data if available
            max_val = feature_ranges.get(name, {}).get('max', 200)
            inputs.append(gr.Number(
                label=name,
                value=None,
                minimum=0,
                maximum=max_val,
                info=f"Grade between 0-{max_val:.0f} (leave empty if unknown)"
            ))
        elif "age" in name.lower():
            # Age should be reasonable
            max_val = feature_ranges.get(name, {}).get('max', 100)
            inputs.append(gr.Number(
                label=name,
                value=None,
                minimum=0,
                maximum=max_val,
                step=1,
                info=f"Age in years (leave empty if unknown)"
            ))
        elif "curricular units" in name.lower():
            # Curricular units are counts, should be integers
            max_val = feature_ranges.get(name, {}).get('max', 50)
            inputs.append(gr.Number(
                label=name,
                value=None,
                minimum=0,
                maximum=max_val,
                step=1,
                info=f"Number of units 0-{max_val:.0f} (leave empty if unknown)"
            ))
        elif "rate" in name.lower():
            # Rates can be decimal values
            max_val = feature_ranges.get(name, {}).get('max', 100)
            inputs.append(gr.Number(
                label=name,
                value=None,
                minimum=0,
                maximum=max_val,
                info=f"Rate 0-{max_val:.1f}% (leave empty if unknown)"
            ))
        elif "gdp" in name.lower():
            # GDP can have large values
            max_val = feature_ranges.get(name, {}).get('max', 100000)
            inputs.append(gr.Number(
                label=name,
                value=None,
                minimum=0,
                info="GDP value (leave empty if unknown)"
            ))
        elif "order" in name.lower():
            # Application order should be a positive integer
            max_val = feature_ranges.get(name, {}).get('max', 10)
            inputs.append(gr.Number(
                label=name,
                value=None,
                minimum=0,
                maximum=max_val,
                step=1,
                info=f"Order of application 0-{max_val:.0f} (leave empty if unknown)"
            ))
        else:
            # Default for other numerical fields
            max_val = feature_ranges.get(name, {}).get('max', None)
            if max_val:
                inputs.append(gr.Number(
                    label=name,
                    value=None,
                    minimum=0,
                    maximum=max_val,
                    step=1,
                    info=f"Value 0-{max_val:.0f} (leave empty if unknown)"
                ))
            else:
                inputs.append(gr.Number(
                    label=name,
                    value=None,
                    minimum=0,
                    step=1,
                    info="Leave empty if unknown (min: 0)"
                ))

# Create the interface
demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Textbox(label="Prediction Result"),
    title="EduShield UG: Student Dropout Risk Predictor",
    description="""This app predicts the risk of student dropout using an Artificial Neural Network trained on the UCI student dropout dataset.
    
    **Note:** You don't need to fill in all fields. The model will use default values for any missing information, though accuracy may be reduced with fewer inputs.""",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
