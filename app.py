import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# Step 1: Load and prepare the dataset
data = pd.read_excel('data.xlsx')

# Features and target
X = data[['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10',
          'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'perc', 'transformer']]
y = data['thd']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train the RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit app title and description
st.title("Total Harmonic Distortion Prediction App")
st.markdown("#### Predict the Total Harmonic Distortion (THD) and Design a Filter Circuit")
st.markdown("#### Voltage level = 0.415KV")

# Initialize session state for storing values if not already present
if 'Ppcc' not in st.session_state:
    st.session_state['Ppcc'] = 0.0
if 'Qpcc' not in st.session_state:
    st.session_state['Qpcc'] = 0.0
if 'Spcc' not in st.session_state:
    st.session_state['Spcc'] = 0.0

# Sidebar to navigate between different pages
page = st.sidebar.selectbox("Select Page", ["THD Prediction", "Design Filter Circuit for THD"])

if page == "THD Prediction":
    # Input fields with updated labels
    st.header("Input Appliance Details")
    P_values = []  # Active power (P) for each appliance
    Q_values = []  # Reactive power (Q) for each appliance

    appliance_names = ["Television", "Compact Fluorescent Lamp", "Refrigerator", "Air Conditioner", "Phones", 
                       "Laptops", "Fans", "Washing Machines", "Microwave", "Personal Computers", "UPS",
                       "Fluorescent Lamp", "Hot Plate", "Iron", "Electric Cooker", "Printer", "Photocopier", "Linear Load"]

    for i in range(0, len(appliance_names), 4):
        cols = st.columns(4)  # Create 4 columns

        for j in range(4):
            if i + j < len(appliance_names):
                with cols[j]:
                    appliance_name = appliance_names[i + j]
                    checkbox = st.checkbox(appliance_name)
                    
                    if checkbox:
                        num_devices = st.number_input(f"Number of {appliance_name}:", value=0, min_value=0, help="Enter the number of devices")
                        wattage = st.number_input(f"Wattage of {appliance_name} (KW):", value=0.0, min_value=0.0, help="Enter wattage in KW")  
                        power_factor = st.number_input(f"Power Factor of {appliance_name}:", min_value=0.0, max_value=1.0, value=1.0, step=0.01, help="Enter power factor (0 to 1)")  
                    else:
                        num_devices = 0.0
                        wattage = 0.0
                        power_factor = 1.0

                    # Calculate P (Active Power) for each appliance
                    P = num_devices * wattage
                    P_values.append(P)

                    # Calculate Q (Reactive Power) for each appliance
                    Q = np.tan(np.arccos(power_factor)) * wattage * num_devices
                    Q_values.append(Q)

    # Calculate P_total (Total Active Power) and Q_total (Total Reactive Power)
    Ppcc = sum(P_values)
    Qpcc = sum(Q_values)

    # Calculate S (Apparent Power)
    Spcc = np.sqrt(Ppcc**2 + Qpcc**2)

    # Store calculated values in session state
    st.session_state['Ppcc'] = Ppcc
    st.session_state['Qpcc'] = Qpcc
    st.session_state['Spcc'] = Spcc

    # Display the calculated P_total, Q_total, and S
    st.markdown("### Calculated Power Values:")
    st.info(f"**Total Active Power (P_total):** {Ppcc:.2f} KW")
    st.info(f"**Total Reactive Power (Q_total):** {Qpcc:.2f} KVAR")
    st.info(f"**Apparent Power (S):** {Spcc:.2f} KVA")

    # Input field for transformer rating
    trans = st.number_input('Transformer Ratings (KVA):', value=1, min_value=1, help="Enter transformer rating in KVA")  

    # Calculate 'perc' feature using P_total
    perc = Ppcc / (trans * 8)

    # Combine inputs into the feature list
    features = P_values + [perc, trans]

    # Predict THD and display result
    if st.button('Predict THD'):
        prediction = model.predict([features])[0]
        
        # Display prediction result in a colored box
        st.success(f"**Predicted THD:** {prediction:.2f}")

        # Show instructions to switch to the next page
        st.info("Switch to the 'Design Filter Circuit for THD' page in the sidebar for further calculations.")

elif page == "Design Filter Circuit for THD":
    st.header("Step 2: Design Filter Circuit for THD")

    # Retrieve stored values from session state
    Ppcc = st.session_state['Ppcc']
    Qpcc = st.session_state['Qpcc']
    Spcc = st.session_state['Spcc']
    
    # Calculate PFpcc
    PFpcc = Ppcc / Spcc
    
    # Calculate Qf
    Qf = Ppcc * (np.tan(np.arccos(PFpcc) - np.arccos(1)))
    
    # Calculate Xc
    Vpcc = 0.415  # Voltage level
    Xc = 0.240 / (Qf / 3)
    
    # Calculate C
    C = 1 / (314.29 * Xc)
    
    # Calculate Xl
    Xl = Xc / 49
    
    # Calculate L
    L = Xl / (314.29)
    
    # Display calculated values using LaTeX formatting for scientific notation
    st.markdown("### Filter Circuit Parameters:")
    st.latex(f"Q_f = {Qf:.2f} KVar") 
    st.latex(f"X_c = {Xc:.10f} Ω")
    st.latex(f"C =  {C:.10f} F")
    st.latex(f"X_l = {Xl:.10f} Ω")
    st.latex(f"L = {L:.10f} H")
    st.latex(f"Q\\text{{-factor}} = 40")
