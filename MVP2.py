import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim

# Standard coordinates for St. Gallen
default_lat, default_lon = 47.424482, 9.376717

# Initialize session state variables
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'address' not in st.session_state:
    st.session_state.address = ""

# Define functions to handle navigation
def go_to_next_step():
    st.session_state.current_step += 1

def go_to_previous_step():
    st.session_state.current_step -= 1

# Backend Code: Data Preprocessing and Model Training
def preprocess_and_train():
    # Load the dataset (replace with your actual file path)
    real_estate_data = pd.read_excel('real-estate-scraped-data.xlsx')

    # Data Preprocessing
    # Define the function to split 'Col3'
    def split_column(row):
        parts = row.split(' • ')
        property_type = parts[0]
        rooms = parts[1] if len(parts) > 1 else None
        size_m2 = parts[2] if len(parts) > 2 else None
        return {'Property_Type': property_type, 'Rooms': rooms, 'Size_m2': size_m2}

    # Apply the function to each row in 'Col3' and create a new DataFrame
    split_data = real_estate_data['Col3'].apply(split_column).apply(pd.Series)

    # Cleaning and renaming columns
    real_estate_data['area_code'] = real_estate_data['Col4'].str.extract(r'\b(\d{4})\b')

    # Extracting numeric values from 'Col5' and 'Col6'
    real_estate_data['price_per_month'] = real_estate_data['Col5'].str.extract(r'(\d+[\’\']?\d*)')[0].str.replace("’", "").str.replace("'", "").str.strip()
    real_estate_data['price_per_m2_per_year'] = real_estate_data['Col6'].str.extract(r'(\d+[\’\']?\d*)')[0].str.replace("’", "").str.replace("'", "").str.strip()

    # Remove 'Zi.' from 'Rooms' and 'm²' from 'Size_m2', with checks for non-string data
    split_data['Rooms'] = split_data['Rooms'].str.replace(' Zi.', '').str.strip() if split_data['Rooms'].dtype == "object" else split_data['Rooms']
    split_data['Size_m2'] = split_data['Size_m2'].str.replace(' m²', '').str.strip() if split_data['Size_m2'].dtype == "object" else split_data['Size_m2']

    # Concatenate the new DataFrame with the original one, now including cleaned columns
    real_estate_data = pd.concat([split_data, real_estate_data.drop(columns=['Col3', 'Col4', 'Col5', 'Col6'])], axis=1)

    # Rearrange columns
    new_columns = ['Property_Type', 'Rooms', 'Size_m2', 'area_code', 'price_per_month', 'price_per_m2_per_year']
    real_estate_data = real_estate_data[new_columns]

    real_estate_data.dropna(inplace=True)

    # Convert columns to numeric as necessary
    real_estate_data['Rooms'] = pd.to_numeric(real_estate_data['Rooms'], errors='coerce')
    real_estate_data['Size_m2'] = pd.to_numeric(real_estate_data['Size_m2'], errors='coerce')
    real_estate_data['area_code'] = pd.to_numeric(real_estate_data['area_code'], errors='coerce')
    real_estate_data['price_per_month'] = pd.to_numeric(real_estate_data['price_per_month'], errors='coerce')

    # Drop any rows with NaN values
    real_estate_data.dropna(inplace=True)

    # Selecting features and target for the model
    X = real_estate_data[['Rooms', 'Size_m2', 'area_code']]  # Example features
    y = real_estate_data['price_per_month']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

def extract_zip_code(input_text):
    # Zerlegen des Strings anhand von Kommata oder Leerzeichen
    parts = input_text.replace(',', ' ').split()

    # Durchsuchen der Teile nach einer Zahlenfolge
    for part in parts:
        if part.isdigit() and len(part) == 4:  # Schweizer Postleitzahlen haben 4 Ziffern
            return part
    return None  # Keine gültige Postleitzahl gefunden

#NEWER VERSION PRICE PREDICT NOT MY AEREA SO NOT SURE
def predict_price(size_m2, extracted_zip_code, rooms, model):
    try:
        area_code = int(extracted_zip_code)
    except ValueError:
        st.error("Bitte geben Sie eine gültige Postleitzahl ein.")
        return None

    input_features = pd.DataFrame({
        'Rooms': [rooms],
        'Size_m2': [size_m2],
        'area_code': [area_code]  # Verwenden Sie hier den konvertierten numerischen Wert
    })
    predicted_price = model.predict(input_features)
    return predicted_price[0]


## Function to predict the price based on the model
#def predict_price(size_m2, area_code, rooms, model): OLD VERSION JUST KEPT IT FOR SECURITA REASONS
    input_features = pd.DataFrame({
        'Rooms': [rooms],
        'Size_m2': [size_m2],
        'area_code': [zip_code]
    })
    predicted_price = model.predict(input_features)
    return predicted_price[0]

def extract_zip_from_address(address):
    valid_st_gallen_zip_codes = ['9000', '9001', '9004', '9006', '9007', '9008', '9010', '9011', '9012', '9013', '9014', '9015', '9016', '9020', '9021', '9023', '9024', '9026', '9027', '9028', '9029']
    non_specific_inputs = ['st. gallen', 'st gallen', 'sankt gallen']

    # Check for non-specific input
    if address.lower().strip() in non_specific_inputs:
        return "non-specific"

    # If the input is a specific zip code, use it as is
    if address.strip() in valid_st_gallen_zip_codes:
        return address.strip()

    # Otherwise, append ", St. Gallen" to localize the search
    address += ", St. Gallen"

    # Extract zip code from the full address
    geolocator = Nominatim(user_agent="http")
    location = geolocator.geocode(address, country_codes='CH')
    if location:
        address_components = location.raw.get('display_name', '').split(',')
        for component in address_components:
            if component.strip() in valid_st_gallen_zip_codes:
                return component.strip()
    return None

def get_lat_lon_from_address_or_zip(input_text):
    geolocator = Nominatim(user_agent="http")
    # Add 'St. Gallen' suffix for zip codes to narrow down the search
    if input_text.isdigit() and len(input_text) == 4:
        input_text += ", St. Gallen, Switzerland"

    location = geolocator.geocode(input_text)
    if location:
        return location.latitude, location.longitude
    else:
        return default_lat, default_lon  # Return default coordinates if no location is found

# Preprocess data and train the model
model = preprocess_and_train()

# Define functions to handle navigation
def go_to_next_step():
    st.session_state.current_step += 1

def go_to_previous_step():
    st.session_state.current_step -= 1

# Create tabs for each step
tab_titles = ["Location", "Rooms", "Size", "Current Rent", "Result"]
tabs = st.tabs(tab_titles)

# Streamlit UI
st.title("Rental Price Prediction")

# Step 1: Location
with tabs[0]:
    if st.session_state.current_step == 0:
        address_input = st.text_input("Enter an address or zip code in St. Gallen:")
        extracted_zip_code = extract_zip_from_address(address_input)

        lat, lon = get_lat_lon_from_address_or_zip(address_input) if extracted_zip_code else (default_lat, default_lon)

        if extracted_zip_code == "non-specific":
            st.error("Please enter a more specific address or zip code in St. Gallen.")
        elif extracted_zip_code:
            # Display map
            map = folium.Map(location=[lat, lon], zoom_start=16)
            if address_input:
                folium.Marker(
                    [lat, lon],
                    popup=f"Eingegebene Adresse: {address_input}",
                    icon=folium.Icon(color='red', icon="glyphicon glyphicon-menu-down")
                ).add_to(map)
            folium_static(map)
        else:
            st.write("Please enter a valid address or zip code in St. Gallen.")

        st.button("Next", on_click=go_to_next_step)

# Step 2: Rooms
with tabs[1]:
    if st.session_state.current_step == 1:
        rooms = st.selectbox("Select the number of rooms", range(1, 7), key='rooms')

        st.button("Previous", on_click=go_to_previous_step)
        st.button("Next", on_click=go_to_next_step)

# Step 3: Size
with tabs[2]:
    if st.session_state.current_step == 2:
        size_m2 = st.number_input("Enter the size in square meters", min_value=0)

        st.button("Previous", on_click=go_to_previous_step)
        st.button("Next", on_click=go_to_next_step)

# Step 4: Current Rent
with tabs[3]:
    if st.session_state.current_step == 3:
        # Collect current rent
        st.session_state.current_rent = st.number_input("Enter your current rent in CHF:", 
                                                        min_value=0, 
                                                        value=st.session_state.get('current_rent', 0), 
                                                        step=10)

        if st.button("Previous", key='prev4', on_click=go_to_previous_step):
            # Logic for previous button (if needed)
            pass

        if st.button("Next", key='next4', on_click=go_to_next_step):

## Step 5: Result
with tabs[4]:
    if st.session_state.current_step == 4:
        # Previous button
        if st.button("Previous", key='prev5'):
            go_to_previous_step()

        # Display results or predictions
        if st.session_state.address and not st.session_state.address == "non-specific":
            # Ensure that the necessary data is available
            if 'size_m2' in st.session_state and 'rooms' in st.session_state:
                size_m2 = st.session_state.size_m2
                rooms = st.session_state.rooms
                extracted_zip_code = extract_zip_code(st.session_state.address)
                if extracted_zip_code:
                    predicted_price = predict_price(size_m2, extracted_zip_code, rooms, model)
                    if predicted_price is not None:
                        st.write(f"The predicted price for the apartment is CHF {predicted_price:.2f}")
                    else:
                        st.write("Unable to predict price. Please check your inputs.")

# Make tabs visible based on the current step
for i in range(len(tab_titles)):
    tabs[i].empty() if i != st.session_state.current_step else None

# Input für eine Adresse oder Postleitzahl
address_input = st.text_input("Enter an address or zip code in St. Gallen:")

# Extrahieren der Postleitzahl aus der Eingabe
extracted_zip_code = extract_zip_from_address(address_input)

# Use the function to get latitude and longitude from the input
lat, lon = get_lat_lon_from_address_or_zip(address_input) if extracted_zip_code else (default_lat, default_lon)

# Standardkoordinaten für St. Gallen setzen
default_lat, default_lon = 47.424482, 9.376717
lat, lon = default_lat, default_lon

# Check for the input type and update coordinates accordingly
if extracted_zip_code == "non-specific":
    st.error("Please enter a more specific address or zip code in St. Gallen.")
    lat, lon = default_lat, default_lon  # Reset to standard coordinates of St. Gallen
elif extracted_zip_code:
    # Get lat, lon based on either the full address or zip code
    lat, lon = get_lat_lon_from_address_or_zip(address_input) if address_input else (default_lat, default_lon)
else:
    st.write("Please enter a valid address or zip code in St. Gallen.")
    lat, lon = default_lat, default_lon  # Reset to standard coordinates of St. Gallen


# Vorhersagefunktionalität nur aktiviert, wenn eine gültige Postleitzahl vorliegt
if extracted_zip_code and not extracted_zip_code == "non-specific":
    room_options = list(range(1, 7))  # Liste von 1 bis 6
    rooms = st.selectbox("Select the number of rooms", room_options)
    size_m2 = st.number_input("Enter the size in square meters", min_value=0)

    if st.button('Predict Rental Price'):
        predicted_price = predict_price(size_m2, extracted_zip_code, rooms, model)
        if predicted_price is not None:
            st.write(f"The predicted price for the apartment is CHF {predicted_price:.2f}")
        else:
            st.write("Unable to predict price. Please check your inputs.")