import re
import pandas as pd
from bs4 import BeautifulSoup
import requests
from datetime import datetime

def clean_price(price):
    # Ersetzt Nicht-Brechende Leerzeichen und entfernt "-Zeichen
    return price.replace('\xa0', ' ').replace('-', '').strip()

def clean_pricem2(pricem2):
    # Entfernt Nicht-Brechende Leerzeichen und alles nach "/ m²"
    pricem2 = pricem2.replace('\xa0', ' ')
    return re.split(r' / m²', pricem2)[0].strip()

# Basis-URL und gemeinsame Parameter
base_url = 'https://realadvisor.ch/de/mieten/9000-st-gallen/haus-wohnung'
params = '?east=9.651524100294182&north=47.78651906423726&south=47.05433631754212&west=9.091221365919182'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

all_data = []

# Schleife über die Seitenzahlen
for page in range(1,2):  # Ändern Sie 5 auf die tatsächliche Anzahl der Seiten + 1
    if page == 1:
        url = base_url + params  # URL für die erste Seite
    else:
        url = f'{base_url}/seite-{page}{params}'  # URL für die anderen Seiten

    # Anfrage an die Seite senden
    response = requests.get(url, headers=headers)
    
    # Falls es sich um die zweite Seite handelt, die Anfrage nochmal senden
    if page == 2:
        response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'lxml')
        
        names_brooker_elements = soup.find_all('div', class_='css-1mtsvd6-AggregatesListingCard')
        names_brooker = [element.text.strip() for element in names_brooker_elements]

        zip_elements = soup.find_all('div', class_='css-1wo7mki-AggregatesListingCard')
        zips = [element.text.strip() for element in zip_elements]

        description_elements = soup.find_all('div', class_='css-qb670f-AggregatesListingCard')
        descriptions = [element.text.strip() for element in description_elements]

        description2_elements = soup.find_all('div', class_='css-1lelbas-AggregatesListingCard')
        descriptions2 = [element.text.strip() for element in description2_elements]

        location_elements = soup.find_all('div', class_='css-1lelbas-AggregatesListingCard')
        locations = [element.text.strip() for element in location_elements]

        price_elements = soup.find_all('span', class_='css-1r801wc')
        prices = [clean_price(element.text) for element in price_elements]

        pricem2_elements = soup.find_all('div', class_='css-1eo6i6u-AggregatesListingCard')
        pricem2 = [clean_pricem2(element.text) for element in pricem2_elements]

        website_elements = soup.find_all('div', class_='css-vc4s6w-AggregatesListingCard')
        websites = [element.text.strip() for element in website_elements]

        page_data = {
            'Name': names_brooker,
            'Description': descriptions,
            'Details': descriptions2,
            'Place': locations,
            'Price': prices,
            'Price_per_m2_per_year': pricem2,
            'zip': zips,
            'websiten': websites 
        }

        all_data.append(pd.DataFrame(page_data))

    else:
        print(f"Fehler beim Laden der Seite {page}")

# Zusammenführen aller Daten in einem DataFrame
# Zusammenführen aller Daten in einem DataFrame
df = pd.concat(all_data, ignore_index=True)

#Save file in relative path
file_path = 'Immobilienliste.xlsx'  # Relative path
df.to_excel(file_path, index=False)