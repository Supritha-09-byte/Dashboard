Vahan Vehicle Registration Analytics Dashboard

This is a Streamlit web application for analyzing vehicle registration data from the Ministry of Road Transport & Highways (VAHAN).
It provides Year-over-Year (YoY) and Quarter-over-Quarter (QoQ) growth analysis for different vehicle categories and manufacturers.

Features:

Interactive filters for year range, category, and manufacturer

YoY and QoQ growth analysis

Line charts, bar charts, and data tables

Option to download filtered data as CSV

Project Structure:
vahan-dashboard/
│
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── vehicle_registrations_cleaned.xlsx  # Dataset
├── README.md               # Documentation
└── assets/                 # Images, icons, etc.

Installation:

Clone the repository:

git clone https://github.com/yourusername/vahan-dashboard.git
cd vahan-dashboard


Create a virtual environment:

python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows


Install dependencies:

pip install -r requirements.txt


Place the dataset file (vehicle_registrations_cleaned.xlsx) in the project root.

Run the Application:
streamlit run app.py


The app will be available at:
http://localhost:8501
