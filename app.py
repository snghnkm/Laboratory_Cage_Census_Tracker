import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import io
import base64
from datetime import datetime
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# --- Configuration ---
st.set_page_config(page_title="Cage Census Tracker", layout="wide", page_icon="🐭")

DATA_FOLDER = "census_uploads"
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# --- Helper: Decryption ---
def derive_key(password: str, salt: bytes) -> bytes:
    """Derives a cryptographic key from a password and salt."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

def load_data_securely(filepath, password=None):
    """
    Loads CSV data. If extension is .enc, attempts to decrypt with password.
    Returns: (DataFrame, ErrorMessage)
    """
    if filepath.endswith('.enc'):
        if not password:
            return None, "Password required"
        
        try:
            with open(filepath, 'rb') as f:
                file_content = f.read()
            
            # Extract salt (first 16 bytes) and data
            salt = file_content[:16]
            encrypted_data = file_content[16:]
            
            key = derive_key(password, salt)
            fernet = Fernet(key)
            decrypted_data = fernet.decrypt(encrypted_data)
            
            return pd.read_csv(io.BytesIO(decrypted_data)), None
        except InvalidToken:
            return None, "Incorrect Password"
        except Exception as e:
            return None, f"Decryption Error: {str(e)}"
            
    else:
        # Regular CSV
        return pd.read_csv(filepath), None

# --- App Title ---
st.title("📊 Laboratory Cage Census Tracker")
st.markdown("Upload your census CSV (or encrypted .enc file) to track cage usage by Room, Rack, and Investigator.")

# --- Session State Initialization ---
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# --- Sidebar: File Management ---
st.sidebar.header("📂 Data Management")

# Update allowed types to include 'enc'
uploaded_file = st.sidebar.file_uploader("Upload New Census", type=["csv", "enc"])

if uploaded_file is not None:
    if uploaded_file.name not in st.session_state.processed_files:
        file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.processed_files.add(uploaded_file.name)
        st.session_state["census_selector"] = uploaded_file.name
        
        st.sidebar.success(f"✅ Saved: {uploaded_file.name}")
        st.rerun()

try:
    # Look for both csv and enc files
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv') or f.endswith('.enc')]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(DATA_FOLDER, x)), reverse=True)
except Exception as e:
    files = []

if not files:
    st.info("👋 No data found. Please upload a file to begin.")
    st.stop()

st.sidebar.divider()
st.sidebar.header("🕒 History")

selected_filename = st.sidebar.selectbox(
    "Select Census Result",
    options=files,
    index=0,
    key="census_selector"
)

# --- Main Logic ---
if selected_filename:
    current_file_path = os.path.join(DATA_FOLDER, selected_filename)
    
    # Handle Password for Encrypted Files
    df = None
    if selected_filename.endswith('.enc'):
        st.warning(f"🔒 {selected_filename} is encrypted.")
        password_input = st.text_input("Enter decryption password:", key="decryption_password")
        
        if password_input:
            df, error = load_data_securely(current_file_path, password_input)
            if error:
                st.error(f"❌ {error}")
    else:
        # Load regular CSV directly
        df, error = load_data_securely(current_file_path)
        if error:
            st.error(error)

    # Proceed if Data is Loaded
    if df is not None:
        try:
            required_columns = ['Cage', 'Room', 'Rack', 'LabContact']
            missing = [col for col in required_columns if col not in df.columns]
            
            if missing:
                st.error(f"⚠️ Error: The file is missing these columns: {', '.join(missing)}")
            else:
                file_stats = os.stat(current_file_path)
                upload_time = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M')
                st.caption(f"Showing data for: **{selected_filename}** (Uploaded: {upload_time})")
                
                # --- Analysis 1: Graphs (Moved to Top) ---
                st.divider()
                st.subheader("1. 📉 Census Visualization")
                
                # 1. Prepare Base Data (Per Person)
                graph_data = df.groupby(['LabContact', 'Room', 'Rack']).size().reset_index(name='Cage Count')
                graph_data['Location'] = graph_data['Room'] + " - " + graph_data['Rack']
                
                # 2. Prepare Grand Total Data
                grand_total_data = df.groupby(['Room', 'Rack']).size().reset_index(name='Cage Count')
                grand_total_data['LabContact'] = "Total" # Shortened name
                grand_total_data['Location'] = grand_total_data['Room'] + " - " + grand_total_data['Rack']

                # --- Graph Layout: 1 column vs 4 columns ratio ---
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    st.markdown("### 🔹 Total")
                    total_sum = grand_total_data['Cage Count'].sum()
                    
                    fig_total = px.bar(
                        grand_total_data, 
                        x='LabContact', 
                        y='Cage Count', 
                        color='Location',
                        text_auto=True,
                        title="", # Removed title to save space
                        height=500
                    )
                    
                    # Make the single bar narrower explicitly
                    fig_total.update_traces(width=0.6)
                    
                    # Annotation for Grand Total
                    fig_total.add_annotation(
                        x="Total",
                        y=total_sum,
                        text=f"<b>{total_sum}</b>",
                        showarrow=False,
                        yshift=10,
                        bgcolor="rgba(220, 220, 220, 0.7)",
                        borderpad=3,
                        font=dict(size=16, color="black")
                    )
                    
                    fig_total.update_layout(
                        xaxis_title="", 
                        yaxis_title="Cages",
                        margin=dict(t=20, l=10, r=10),
                        showlegend=False
                    )
                    st.plotly_chart(fig_total, use_container_width=True)

                # --- Graph B: By Investigator ---
                with col2:
                    st.markdown("### 🔹 Usage by Investigator")
                    
                    # Sort by total count per contact
                    contact_totals = graph_data.groupby('LabContact')['Cage Count'].sum().sort_values(ascending=False)
                    x_order = contact_totals.index.tolist()

                    fig_inv = px.bar(
                        graph_data, 
                        x='LabContact', 
                        y='Cage Count', 
                        color='Location', 
                        text_auto=True,
                        title="",
                        height=500,
                        category_orders={'LabContact': x_order}
                    )
                    
                    # Floating Annotations for Investigators
                    totals_df = contact_totals.reset_index()
                    annotations = []
                    for index, row in totals_df.iterrows():
                        annotations.append(dict(
                            x=row['LabContact'],
                            y=row['Cage Count'],
                            text=f"<b>{row['Cage Count']}</b>",
                            xanchor='center',
                            yanchor='bottom',
                            showarrow=False,
                            yshift=8,
                            bgcolor="rgba(220, 220, 220, 0.7)",
                            borderpad=3,
                            font=dict(size=14, color="black")
                        ))

                    fig_inv.update_layout(
                        xaxis_title="Investigator", 
                        yaxis_title="", # Hide y-axis title to avoid duplication
                        margin=dict(t=20),
                        annotations=annotations,
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.2,
                            xanchor="center",
                            x=0.5
                        )
                    )
                    
                    st.plotly_chart(fig_inv, use_container_width=True)

                # --- Analysis 2: Pivot Tables (Separated by Room) ---
                st.divider()
                st.subheader("2. 📋 Cage Location Summary")
                st.markdown("Tables separated by **Room**.")
                
                unique_rooms = sorted(df['Room'].astype(str).unique())

                for room in unique_rooms:
                    st.markdown(f"### 🏠 Room: {room}")
                    room_df = df[df['Room'] == room]
                    
                    pivot_data = room_df.groupby(['LabContact', 'Rack']).size()
                    pivot_df = pivot_data.to_frame(name='Total Cages').sort_values(by='Total Cages', ascending=False)
                    
                    total_count = pivot_df['Total Cages'].sum()
                    total_row = pd.DataFrame({'Total Cages': [total_count]}, 
                                             index=pd.MultiIndex.from_tuples([('🔴 ROOM TOTAL', '')], 
                                             names=['LabContact', 'Rack']))
                    
                    final_display_df = pd.concat([total_row, pivot_df])

                    st.dataframe(
                        final_display_df,
                        use_container_width=True,
                        column_config={"Total Cages": st.column_config.NumberColumn(format="%d 🐭")}
                    )
                    st.write("")

        except Exception as e:
            st.error(f"An unexpected error occurred processing the file data: {e}")