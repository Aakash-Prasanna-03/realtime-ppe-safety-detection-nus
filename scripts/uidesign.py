import os
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# Page Configuration
st.set_page_config(
    page_title="Employee Safety Violation Tracker",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #fafbfc;
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 3rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        margin: 1rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.95;
        font-weight: 300;
    }
    
    /* Section headers */
    .section-header {
        color: #2d3748;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Violation card styling */
    .violation-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .violation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        border-color: #cbd5e0;
    }
    
    /* Employee info styling */
    .employee-id {
        background: #f7fafc;
        padding: 0.75rem 1.2rem;
        border-radius: 8px;
        font-family: 'Inter', monospace;
        font-weight: 600;
        color: #4a5568;
        border: 1px solid #e2e8f0;
        text-align: center;
        font-size: 0.95rem;
    }
    
    .employee-name {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.5rem;
        line-height: 1.4;
    }
    
    .field-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .violation-count {
        background: linear-gradient(135deg, #fc8181 0%, #f56565 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        text-align: center;
        font-size: 1.1rem;
        box-shadow: 0 2px 8px rgba(245, 101, 101, 0.3);
    }
    
    .violation-count-high {
        background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    /* Image container */
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 180px;
        background: #f7fafc;
        border-radius: 12px;
        border: 2px dashed #cbd5e0;
        color: #a0aec0;
        font-weight: 500;
    }
    
    /* No data message */
    .no-data {
        text-align: center;
        padding: 4rem 2rem;
        background: white;
        border-radius: 16px;
        margin: 3rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .no-data h3 {
        color: #4a5568;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .no-data p {
        color: #718096;
        line-height: 1.6;
    }
    
    /* Stats cards */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-bottom: 3rem;
    }
    
    .stats-card {
        background: white;
        padding: 2rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: transform 0.2s ease;
    }
    
    .stats-card:hover {
        transform: translateY(-2px);
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        line-height: 1;
    }
    
    .stats-number.primary { color: #667eea; }
    .stats-number.warning { color: #ed8936; }
    .stats-number.success { color: #48bb78; }
    .stats-number.danger { color: #f56565; }
    
    .stats-label {
        color: #718096;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #a0aec0;
        padding: 3rem 2rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 4rem;
        background: white;
        border-radius: 12px;
    }
    
    /* Custom divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="main-header">
    <h1>Safety Violation Management System</h1>
    <p>Comprehensive Employee Safety Monitoring & Compliance Dashboard</p>
</div>
""", unsafe_allow_html=True)

# File paths
csv_path = "outputs/reports/violations_report.csv"
profile_folder = "data/user_profiles"
screenshot_folder = "outputs/screenshots"
os.makedirs(profile_folder, exist_ok=True)

def format_passport_image(img_path):
    """Resize image to passport dimensions (120x160) with proper formatting"""
    try:
        img = Image.open(img_path).convert("RGB")
        passport_size = (120, 160)
        img = ImageOps.fit(img, passport_size, method=Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        return None

def display_violation_stats(df):
    """Display summary statistics"""
    if not df.empty:
        total_employees = len(df)
        total_violations = df['Violations'].sum() if 'Violations' in df.columns else 0
        avg_violations = round(df['Violations'].mean(), 1) if 'Violations' in df.columns else 0
        high_risk = len(df[df['Violations'] >= 3]) if 'Violations' in df.columns else 0
        
        st.markdown(f"""
        <div class="stats-container">
            <div class="stats-card">
                <div class="stats-number primary">{total_employees}</div>
                <div class="stats-label">Total Employees</div>
            </div>
            <div class="stats-card">
                <div class="stats-number warning">{total_violations}</div>
                <div class="stats-label">Total Violations</div>
            </div>
            <div class="stats-card">
                <div class="stats-number success">{avg_violations}</div>
                <div class="stats-label">Average per Employee</div>
            </div>
            <div class="stats-card">
                <div class="stats-number danger">{high_risk}</div>
                <div class="stats-label">High Risk Cases</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Main Content
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    
    if not df.empty:
        # Display statistics
        st.markdown('<div class="section-header">Performance Overview</div>', unsafe_allow_html=True)
        display_violation_stats(df)
        
        # Violation Reports Section
        st.markdown('<div class="section-header">Employee Violation Records</div>', unsafe_allow_html=True)
        
        # Sort by violations count (highest first)
        df_sorted = df.sort_values('Violations', ascending=False) if 'Violations' in df.columns else df
        
        for idx, row in df_sorted.iterrows():
            st.markdown('<div class="violation-card">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([1.5, 1.2, 2.5, 1.3])
            
            # Column 1: Employee Photo
            with col1:
                img_path = None
                
                # Search for profile image
                if row['Emp ID'] != "N/A":
                    for file in os.listdir(profile_folder):
                        if file.startswith(str(row['Emp ID'])) and str(row['Emp Name']) in file:
                            img_path = os.path.join(profile_folder, file)
                            break
                
                # Fallback to URL path
                if not img_path and pd.notna(row.get('Pic URL', '')) and os.path.exists(str(row['Pic URL'])):
                    img_path = str(row['Pic URL'])
                
                if img_path and os.path.exists(img_path):
                    passport_img = format_passport_image(img_path)
                    if passport_img:
                        st.image(passport_img, width=120)
                    else:
                        st.markdown("""
                        <div class="image-container">
                            <span>Image Unavailable</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="image-container">
                        <span>No Photo Available</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Column 2: Employee ID
            with col2:
                st.markdown('<div class="field-label">Employee ID</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="employee-id">{row["Emp ID"]}</div>', unsafe_allow_html=True)
            
            # Column 3: Employee Name
            with col3:
                st.markdown('<div class="field-label">Full Name</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="employee-name">{row["Emp Name"]}</div>', unsafe_allow_html=True)
            
            # Column 4: Violation Count
            with col4:
                st.markdown('<div class="field-label">Violations Count</div>', unsafe_allow_html=True)
                violation_count = row.get('Violations', 0)
                card_class = "violation-count-high" if violation_count >= 3 else "violation-count"
                st.markdown(f'<div class="violation-count {card_class}">{violation_count}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="no-data">
            <h3>No Violation Records Found</h3>
            <p>The violations report file exists but contains no data.</p>
            <p>Please ensure the monitoring system has processed employee data properly.</p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="no-data">
        <h3>System Data Not Available</h3>
        <p>The violations report file <code>violations_report.csv</code> could not be located.</p>
        <p>Please ensure the face detection monitoring system has been executed to generate the required data.</p>
        <br>
        <p><strong>Expected File Location:</strong> <code>violations_report.csv</code></p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p><strong>Safety Violation Management System</strong></p>
    <p>Ensuring Workplace Safety Standards & Regulatory Compliance</p>
    <p style="font-size: 0.8rem; margin-top: 1rem;">Confidential & Secure • Real-time Data Processing</p>
</div>
""", unsafe_allow_html=True)