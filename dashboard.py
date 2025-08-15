import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

# --- Enhanced Configuration ---
st.set_page_config(
    page_title="Vahan Registration Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': "Professional Vehicle Registration Analytics Dashboard v2.0"
    }
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main background and container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Custom metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        color: white;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
        margin: 0;
    }
    
    .metric-delta {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .dashboard-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0.5rem 0;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1e3c72;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Filter section styling */
    .filter-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    
    /* Data table styling */
    .dataframe {
        border: none !important;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    /* Hover effects */
    .metric-card:hover {
        transform: translateY(-5px);
        transition: all 0.3s ease;
    }
    
    /* Success/Warning/Error styling */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    /* Progress bar custom styling */
    .progress-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Enhanced Functions ---

# Constants for month and quarter mapping
INDIAN_MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
QUARTER_MAP = {
    'JAN': 'Q1', 'FEB': 'Q1', 'MAR': 'Q1',
    'APR': 'Q2', 'MAY': 'Q2', 'JUN': 'Q2',
    'JUL': 'Q3', 'AUG': 'Q3', 'SEP': 'Q3',
    'OCT': 'Q4', 'NOV': 'Q4', 'DEC': 'Q4'
}

# Color palette for professional charts
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

CHART_COLORS = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#43e97b', '#fa709a', '#ffecd2']

@st.cache_data(show_spinner=False)
def load_data(file_path):
    """Enhanced data loading with better error handling and progress indication."""
    try:
        with st.spinner("Loading and processing data..."):
            # Load the data, skipping the blank row and handling spaces
            df = pd.read_csv(file_path, skiprows=[1], skipinitialspace=True)
            
            # Rename the 'Maker' column to 'Manufacturer'
            df.rename(columns={'Maker': 'Manufacturer'}, inplace=True)
            
            # Define the columns that represent months
            month_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
            
            # Reshape the data from wide to long format
            df_long = df.melt(
                id_vars=['sl.no', 'Manufacturer', 'vehicle_type', 'year'],
                value_vars=month_cols,
                var_name='Month',
                value_name='Count'
            )

            # Convert the 'Count' column to numeric, coercing errors
            df_long['Count'] = pd.to_numeric(df_long['Count'], errors='coerce').fillna(0)

            # Drop any rows where Count is 0 or NaN after conversion
            df_long = df_long[df_long['Count'] > 0]
            
            # Combine year and month to create a proper 'Registration_Date'
            df_long['Registration_Date'] = pd.to_datetime(
                df_long['year'].astype(str) + '-' + df_long['Month'],
                format='%Y-%b',
                errors='coerce'
            )
            
            # Map month names to quarter names
            df_long['Quarter'] = df_long['Month'].map(QUARTER_MAP)

            # Filter out any rows where date conversion failed
            df_long.dropna(subset=['Registration_Date'], inplace=True)

            return df_long

    except FileNotFoundError:
        st.error("🚫 **Data file not found!** Please ensure 'vahan_dashboard.csv' is in the correct location.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ **Data loading error:** {str(e)}")
        st.stop()

def add_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced growth calculations with better performance."""
    df = df.sort_values(['Manufacturer', 'vehicle_type', 'year', 'Registration_Date']).copy()

    # Calculate YoY growth
    df['yoy_growth'] = np.nan
    prev = df.copy()
    prev['Registration_Date'] = prev['Registration_Date'] + pd.DateOffset(years=1)
    prev = prev[['Manufacturer', 'vehicle_type', 'Registration_Date', 'Count']].rename(columns={'Count': 'prev_year_count'})
    df = df.merge(prev, on=['Manufacturer', 'vehicle_type', 'Registration_Date'], how='left')
    df['yoy_growth'] = ((df['Count'] - df['prev_year_count']) / df['prev_year_count'] * 100).replace([np.inf, -np.inf], np.nan)
    df.drop(columns=['prev_year_count'], inplace=True)

    # Calculate QoQ growth
    q_sum = (
        df.groupby(['Manufacturer', 'vehicle_type', 'year', 'Quarter'], as_index=False)['Count'].sum()
        .rename(columns={'Count': 'q_sum'})
    )
    q_sum = q_sum.sort_values(['Manufacturer', 'vehicle_type', 'year', 'Quarter'])
    q_sum['prev_q_sum'] = q_sum.groupby(['Manufacturer', 'vehicle_type'])['q_sum'].shift(1)
    q_sum['qoq_growth'] = ((q_sum['q_sum'] - q_sum['prev_q_sum']) / q_sum['prev_q_sum'] * 100).replace([np.inf, -np.inf], np.nan)
    
    # Merge QoQ growth back to the main DataFrame
    df['qoq_growth'] = np.nan
    df = df.merge(q_sum[['Manufacturer', 'vehicle_type', 'year', 'Quarter', 'qoq_growth']],
                  on=['Manufacturer', 'vehicle_type', 'year', 'Quarter'], how='left', suffixes=('', '_q'))
    df['qoq_growth'] = np.where(df['Month'].isin(['MAR', 'JUN', 'SEP', 'DEC']), df['qoq_growth_q'], np.nan)
    df.drop(columns=['qoq_growth_q'], inplace=True)
    
    return df

def create_enhanced_metric_card(label, value, delta=None, delta_color="normal"):
    """Create enhanced metric cards with custom styling."""
    delta_html = ""
    if delta is not None:
        color = COLORS['success'] if delta_color == "normal" and delta > 0 else COLORS['danger'] if delta < 0 else COLORS['info']
        delta_html = f'<p class="metric-delta" style="color: {color};">{"+" if delta > 0 else ""}{delta:.2f}%</p>'
    
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-label">{label}</p>
        <p class="metric-value">{value}</p>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def create_advanced_chart(fig, title):
    """Apply consistent styling to charts."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, family="Arial, sans-serif", color=COLORS['dark']),
            x=0.05
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(t=60, l=60, r=60, b=60),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig.update_xaxes(
        gridcolor='rgba(0,0,0,0.1)',
        gridwidth=1,
        zeroline=False
    )
    fig.update_yaxes(
        gridcolor='rgba(0,0,0,0.1)',
        gridwidth=1,
        zeroline=False
    )
    
    return fig

# --- Main Dashboard Logic ---
def main():
    # Dashboard Header
    st.markdown("""
    <div class="dashboard-header">
        <h1 class="dashboard-title">🚗 Vahan Registration Analytics</h1>
        <p class="dashboard-subtitle">Professional Vehicle Registration Intelligence Dashboard</p>
        <p style="margin: 0; opacity: 0.8;">Real-time insights • Growth Analytics • Market Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for data caching
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.df = None

    # File upload or default loading
    uploaded_file = st.file_uploader("📁 Upload your CSV file", type=['csv'], help="Upload a CSV file with vehicle registration data")
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.session_state.df = df
        st.session_state.data_loaded = True
    elif not st.session_state.data_loaded:
        # Try to load default file
        FILE_PATH = 'vahan_dashboard.csv'
        try:
            df = load_data(FILE_PATH)
            st.session_state.df = df
            st.session_state.data_loaded = True
        except:
            st.warning("📤 Please upload a CSV file to begin analysis.")
            st.stop()
    else:
        df = st.session_state.df

    if df is not None and not df.empty:
        # Add growth columns
        df = add_growth(df)
        
        # Get unique values for filters
        vehicle_types = sorted(df['vehicle_type'].unique().tolist())
        manufacturers = sorted(df['Manufacturer'].unique().tolist())
        years = sorted(df['year'].unique().tolist())
        
        # --- Enhanced Sidebar with Professional Filters ---
        with st.sidebar:
            st.markdown("""
            <div class="filter-container">
                <h2 style="color: #1e3c72; margin-top: 0;">🎛️ Control Panel</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Date Range Selection with presets
            st.subheader("📅 Date Range")
            min_date = df['Registration_Date'].min().date()
            max_date = df['Registration_Date'].max().date()
            
            # Preset date ranges
            preset_option = st.selectbox(
                "Quick Date Presets",
                ["Custom Range", "Last 12 Months", "Last 6 Months", "Current Year", "Previous Year", "All Data"]
            )
            
            if preset_option == "Custom Range":
                date_range = st.date_input(
                    "Select Custom Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
            elif preset_option == "Last 12 Months":
                end_date = max_date
                start_date = (datetime.combine(max_date, datetime.min.time()) - relativedelta(months=12)).date()
                date_range = (start_date, end_date)
            elif preset_option == "Last 6 Months":
                end_date = max_date
                start_date = (datetime.combine(max_date, datetime.min.time()) - relativedelta(months=6)).date()
                date_range = (start_date, end_date)
            elif preset_option == "Current Year":
                current_year = max_date.year
                date_range = (date(current_year, 1, 1), max_date)
            elif preset_option == "Previous Year":
                prev_year = max_date.year - 1
                date_range = (date(prev_year, 1, 1), date(prev_year, 12, 31))
            else:  # All Data
                date_range = (min_date, max_date)
            
            if len(date_range) == 2:
                start_date, end_date = date_range
            else:
                st.sidebar.warning("Please select a start and end date.")
                st.stop()
            
            # Advanced filters
            st.subheader("🚗 Vehicle Filters")
            
            # Vehicle Type with Select All option
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_vehicle_types = st.multiselect(
                    "Vehicle Types",
                    options=vehicle_types,
                    default=vehicle_types,
                    help="Select vehicle types to analyze"
                )
            with col2:
                if st.button("Select All", key="select_all_vehicles"):
                    selected_vehicle_types = vehicle_types
                    st.rerun()
            
            # Manufacturer with search and Select All
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_manufacturers = st.multiselect(
                    "Manufacturers",
                    options=manufacturers,
                    default=manufacturers[:10] if len(manufacturers) > 10 else manufacturers,
                    help="Select manufacturers to analyze"
                )
            with col2:
                if st.button("Select All", key="select_all_manufacturers"):
                    selected_manufacturers = manufacturers
                    st.rerun()
            
            # Year filter
            selected_years = st.multiselect(
                "Years",
                options=years,
                default=years,
                help="Select years to analyze"
            )
            
            # Advanced analytics options
            st.subheader("📊 Analytics Options")
            show_growth_analysis = st.checkbox("Show Growth Analysis", value=True)
            show_market_share = st.checkbox("Show Market Share Analysis", value=True)
            show_forecasting = st.checkbox("Show Trend Forecasting", value=False)
            
            # Data refresh
            st.markdown("---")
            if st.button("🔄 Refresh Data", type="primary"):
                st.cache_data.clear()
                st.rerun()
        
        # --- Apply Filters ---
        filtered_df = df[
            (df['Registration_Date'].dt.date >= start_date) & 
            (df['Registration_Date'].dt.date <= end_date) &
            (df['vehicle_type'].isin(selected_vehicle_types)) &
            (df['Manufacturer'].isin(selected_manufacturers)) &
            (df['year'].isin(selected_years))
        ]
        
        # Handle case where no data is found after filtering
        if filtered_df.empty:
            st.warning("⚠️ No data available for the selected filters. Please adjust your selection.")
            st.stop()
        
        # Data summary info
        st.info(f"📈 **Analysis Summary:** {len(filtered_df):,} records • {filtered_df['Manufacturer'].nunique()} manufacturers • {filtered_df['vehicle_type'].nunique()} vehicle types")
        
        # --- Enhanced KPIs Section ---
        st.markdown('<h2 class="section-header">📊 Key Performance Indicators</h2>', unsafe_allow_html=True)
        
        # Calculate KPIs
        total_registrations = filtered_df['Count'].sum()
        avg_yoy = filtered_df['yoy_growth'].mean()
        avg_qoq = filtered_df['qoq_growth'].mean()
        
        # Calculate additional KPIs
        peak_month = filtered_df.groupby(filtered_df['Registration_Date'].dt.to_period('M'))['Count'].sum().idxmax()
        top_manufacturer = filtered_df.groupby('Manufacturer')['Count'].sum().idxmax()
        top_vehicle_type = filtered_df.groupby('vehicle_type')['Count'].sum().idxmax()
        
        # Display KPIs in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_enhanced_metric_card("Total Registrations", f"{total_registrations:,}")
        
        with col2:
            create_enhanced_metric_card("Avg YoY Growth", f"{avg_yoy:+.1f}%", delta=avg_yoy)
        
        with col3:
            create_enhanced_metric_card("Avg QoQ Growth", f"{avg_qoq:+.1f}%", delta=avg_qoq)
        
        with col4:
            create_enhanced_metric_card("Peak Month", f"{peak_month}")
        
        # Additional KPI row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_enhanced_metric_card("Top Manufacturer", f"{top_manufacturer}")
        
        with col2:
            create_enhanced_metric_card("Top Vehicle Type", f"{top_vehicle_type}")
        
        with col3:
            create_enhanced_metric_card("Data Points", f"{len(filtered_df):,}")
        
        with col4:
            monthly_avg = total_registrations / max(1, len(filtered_df.groupby(filtered_df['Registration_Date'].dt.to_period('M'))))
            create_enhanced_metric_card("Monthly Average", f"{monthly_avg:,.0f}")
        
        # --- Enhanced Charts Section ---
        st.markdown('<h2 class="section-header">📈 Registration Trends & Analytics</h2>', unsafe_allow_html=True)
        
        # Chart layout tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Vehicle Analysis", "🏭 Manufacturer Analysis", "📈 Growth Insights"])
        
        with tab1:
            # 1. Enhanced Monthly Trend
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            monthly_data = filtered_df.groupby(filtered_df['Registration_Date'].dt.to_period('M'))['Count'].sum().reset_index()
            monthly_data['Registration_Date'] = monthly_data['Registration_Date'].astype(str)
            
            fig_monthly = px.line(
                monthly_data,
                x='Registration_Date',
                y='Count',
                title="📅 Monthly Registration Trends",
                markers=True,
                color_discrete_sequence=[COLORS['primary']]
            )
            
            # Add trend line
            fig_monthly.add_scatter(
                x=monthly_data['Registration_Date'],
                y=monthly_data['Count'].rolling(window=3).mean(),
                mode='lines',
                name='3-Month Moving Average',
                line=dict(dash='dash', color=COLORS['secondary'])
            )
            
            fig_monthly = create_advanced_chart(fig_monthly, "📅 Monthly Registration Trends")
            st.plotly_chart(fig_monthly, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 2. Registration Distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                quarterly_data = filtered_df.groupby(['year', 'Quarter'])['Count'].sum().reset_index()
                quarterly_data['Year_Quarter'] = quarterly_data['year'].astype(str) + '-' + quarterly_data['Quarter']
                
                fig_quarterly = px.bar(
                    quarterly_data,
                    x='Year_Quarter',
                    y='Count',
                    title="📊 Quarterly Distribution",
                    color='Count',
                    color_continuous_scale='viridis'
                )
                fig_quarterly = create_advanced_chart(fig_quarterly, "📊 Quarterly Distribution")
                st.plotly_chart(fig_quarterly, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                yearly_data = filtered_df.groupby('year')['Count'].sum().reset_index()
                
                fig_yearly = px.bar(
                    yearly_data,
                    x='year',
                    y='Count',
                    title="📅 Yearly Registration Trends",
                    color='Count',
                    color_continuous_scale='blues'
                )
                fig_yearly = create_advanced_chart(fig_yearly, "📅 Yearly Registration Trends")
                st.plotly_chart(fig_yearly, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            # Vehicle Type Analysis
            vehicle_type_data = filtered_df.groupby('vehicle_type')['Count'].sum().reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig_vehicle_pie = px.pie(
                    vehicle_type_data,
                    values='Count',
                    names='vehicle_type',
                    title="🚗 Vehicle Type Distribution",
                    color_discrete_sequence=CHART_COLORS
                )
                fig_vehicle_pie = create_advanced_chart(fig_vehicle_pie, "🚗 Vehicle Type Distribution")
                st.plotly_chart(fig_vehicle_pie, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig_vehicle_bar = px.bar(
                    vehicle_type_data.sort_values('Count', ascending=True),
                    x='Count',
                    y='vehicle_type',
                    orientation='h',
                    title="📊 Vehicle Type Rankings",
                    color='Count',
                    color_continuous_scale='plasma'
                )
                fig_vehicle_bar = create_advanced_chart(fig_vehicle_bar, "📊 Vehicle Type Rankings")
                st.plotly_chart(fig_vehicle_bar, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Vehicle type trends over time
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            vehicle_time_data = filtered_df.groupby([filtered_df['Registration_Date'].dt.to_period('M'), 'vehicle_type'])['Count'].sum().reset_index()
            vehicle_time_data['Registration_Date'] = vehicle_time_data['Registration_Date'].astype(str)
            
            fig_vehicle_trend = px.line(
                vehicle_time_data,
                x='Registration_Date',
                y='Count',
                color='vehicle_type',
                title="🔄 Vehicle Type Trends Over Time",
                markers=True,
                color_discrete_sequence=CHART_COLORS
            )
            fig_vehicle_trend = create_advanced_chart(fig_vehicle_trend, "🔄 Vehicle Type Trends Over Time")
            st.plotly_chart(fig_vehicle_trend, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            # Top manufacturers analysis
            manufacturer_data = filtered_df.groupby('Manufacturer')['Count'].sum().sort_values(ascending=False).head(15).reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig_mfg_bar = px.bar(
                    manufacturer_data,
                    x='Count',
                    y='Manufacturer',
                    orientation='h',
                    title="🏭 Top 15 Manufacturers",
                    color='Count',
                    color_continuous_scale='viridis'
                )
                fig_mfg_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                fig_mfg_bar = create_advanced_chart(fig_mfg_bar, "🏭 Top 15 Manufacturers")
                st.plotly_chart(fig_mfg_bar, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                # Market share analysis
                if show_market_share:
                    top_10_mfg = manufacturer_data.head(10)
                    others_count = manufacturer_data.iloc[10:]['Count'].sum()
                    if others_count > 0:
                        others_df = pd.DataFrame({'Manufacturer': ['Others'], 'Count': [others_count]})
                        market_share_data = pd.concat([top_10_mfg, others_df], ignore_index=True)
                    else:
                        market_share_data = top_10_mfg
                    
                    fig_market_share = px.pie(
                        market_share_data,
                        values='Count',
                        names='Manufacturer',
                        title="📊 Market Share (Top 10 + Others)",
                        color_discrete_sequence=CHART_COLORS
                    )
                    fig_market_share = create_advanced_chart(fig_market_share, "📊 Market Share Analysis")
                    st.plotly_chart(fig_market_share, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Manufacturer performance matrix
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            mfg_performance = filtered_df.groupby(['Manufacturer', 'vehicle_type'])['Count'].sum().reset_index()
            top_manufacturers = manufacturer_data.head(10)['Manufacturer'].tolist()
            mfg_performance_filtered = mfg_performance[mfg_performance['Manufacturer'].isin(top_manufacturers)]
            
            fig_heatmap = px.density_heatmap(
                mfg_performance_filtered,
                x='vehicle_type',
                y='Manufacturer',
                z='Count',
                title="🔥 Manufacturer vs Vehicle Type Performance Matrix",
                color_continuous_scale='viridis'
            )
            fig_heatmap = create_advanced_chart(fig_heatmap, "🔥 Performance Matrix")
            st.plotly_chart(fig_heatmap, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            if show_growth_analysis:
                # Growth Analysis Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    # Vehicle Type YoY Growth
                    cat_yoy = filtered_df.groupby('vehicle_type')['yoy_growth'].mean().reset_index()
                    cat_yoy = cat_yoy.dropna()
                    
                    fig_cat_yoy = px.bar(
                        cat_yoy,
                        x='vehicle_type',
                        y='yoy_growth',
                        title="📈 Average YoY Growth by Vehicle Type",
                        color='yoy_growth',
                        color_continuous_scale='RdYlGn',
                        color_continuous_midpoint=0
                    )
                    fig_cat_yoy = create_advanced_chart(fig_cat_yoy, "📈 YoY Growth by Vehicle Type")
                    st.plotly_chart(fig_cat_yoy, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    # Vehicle Type QoQ Growth
                    cat_qoq = filtered_df[filtered_df['qoq_growth'].notna()].groupby('vehicle_type')['qoq_growth'].mean().reset_index()
                    
                    fig_cat_qoq = px.bar(
                        cat_qoq,
                        x='vehicle_type',
                        y='qoq_growth',
                        title="📊 Average QoQ Growth by Vehicle Type",
                        color='qoq_growth',
                        color_continuous_scale='RdYlBu',
                        color_continuous_midpoint=0
                    )
                    fig_cat_qoq = create_advanced_chart(fig_cat_qoq, "📊 QoQ Growth by Vehicle Type")
                    st.plotly_chart(fig_cat_qoq, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Growth trends over time
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                growth_time_data = filtered_df[filtered_df['yoy_growth'].notna()].groupby(
                    filtered_df[filtered_df['yoy_growth'].notna()]['Registration_Date'].dt.to_period('M')
                )['yoy_growth'].mean().reset_index()
                growth_time_data['Registration_Date'] = growth_time_data['Registration_Date'].astype(str)
                
                fig_growth_trend = px.line(
                    growth_time_data,
                    x='Registration_Date',
                    y='yoy_growth',
                    title="📈 YoY Growth Trends Over Time",
                    markers=True,
                    color_discrete_sequence=[COLORS['success']]
                )
                
                # Add horizontal line at 0
                fig_growth_trend.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
                
                fig_growth_trend = create_advanced_chart(fig_growth_trend, "📈 Growth Trends")
                st.plotly_chart(fig_growth_trend, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Top performing manufacturers by growth
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                manufacturer_growth = filtered_df.groupby('Manufacturer').agg(
                    avg_yoy=('yoy_growth', 'mean'),
                    avg_qoq=('qoq_growth', 'mean'),
                    total_regs=('Count', 'sum')
                ).sort_values('total_regs', ascending=False).head(15).reset_index()
                
                # Create subplot for dual axis
                fig_mfg_growth = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("YoY Growth by Top Manufacturers", "QoQ Growth by Top Manufacturers"),
                    horizontal_spacing=0.1
                )
                
                fig_mfg_growth.add_trace(
                    go.Bar(x=manufacturer_growth['Manufacturer'], y=manufacturer_growth['avg_yoy'], 
                           name='YoY Growth %', marker_color=COLORS['primary']),
                    row=1, col=1
                )
                
                fig_mfg_growth.add_trace(
                    go.Bar(x=manufacturer_growth['Manufacturer'], y=manufacturer_growth['avg_qoq'], 
                           name='QoQ Growth %', marker_color=COLORS['secondary']),
                    row=1, col=2
                )
                
                fig_mfg_growth.update_xaxes(tickangle=45)
                fig_mfg_growth.update_layout(
                    title_text="🏆 Top Manufacturer Growth Performance",
                    showlegend=False,
                    height=500
                )
                
                st.plotly_chart(fig_mfg_growth, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # --- Advanced Analytics Section ---
        if show_forecasting:
            st.markdown('<h2 class="section-header">🔮 Predictive Analytics</h2>', unsafe_allow_html=True)
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            # Simple trend forecasting (basic linear regression)
            monthly_trend = filtered_df.groupby(filtered_df['Registration_Date'].dt.to_period('M'))['Count'].sum().reset_index()
            monthly_trend['Period_Num'] = range(len(monthly_trend))
            
            # Simple linear trend
            coeffs = np.polyfit(monthly_trend['Period_Num'], monthly_trend['Count'], 1)
            trend_line = np.poly1d(coeffs)
            
            # Extend for next 6 months
            future_periods = range(len(monthly_trend), len(monthly_trend) + 6)
            future_values = [trend_line(x) for x in future_periods]
            future_dates = pd.date_range(
                start=monthly_trend['Registration_Date'].iloc[-1].to_timestamp() + pd.DateOffset(months=1),
                periods=6,
                freq='M'
            ).to_period('M')
            
            # Combine historical and forecast data
            forecast_df = pd.DataFrame({
                'Registration_Date': list(monthly_trend['Registration_Date']) + list(future_dates),
                'Count': list(monthly_trend['Count']) + future_values,
                'Type': ['Historical'] * len(monthly_trend) + ['Forecast'] * 6
            })
            forecast_df['Registration_Date_Str'] = forecast_df['Registration_Date'].astype(str)
            
            fig_forecast = px.line(
                forecast_df,
                x='Registration_Date_Str',
                y='Count',
                color='Type',
                title="🔮 6-Month Registration Forecast",
                markers=True,
                color_discrete_map={'Historical': COLORS['primary'], 'Forecast': COLORS['warning']}
            )
            
            fig_forecast = create_advanced_chart(fig_forecast, "🔮 Registration Forecast")
            st.plotly_chart(fig_forecast, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Forecast insights
            st.info(f"📊 **Forecast Insights:** Based on historical trends, next 6 months show an average of {np.mean(future_values):,.0f} registrations per month")
        
        # --- Summary Statistics ---
        st.markdown('<h2 class="section-header">📋 Data Summary & Insights</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("📊 Registration Statistics")
            
            stats_data = {
                'Metric': ['Total Records', 'Unique Manufacturers', 'Unique Vehicle Types', 'Date Range (Months)', 
                          'Average Monthly Registrations', 'Peak Registration Month', 'Minimum Registration Month'],
                'Value': [
                    f"{len(filtered_df):,}",
                    f"{filtered_df['Manufacturer'].nunique()}",
                    f"{filtered_df['vehicle_type'].nunique()}",
                    f"{filtered_df['Registration_Date'].dt.to_period('M').nunique()}",
                    f"{filtered_df['Count'].sum() / max(1, filtered_df['Registration_Date'].dt.to_period('M').nunique()):,.0f}",
                    f"{filtered_df.groupby(filtered_df['Registration_Date'].dt.to_period('M'))['Count'].sum().idxmax()}",
                    f"{filtered_df.groupby(filtered_df['Registration_Date'].dt.to_period('M'))['Count'].sum().idxmin()}"
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("🎯 Top Performers")
            
            # Top 5 manufacturers
            top_mfg = filtered_df.groupby('Manufacturer')['Count'].sum().nlargest(5)
            st.write("**Top 5 Manufacturers:**")
            for i, (mfg, count) in enumerate(top_mfg.items(), 1):
                percentage = (count / total_registrations) * 100
                st.write(f"{i}. {mfg}: {count:,} ({percentage:.1f}%)")
            
            st.write("")
            
            # Top vehicle types
            top_vehicles = filtered_df.groupby('vehicle_type')['Count'].sum().nlargest(3)
            st.write("**Top 3 Vehicle Types:**")
            for i, (vehicle, count) in enumerate(top_vehicles.items(), 1):
                percentage = (count / total_registrations) * 100
                st.write(f"{i}. {vehicle}: {count:,} ({percentage:.1f}%)")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # --- Interactive Data Explorer ---
        st.markdown('<h2 class="section-header">🔍 Data Explorer</h2>', unsafe_allow_html=True)
        
        with st.expander("📊 Interactive Data Table", expanded=False):
            # Search and filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                search_manufacturer = st.text_input("🔍 Search Manufacturer", placeholder="Enter manufacturer name...")
            
            with col2:
                sort_column = st.selectbox("Sort by", ['Registration_Date', 'Count', 'yoy_growth', 'qoq_growth'])
            
            with col3:
                sort_order = st.selectbox("Order", ['Descending', 'Ascending'])
            
            # Apply search filter
            display_df = filtered_df.copy()
            if search_manufacturer:
                display_df = display_df[display_df['Manufacturer'].str.contains(search_manufacturer, case=False, na=False)]
            
            # Apply sorting
            ascending = sort_order == 'Ascending'
            display_df = display_df.sort_values(by=sort_column, ascending=ascending)
            
            # Format the dataframe for display
            display_columns = ['Registration_Date', 'Manufacturer', 'vehicle_type', 'Count', 'yoy_growth', 'qoq_growth']
            display_df_formatted = display_df[display_columns].copy()
            display_df_formatted['Registration_Date'] = display_df_formatted['Registration_Date'].dt.strftime('%Y-%m')
            display_df_formatted['yoy_growth'] = display_df_formatted['yoy_growth'].round(2)
            display_df_formatted['qoq_growth'] = display_df_formatted['qoq_growth'].round(2)
            
            st.dataframe(
                display_df_formatted,
                use_container_width=True,
                column_config={
                    "Registration_Date": st.column_config.TextColumn("Date"),
                    "Manufacturer": st.column_config.TextColumn("Manufacturer"),
                    "vehicle_type": st.column_config.TextColumn("Vehicle Type"),
                    "Count": st.column_config.NumberColumn("Registrations", format="%d"),
                    "yoy_growth": st.column_config.NumberColumn("YoY Growth (%)", format="%.2f"),
                    "qoq_growth": st.column_config.NumberColumn("QoQ Growth (%)", format="%.2f"),
                }
            )
            
            # Export functionality
            st.download_button(
                label="📥 Download Filtered Data",
                data=display_df_formatted.to_csv(index=False),
                file_name=f"vahan_filtered_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download the currently filtered data as CSV"
            )
        
        # --- Footer ---
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <p>🚗 <strong>Vahan Registration Analytics Dashboard</strong> • Built with Streamlit & Plotly</p>
            <p>📊 Real-time Vehicle Registration Intelligence • 🔄 Last Updated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()