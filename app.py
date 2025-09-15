import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Document Processing Impact Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def calculate_zero_corrections_impact(historical_df, current_rates_df, new_error_rates):
    """
    Calculate the expected % of documents without corrections after improving field-level accuracy.
    
    Parameters:
    - historical_df: DataFrame with historical correction patterns (1=correct, 0=needs correction)
    - current_rates_df: DataFrame with current error rates for each field
    - new_error_rates: Dictionary with new error rates for fields to improve
    
    Returns:
    - Dictionary with current %, new %, and improvement metrics
    """
    
    # Get field names (exclude 'docs' column)
    field_columns = [col for col in historical_df.columns if col != 'docs']
    
    # Work with a copy to avoid modifying original
    historical_df = historical_df.copy()
    
    # Convert docs column to numeric - handle string formatting (same as debug function)
    historical_df['docs'] = historical_df['docs'].astype(str).str.replace(',', '').str.replace(' ', '')
    historical_df['docs'] = pd.to_numeric(historical_df['docs'], errors='coerce')
    
    # Convert field columns to numeric (ensure 0s and 1s are integers)
    for col in field_columns:
        historical_df[col] = pd.to_numeric(historical_df[col], errors='coerce')
    
    # Remove rows with NaN values
    historical_df = historical_df.dropna()
    
    # Extract current error rates and convert to numeric
    current_error_rates = {}
    for field in field_columns:
        rate_value = current_rates_df.iloc[0][field]
        current_error_rates[field] = float(rate_value) if pd.notna(rate_value) else 0.0
    
    # Create updated error rates (use new rates if provided, otherwise keep current)
    updated_error_rates = current_error_rates.copy()
    updated_error_rates.update(new_error_rates)

    # Calculate current baseline (use same logic as debug function)
    total_docs = int(historical_df['docs'].sum())
    field_sums = historical_df[field_columns].sum(axis=1)  # Calculate field sums like debug
    zero_correction_mask = field_sums == len(field_columns)  # Use field sums approach
    current_zero_correction_docs = int(historical_df[zero_correction_mask]['docs'].sum())
    
    current_percentage = (current_zero_correction_docs / total_docs) * 100 if total_docs > 0 else 0
    
    # Calculate new document counts for each error pattern
    new_zero_correction_docs = float(current_zero_correction_docs)  # Start with docs that already have zero corrections
    
    # Process each row (skip rows that already have zero corrections)
    for idx, row in historical_df.iterrows():
        # Check if this row has any corrections (any field = 0)
        correction_fields = [field for field in field_columns if row[field] == 0]
        
        if len(correction_fields) > 0:  # Only process rows with corrections
            # Calculate reweighting factor
            old_prob = 1.0
            new_prob = 1.0
            
            for field in correction_fields:
                old_rate = current_error_rates.get(field, 0.0)
                new_rate = updated_error_rates.get(field, 0.0)
                old_prob *= old_rate
                new_prob *= new_rate
            
            # Calculate new document count for this error pattern
            factor = new_prob / old_prob if old_prob > 0 else 0
            new_docs_with_this_pattern = float(row['docs']) * factor
            
            # Documents that move from this error pattern to zero corrections
            docs_moved_to_zero = float(row['docs']) - new_docs_with_this_pattern
            new_zero_correction_docs += docs_moved_to_zero
    
    # Ensure no NaN values in final calculations
    new_zero_correction_docs = new_zero_correction_docs if pd.notna(new_zero_correction_docs) else current_zero_correction_docs
    
    # Calculate new percentage
    new_percentage = (new_zero_correction_docs / total_docs) * 100 if total_docs > 0 else 0
    
    # Calculate improvement metrics (with NaN handling)
    absolute_improvement = new_percentage - current_percentage if pd.notna(new_percentage) and pd.notna(current_percentage) else 0
    relative_improvement = (absolute_improvement / current_percentage) * 100 if current_percentage > 0 and pd.notna(current_percentage) else 0
    
    # Ensure all values are valid numbers before converting to int
    safe_total_docs = int(total_docs) if pd.notna(total_docs) else 0
    safe_current_zero = int(current_zero_correction_docs) if pd.notna(current_zero_correction_docs) else 0
    safe_new_zero = int(new_zero_correction_docs) if pd.notna(new_zero_correction_docs) else safe_current_zero
    safe_additional = safe_new_zero - safe_current_zero
    
    return {
        'current_percentage': round(current_percentage, 2) if pd.notna(current_percentage) else 0,
        'new_percentage': round(new_percentage, 2) if pd.notna(new_percentage) else 0,
        'absolute_improvement': round(absolute_improvement, 2),
        'relative_improvement': round(relative_improvement, 2),
        'total_documents': safe_total_docs,
        'current_zero_correction_docs': safe_current_zero,
        'new_zero_correction_docs': safe_new_zero,
        'additional_zero_correction_docs': safe_additional,
        'current_error_rates': current_error_rates,
        'updated_error_rates': updated_error_rates
    }

# Main App
def main():
    st.title("📊 Document Processing Impact Calculator")
    st.markdown("**Predict the impact of improving field-level accuracy on overall document processing quality**")
    
    # Sidebar for file uploads
    st.sidebar.header("📁 Data Upload")
    
    # File uploads
    historical_file = st.sidebar.file_uploader(
        "Upload Historical Data CSV", 
        type=['csv'],
        help="CSV with historical correction patterns (1=correct, 0=needs correction)"
    )
    
    current_rates_file = st.sidebar.file_uploader(
        "Upload Current Error Rates CSV",
        type=['csv'],
        help="CSV with current error rates for each field"
    )
    
    if historical_file is not None and current_rates_file is not None:
        try:
            # Load the data
            historical_df = pd.read_csv(historical_file)
            current_rates_df = pd.read_csv(current_rates_file)
            
            # Validate data
            if 'docs' not in historical_df.columns:
                st.error("Historical data must contain a 'docs' column")
                return
                
            # Get field columns
            field_columns = [col for col in historical_df.columns if col != 'docs']
            
            # Display data info
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Historical Data")
                st.write(f"**Rows:** {len(historical_df):,}")
                st.write(f"**Total Documents:** {pd.to_numeric(historical_df['docs'].astype(str).str.replace(',', ''), errors='coerce').sum():,.0f}")
                st.write(f"**Fields:** {len(field_columns)}")
                
            with col2:
                st.subheader("🎯 Current Error Rates")
                st.write(f"**Fields covered:** {len(field_columns)}")
                avg_error_rate = current_rates_df.iloc[0][field_columns].mean()
                st.write(f"**Average Error Rate:** {avg_error_rate:.1%}")
            
            # Show data previews
            with st.expander("📋 Data Preview", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Historical Data (first 5 rows):**")
                    st.dataframe(historical_df.head())
                with col2:
                    st.write("**Current Error Rates:**")
                    st.dataframe(current_rates_df)
            
            st.markdown("---")
            
            # Error Rate Adjustment Section
            st.header("🔧 Adjust Error Rates")
            st.markdown("Set new error rates for each field to see the impact on document processing quality.")
            
            # Individual field adjustments only
            st.subheader("Adjust Individual Fields")
            
            current_rates = current_rates_df.iloc[0][field_columns].to_dict()
            
            # Add reset button that clears session state
            if st.button("🔄 Reset All Fields to Current Rates", help="Reset all fields to their current error rates"):
                # Clear all field-related session state
                for key in list(st.session_state.keys()):
                    if key.startswith("field_"):
                        del st.session_state[key]
                st.rerun()
            
            # Create columns for field inputs
            num_cols = 3
            cols = st.columns(num_cols)
            new_error_rates = {}
            
            for i, field in enumerate(field_columns):
                col_idx = i % num_cols
                with cols[col_idx]:
                    current_rate = float(current_rates[field])
                    
                    new_rate = st.number_input(
                        f"**{field.replace('_', ' ').title()}**",
                        min_value=0.0,
                        max_value=1.0,
                        value=current_rate,
                        step=0.001,
                        format="%.3f",
                        help=f"Current: {current_rate:.1%}",
                        key=f"field_{field}"
                    )
                    new_error_rates[field] = new_rate
            
            st.markdown("---")
            
            # Show debug info for input values before calculation
            with st.expander("🔍 Input Values Debug", expanded=False):
                st.write("**Values being passed to calculation function:**")
                current_rates = current_rates_df.iloc[0][field_columns].to_dict()
                
                for field in field_columns:
                    current_rate = float(current_rates[field])
                    new_rate = new_error_rates.get(field, current_rate)
                    diff = new_rate - current_rate
                    st.write(f"- **{field}**: Current={current_rate:.6f}, New={new_rate:.6f}, Diff={diff:.6f}")
            
            # Calculate and Display Results
            if st.button("🚀 Calculate Impact", type="primary", use_container_width=True):
                with st.spinner("Calculating impact..."):
                    try:
                        result = calculate_zero_corrections_impact(historical_df, current_rates_df, new_error_rates)
                        
                        # Display results
                        st.header("📊 Impact Analysis Results")
                        
                        # Key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Current Zero-Correction %",
                                f"{result['current_percentage']:.1f}%",
                                help="Percentage of documents currently processed without any corrections"
                            )
                        
                        with col2:
                            st.metric(
                                "New Zero-Correction %",
                                f"{result['new_percentage']:.1f}%",
                                delta=f"+{result['absolute_improvement']:.1f}pp",
                                help="Predicted percentage after improvements"
                            )
                        
                        with col3:
                            st.metric(
                                "Additional Perfect Docs",
                                f"{result['additional_zero_correction_docs']:,}",
                                help="Additional documents that would need zero corrections"
                            )
                        
                        with col4:
                            st.metric(
                                "Relative Improvement",
                                f"+{result['relative_improvement']:.1f}%",
                                help="Percentage improvement over current baseline"
                            )
                        
                        # Export results
                        st.subheader("💾 Export Results")
                        
                        # Create summary report
                        summary_report = f"""
# Document Processing Impact Analysis Report

## Summary
- **Total Documents Analyzed:** {result['total_documents']:,}
- **Current Zero-Correction Rate:** {result['current_percentage']:.1f}%
- **Predicted Zero-Correction Rate:** {result['new_percentage']:.1f}%
- **Absolute Improvement:** +{result['absolute_improvement']:.1f} percentage points
- **Additional Perfect Documents:** {result['additional_zero_correction_docs']:,}

## Field Improvements
"""
                        for field in field_columns:
                            current = result['current_error_rates'][field]
                            new = result['updated_error_rates'][field]
                            summary_report += f"- **{field.replace('_', ' ').title()}:** {current:.1%} → {new:.1%} ({new-current:+.1%})\n"
                        
                        st.download_button(
                            "📄 Download Report",
                            summary_report,
                            file_name="document_processing_impact_report.md",
                            mime="text/markdown"
                        )
                        
                    except Exception as e:
                        st.error(f"Error calculating impact: {str(e)}")
                        st.error("Please check your data format and try again.")
        
        except Exception as e:
            st.error(f"Error loading files: {str(e)}")
            st.error("Please check your CSV format and try again.")
    
    else:
        # Instructions when no files uploaded
        st.info("👆 Please upload both CSV files in the sidebar to begin analysis.")
        
        with st.expander("📖 How to use this tool", expanded=True):
            st.markdown("""
            ### Required Files:
            
            1. **Historical Data CSV**: Contains correction patterns from past documents
               - Must have a 'docs' column with document counts
               - Each field column should contain 1 (correct) or 0 (needs correction)
               
            2. **Current Error Rates CSV**: Contains current error rates for each field
               - Single row with error rates as decimals (e.g., 0.15 for 15% error rate)
               - Column names must match historical data fields
            
            ### Steps:
            1. Upload both CSV files
            2. Review your data in the preview section
            3. Adjust error rates for individual fields
            4. Click "Calculate Impact" to see results
            5. Export your analysis report
            """)

if __name__ == "__main__":
    main()