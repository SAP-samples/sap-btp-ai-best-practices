import streamlit as st
import pandas as pd
from src.load_data import load_acdocu
from src.align_documents import align_periods
from src.detect_variance import detect_variances
from src.classify_drivers import apply_driver_logic
from src.explain_llm import analyze_variances_with_llm
from src.wbs_analysis import perform_wbs_analysis


st.set_page_config("Rivian", layout="wide")
st.title("📊 R&D YTD Variance – SAP ACDOCU with AI Analysis")

FILE_24 = "data/ACDOCU_YTD Q3 24 R&D (Consolidation Currency Translation, Elimination and Topsides).xlsx"
FILE_25 = "data/ACDOCU_YTD Q3 25 R&D (Consolidation Currency Translation, Elimination and Topsides).xlsx"

# Sidebar configuration
st.sidebar.header("Analysis Configuration")
materiality_threshold = st.sidebar.number_input(
    "Materiality Threshold ($)", 
    min_value=0, 
    value=1000, 
    step=100,
    help="Filter variances below this absolute amount"
)

run_llm_analysis = st.sidebar.checkbox(
    "Generate AI Explanation", 
    value=True,
    help="Use Gen AI Hub to generate variance explanation"
)

# Create two columns for the main buttons
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    run_analysis_btn = st.button("Run Analysis", type="primary", use_container_width=True)

with col_btn2:
    run_wbs_analysis_btn = st.button("WBS Analysis", type="primary", use_container_width=True)

if run_analysis_btn:
    with st.spinner("Loading data..."):
        df24 = load_acdocu(FILE_24, "Q3_24")
        df25 = load_acdocu(FILE_25, "Q3_25")

    with st.spinner("Analyzing variances..."):
        aligned = align_periods(df24, df25)
        variances = detect_variances(aligned, materiality=materiality_threshold)
        classified = apply_driver_logic(variances)

    # Display document-level variances
    st.subheader("📋 Document-Level Variances")
    
    # Check if dataframe is too large for styling
    total_cells = classified.shape[0] * classified.shape[1]
    max_cells = 262144
    
    if total_cells > max_cells:
        # Display without styling for large dataframes
        st.info(f"Displaying {len(classified):,} variances (styling disabled for performance)")
        st.dataframe(classified, use_container_width=True, height=400, hide_index=True)
    else:
        # Apply styling for smaller dataframes
        st.dataframe(
            classified.style.format({
                "Amount_Q3_24": "${:,.2f}",
                "Amount_Q3_25": "${:,.2f}",
                "Delta": "${:,.2f}",
                "Abs_Delta": "${:,.2f}"
            }),
            use_container_width=True,
            height=400,
            hide_index=True
        )
    
    # Download option for document-level data
    csv_documents = classified.to_csv(index=False)
    st.download_button(
        label="Download Document-Level Data (CSV)",
        data=csv_documents,
        file_name="variance_documents.csv",
        mime="text/csv"
    )
    
    # Run comprehensive analysis
    if run_llm_analysis:
        with st.spinner("Running AI analysis (summarizing, classifying, detecting patterns)..."):
            analysis_results = analyze_variances_with_llm(classified)
        
        # Display summary
        st.subheader("📊 Variance Summary by GL & Cost Center")
        st.dataframe(
            analysis_results["summary"].style.format({
                "Total_Delta": "${:,.2f}",
                "Total_Abs_Delta": "${:,.2f}",
                "Document_Count": "{:,}"
            }),
            use_container_width=True,
            height=400
        )
        
        # Download summary
        csv_summary = analysis_results["summary"].to_csv(index=False)
        st.download_button(
            label="Download Summary Data (CSV)",
            data=csv_summary,
            file_name="variance_summary.csv",
            mime="text/csv"
        )
        
        # Display classifications
        st.subheader("🎯 Variance Classification by Magnitude")
        col1, col2, col3, col4 = st.columns(4)
        
        classifications = analysis_results["classifications"]
        with col1:
            st.metric("Critical (Top 10%)", len(classifications["Critical"]))
        with col2:
            st.metric("High (10-30%)", len(classifications["High"]))
        with col3:
            st.metric("Medium (30-60%)", len(classifications["Medium"]))
        with col4:
            st.metric("Low (Bottom 40%)", len(classifications["Low"]))
        
        # Show critical items
        if len(classifications["Critical"]) > 0:
            with st.expander("View Critical Variances", expanded=True):
                st.dataframe(
                    classifications["Critical"].style.format({
                        "Amount_Q3_24": "${:,.2f}",
                        "Amount_Q3_25": "${:,.2f}",
                        "Delta": "${:,.2f}",
                        "Abs_Delta": "${:,.2f}"
                    }),
                    use_container_width=True
                )
        
        # Display pattern insights
        st.subheader("🔍 Pattern Detection & Insights")
        st.text(analysis_results["insights_summary"])
        
        # Display text analysis insights
        text_analysis = analysis_results.get("text_analysis", {})
        if "error" not in text_analysis:
            with st.expander("📝 Transaction Description Analysis"):
                st.write(f"**Unique Descriptions:** {text_analysis['unique_descriptions']:,}")
                st.write(f"**Total Items with Descriptions:** {text_analysis['total_with_text']:,}")
                
                st.write("**Most Common Descriptions:**")
                top_descs = pd.DataFrame(
                    list(text_analysis["top_descriptions"].items())[:10],
                    columns=["Description", "Count"]
                )
                st.dataframe(top_descs, use_container_width=True)
        
        # Display LLM explanation
        st.subheader("🤖 AI-Generated Variance Explanation")
        st.markdown(analysis_results["llm_explanation"])
        
        # Display Progressive Drill-Down Results
        drilldown_results = analysis_results.get("drilldown_results", [])
        if drilldown_results:
            st.subheader("🔬 Progressive Drill-Down Analysis (Top Variances)")
            st.info(f"Analyzed {len(drilldown_results)} top variances with progressive drill-down to WBS/Project/Program levels")
            
            # Create summary dataframe
            drill_summary = []
            for result in drilldown_results:
                drill_summary.append({
                    "GL Account": result.gl_account,
                    "Cost Center": result.cost_center,
                    "Driver": result.driver,
                    "Amount": f"${result.amount:,.2f}",
                    "Investigation Path": " → ".join(result.drill_path),
                    "Confidence": result.confidence,
                    "Escalation": "Yes" if result.requires_escalation else "No"
                })
            
            drill_df = pd.DataFrame(drill_summary)
            st.dataframe(drill_df, use_container_width=True, hide_index=True)
            
            # Show detailed drill-down for top items
            with st.expander("📊 Detailed Drill-Down Evidence", expanded=False):
                for i, result in enumerate(drilldown_results[:5], 1):
                    st.markdown(f"### {i}. GL {result.gl_account} | CC {result.cost_center} | {result.driver}")
                    st.write(f"**Amount:** ${result.amount:,.2f}")
                    st.write(f"**Confidence:** {result.confidence}")
                    st.write(f"**Investigation Path:** {' → '.join(result.drill_path)}")
                    
                    st.markdown("**Root Cause:**")
                    st.info(result.root_cause)
                    
                    # Show evidence collected at each level
                    st.markdown("**Evidence Chain:**")
                    for level, evidence in result.evidence_chain.items():
                        if evidence.get('found', False):
                            st.success(f"✅ {level}: Found")
                            if 'top_contributors' in evidence and evidence['top_contributors']:
                                # Show details in a collapsed section using markdown
                                with st.container():
                                    st.markdown(f"**{level} Top Contributors:**")
                                    contrib_df = pd.DataFrame(evidence['top_contributors'])
                                    st.dataframe(contrib_df, use_container_width=True, hide_index=True)
                        else:
                            st.warning(f"⚠️ {level}: {evidence.get('missing', 'Not found')}")
                    
                    if result.requires_escalation:
                        st.error(f"**Escalation Required:** {result.escalation_reason}")
                        if result.missing_data:
                            st.write(f"**Missing Data:** {', '.join(result.missing_data)}")
                    
                    st.markdown("---")
        
        # Display escalations
        escalations = analysis_results.get("escalations", [])
        if escalations:
            st.subheader("⚠️ Items Requiring Escalation")
            st.warning(f"Found {len(escalations)} items that need cost center owner clarification")
            
            escalation_df = pd.DataFrame(escalations)
            
            # Handle Reasons column if it's a list
            if 'Reasons' in escalation_df.columns:
                escalation_df['Reasons'] = escalation_df['Reasons'].apply(
                    lambda x: '; '.join(x) if isinstance(x, list) else x
                )
            
            st.dataframe(
                escalation_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Download escalations
            csv_escalations = escalation_df.to_csv(index=False)
            st.download_button(
                label="Download Escalation List (CSV)",
                data=csv_escalations,
                file_name="escalations_for_review.csv",
                mime="text/csv"
            )
        else:
            st.success("✅ No escalations needed - all variances have sufficient evidence")
        
        # Additional metrics
        st.subheader("📈 Key Metrics")
        patterns = analysis_results["patterns"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Total Variance", 
                f"${patterns['concentration']['total_variance']:,.2f}"
            )
        with col2:
            net_var = patterns["sign_analysis"]["net_variance"]
            st.metric(
                "Net Variance", 
                f"${net_var:,.2f}",
                delta=f"{'Increase' if net_var > 0 else 'Decrease'}"
            )
        with col3:
            concentration = patterns["concentration"]["concentration_ratio"]
            st.metric(
                "Top 20% Concentration", 
                f"{concentration*100:.1f}%"
            )
    else:
        st.info("AI analysis disabled. Enable it in the sidebar to get summaries, classifications, and pattern detection.")

# WBS Analysis Section
if run_wbs_analysis_btn:
    with st.spinner("Loading data..."):
        df24 = load_acdocu(FILE_24, "Q3_24")
        df25 = load_acdocu(FILE_25, "Q3_25")
    
    with st.spinner("Performing WBS variance analysis..."):
        wbs_results = perform_wbs_analysis(
            df24, df25, 
            materiality_threshold=materiality_threshold,
            run_llm=run_llm_analysis
        )
    
    # Check if there's an error (no WBS data found)
    if "error" in wbs_results:
        st.error(f"⚠️ {wbs_results['error']}")
        st.info("WBS analysis requires transactions with WBS elements, Internal Orders, or Order in the data.")
    else:
        st.success(f"✅ WBS Analysis completed: {len(wbs_results['aligned_data'])} WBS/GL combinations found")
        
        # Display WBS Summary
        st.subheader("📊 Summary by WBS Element")
        wbs_summary = wbs_results['wbs_summary']
        
        if not wbs_summary.empty:
            # Format and display WBS summary
            st.dataframe(
                wbs_summary.style.format({
                    "Total_Q3_24": "${:,.2f}",
                    "Total_Q3_25": "${:,.2f}",
                    "Net_Variance": "${:,.2f}",
                    "Total_Abs_Variance": "${:,.2f}",
                    "GL_Account_Count": "{:,}"
                }),
                use_container_width=True,
                height=400,
                hide_index=True
            )
            
            # Download WBS summary
            csv_wbs_summary = wbs_summary.to_csv(index=False)
            st.download_button(
                label="Download WBS Summary (CSV)",
                data=csv_wbs_summary,
                file_name="wbs_variance_summary.csv",
                mime="text/csv"
            )
        else:
            st.warning("No WBS elements found with significant variances.")
        
        # Display Program Summary
        st.subheader("🎯 Summary by Program")
        program_summary = wbs_results['program_summary']
        
        if not program_summary.empty:
            st.dataframe(
                program_summary.style.format({
                    "Total_Q3_24": "${:,.2f}",
                    "Total_Q3_25": "${:,.2f}",
                    "Net_Variance": "${:,.2f}",
                    "Total_Abs_Variance": "${:,.2f}",
                    "WBS_Count": "{:,}"
                }),
                use_container_width=True,
                height=400,
                hide_index=True
            )
            
            # Download program summary
            csv_program_summary = program_summary.to_csv(index=False)
            st.download_button(
                label="Download Program Summary (CSV)",
                data=csv_program_summary,
                file_name="program_variance_summary.csv",
                mime="text/csv"
            )
        else:
            st.info("No program hierarchy information found for WBS elements.")
        
        # Display Critical WBS Items
        critical_items = wbs_results['critical_items']
        if not critical_items.empty:
            st.subheader("🔴 Critical WBS Elements (Top 10%)")
            st.dataframe(
                critical_items.style.format({
                    "Amount_Q3_24": "${:,.2f}",
                    "Amount_Q3_25": "${:,.2f}",
                    "Delta": "${:,.2f}",
                    "Abs_Delta": "${:,.2f}",
                    "Percent_Change": "{:.1f}%"
                }),
                use_container_width=True,
                height=300,
                hide_index=True
            )
        
        # Display Insights
        st.subheader("🔍 WBS Pattern Analysis")
        st.text(wbs_results['insights'])
        
        # Display LLM Explanation
        if run_llm_analysis and wbs_results['llm_explanation']:
            st.subheader("🤖 AI-Generated WBS Analysis Explanation")
            st.markdown(wbs_results['llm_explanation'])
        
        # Display detailed WBS data
        with st.expander("📋 View Detailed Data by WBS/GL/CC"):
            aligned_data = wbs_results['aligned_data']
            st.dataframe(
                aligned_data.style.format({
                    "Amount_Q3_24": "${:,.2f}",
                    "Amount_Q3_25": "${:,.2f}",
                    "Delta": "${:,.2f}",
                    "Abs_Delta": "${:,.2f}",
                    "Percent_Change": "{:.1f}%"
                }),
                use_container_width=True,
                height=400,
                hide_index=True
            )
            
            # Download detailed data
            csv_detailed = aligned_data.to_csv(index=False)
            st.download_button(
                label="Download Detailed WBS Data (CSV)",
                data=csv_detailed,
                file_name="wbs_detailed_variance.csv",
                mime="text/csv",
                key="detailed_wbs_download"
            )
        
        # Key Metrics
        st.subheader("📈 WBS Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        if not wbs_summary.empty:
            total_net_variance = wbs_summary['Net_Variance'].sum()
            total_abs_variance = wbs_summary['Total_Abs_Variance'].sum()
            unique_wbs = len(wbs_summary)
            unique_programs = len(program_summary) if not program_summary.empty else 0
            
            with col1:
                st.metric("Total Net Variance", f"${total_net_variance:,.2f}")
            with col2:
                st.metric("Total Absolute Variance", f"${total_abs_variance:,.2f}")
            with col3:
                st.metric("Unique WBS Elements", f"{unique_wbs:,}")
            with col4:
                st.metric("Programs Affected", f"{unique_programs:,}")

st.sidebar.markdown("---")
st.sidebar.caption("Powered by SAP Gen AI Hub")
