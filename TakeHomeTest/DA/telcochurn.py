import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# PAGE CONFIG

st.set_page_config(page_title="Customer Churn Dashboard", page_icon="📉", layout="wide")

# LOAD DATA

@st.cache_data
def load_data():
    return pd.read_csv(r"D:\TakeHomeTest\DA\customer_churn_dashboard.csv") 

df = load_data()

# SIDEBAR NAVIGATION

menu = st.sidebar.radio("Navigation", ["📌 Project Overview", "📊 Churn Dashboard","🔍 Churn Driver Analysis", "📌 Segmentation & Cohort Analysis"])

# PAGE 1 : BUSINESS OVERVIEW

if menu == "📌 Project Overview":
    st.title("Customer Churn Analysis: Segmentation and Cohort Insights for Retention Strategy")

    st.subheader("Project Overview")
    st.write("""
    Proyek ini bertujuan untuk menganalisis perilaku churn pelanggan pada perusahaan layanan telekomunikasi
    untuk mendapatkan insight actionable dalam meningkatkan strategi retensi pelanggan dan optimasi pendapatan.
    """)

    st.subheader("Business Problem")
    st.write("""
    Perusahaan menghadapi tingkat churn yang cukup tinggi (±26%), terutama pelanggan baru dan kontrak jangka pendek.
    Dampaknya besar terhadap revenue dan Customer Lifetime Value (CLTV).
    
    Pertanyaan bisnis utama:
    - Faktor apa yang paling mempengaruhi churn pelanggan?
    - Kapan churn paling sering terjadi dalam lifecycle pelanggan?
    - Segmen pelanggan mana yang paling berisiko namun bernilai tinggi?
    - Berapa revenue yang hilang akibat churn?
    """)

    st.subheader("Business Goals")
    st.write("""
    🎯 Mengurangi churn rate  
    🎯 Mengidentifikasi pelanggan berisiko churn lebih awal  
    🎯 Mengurangi revenue loss  
    🎯 Meningkatkan CLTV melalui retensi & upsell  
    🎯 Menyediakan dashboard interaktif berbasis data untuk decision making
    """)

# PAGE 2 : CHURN DASHBOARD
elif menu == "📊 Churn Dashboard":
    st.title("📊 Customer Churn Dashboard")

    st.markdown("### 🔍 Filters")
    colf1, colf2, colf3, col_reset = st.columns([2, 2, 2, 1])

    with colf1:
        contract_filter = st.multiselect(
            "Contract",
            options=df["contract"].unique(),
            default=df["contract"].unique()
        )

    with colf2:
        cohort_filter = st.multiselect(
            "Tenure Cohort",
            options=df["tenure_cohort"].unique(),
            default=df["tenure_cohort"].unique()
        )

    with colf3:
        cluster_filter = st.multiselect(
            "Cluster",
            options=sorted(df["cluster"].unique()),
            default=sorted(df["cluster"].unique())
        )

    if col_reset.button("Reset"):
        contract_filter = df["contract"].unique()
        cohort_filter = df["tenure_cohort"].unique()
        cluster_filter = sorted(df["cluster"].unique())

    # Filter data
    dff = df[
        (df["contract"].isin(contract_filter)) &
        (df["tenure_cohort"].isin(cohort_filter)) &
        (df["cluster"].isin(cluster_filter))
    ]

    st.markdown("---")

    # KPI ROW
    total_customers = len(dff)
    total_churn = dff["churn_value"].sum()
    churn_rate = (total_churn / total_customers) * 100 if total_customers > 0 else 0
    revenue_lost = dff[dff["churn_value"] == 1]["total_revenue"].sum()
    avg_cltv = dff["cltv"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Total Churn Customers", f"{total_churn:,}")
    col3.metric("Churn Rate", f"{churn_rate:.2f}%")
    col4.metric("Revenue Lost", f"${revenue_lost:,.0f}")
    col5.metric("Avg CLTV", f"${avg_cltv:,.0f}")

    st.markdown("---")

    colA, colB = st.columns([2, 1])

    # Churn by Tenure Cohort — diurutkan berdasarkan churn rate terbesar
    churn_by_cohort = dff.groupby("tenure_cohort")["churn_value"].mean().reset_index()
    churn_by_cohort = churn_by_cohort.sort_values("churn_value", ascending=False)

    fig1 = px.bar(
        churn_by_cohort,
        x="churn_value", y="tenure_cohort", orientation="h",
        title="Churn Rate by Tenure Cohort",
        labels={"churn_value": "Churn Rate (%)", "tenure_cohort": "Tenure Cohort"},
        color="churn_value",
        color_continuous_scale="Reds"
    )
    fig1.update_layout(xaxis_tickformat=".0%", height=350)
    colA.plotly_chart(fig1, use_container_width=True)

    # Churn Distribution — ganti judul
    fig2 = px.pie(
        dff, names="churn_label",
        title="Customer by Churn",
        hole=0.4
    )
    colB.plotly_chart(fig2, use_container_width=True)

    # Revenue Lost by Cluster
    fig3 = px.bar(
        dff[dff["churn_value"] == 1].groupby("cluster")["total_revenue"].sum().reset_index(),
        x="cluster", y="total_revenue",
        title="Revenue Lost by Cluster",
        labels={"total_revenue": "Revenue Lost"}
    )
    fig3.update_layout(height=350)

    # Churn by Contract Type
    fig4 = px.bar(
        dff.groupby("contract")["churn_value"].mean().reset_index(),
        x="contract", y="churn_value",
        title="Churn Rate by Contract",
        labels={"churn_value": "Churn Rate (%)"}
    )
    fig4.update_layout(yaxis_tickformat=".0%", height=350)

    colC, colD = st.columns(2)
    colC.plotly_chart(fig3, use_container_width=True)
    colD.plotly_chart(fig4, use_container_width=True)

    st.info("💡 High churn detected on **Month-to-month + Low Tenure**. Retention action recommended")
# PAGE 3 : CHURN DRIVER ANALYSIS 

elif menu == "🔍 Churn Driver Analysis":

    st.markdown(
        "<h1 style='color:#004e89; font-weight:700;'>🔍 Churn Driver Analysis</h1>",
        unsafe_allow_html=True
    )

    # Premium Top Filter Bar
    st.markdown("### 🎛 Advanced Filters")
    colf1, colf2, colf3 = st.columns(3)

    with colf1:
        contract_filter = st.multiselect(
            "Contract",
            options=df['contract'].dropna().unique().tolist(),
            default=df['contract'].dropna().unique().tolist()
        )

    with colf2:
        cohort_filter = st.multiselect(
            "Tenure Cohort",
            options=df['tenure_cohort'].dropna().unique().tolist(),
            default=df['tenure_cohort'].dropna().unique().tolist()
        )

    with colf3:
        cluster_filter = st.multiselect(
            "Cluster",
            options=sorted(df['cluster'].dropna().unique().tolist()),
            default=sorted(df['cluster'].dropna().unique().tolist())
        )

    # Apply Filter
    df_filtered = df[
        (df['contract'].isin(contract_filter)) &
        (df['tenure_cohort'].isin(cohort_filter)) &
        (df['cluster'].isin(cluster_filter))
    ]

    st.markdown("---")
    colA, colB = st.columns([2, 1])

    # WORDCLOUD — CHURN REASON
    
    colA.subheader("☁️ Top Churn Reasons")

    churn_reasons = df_filtered[df_filtered["churn_value"] == 1]["churn_reason"].dropna()

    if not churn_reasons.empty:
        text = " ".join(churn_reasons)
        wc = WordCloud(width=1200, height=600, background_color="white").generate(text)
        fig_wc, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        colA.pyplot(fig_wc)
    else:
        colA.info("⚠ No churn reason data for selected filters.")

    # PAYMENT METHOD IMPACT
    colB.subheader("💳 Payment Method Impact")

    if "payment_method" in df_filtered.columns:
        pay = df_filtered.groupby("payment_method")["churn_value"].mean().reset_index()
        pay["Churn Rate (%)"] = pay["churn_value"] * 100

        fig_pay = px.bar(
            pay.sort_values("Churn Rate (%)", ascending=False),
            x="Churn Rate (%)", y="payment_method",
            orientation="h",
            title="Churn Rate by Payment Method",
            color="Churn Rate (%)",
            color_continuous_scale="Blues"
        )
        fig_pay.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
        fig_pay.update_layout(height=350)
        colB.plotly_chart(fig_pay, use_container_width=True)
    else:
        colB.warning("Payment method data not available.")

    st.markdown("---")

    # ADD-ON SERVICE IMPACT
    st.subheader("🔌 Add-On Service Impact on Churn")

    service_cols = [
        'online_security', 'online_backup', 'device_protection',
        'tech_support', 'streaming_tv', 'streaming_movies', 'streaming_music'
    ]

    impact = []
    for col in service_cols:
        if col in df_filtered.columns:
            churn_yes = df_filtered[df_filtered[col] == "Yes"]["churn_value"].mean()
            impact.append({"Service": col.replace("_", " ").title(),
                           "Churn Rate (%)": churn_yes * 100})

    df_impact = pd.DataFrame(impact).sort_values("Churn Rate (%)", ascending=False)

    fig_service = px.bar(
        df_impact,
        x="Churn Rate (%)", y="Service",
        orientation="h",
        title="Which Optional Services Drive Churn?",
        color="Churn Rate (%)",
        color_continuous_scale="Aggrnyl"
    )
    fig_service.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
    fig_service.update_layout(height=420)
    st.plotly_chart(fig_service, use_container_width=True)

    st.markdown("---")

    # SATISFACTION vs CHURN
    st.subheader("😊 Customer Satisfaction vs Churn")

    fig_sat = px.box(
        df_filtered,
        x="churn_label",
        y="satisfaction_score",
        color="churn_label",
        title="Do Dissatisfied Customers Churn More?",
        color_discrete_sequence=["#1f4068", "#e63946"]
    )
    fig_sat.update_layout(height=450)
    st.plotly_chart(fig_sat, use_container_width=True)

    st.info(
        "📌 Insight: Churn is heavily influenced by **payment friction**, **service issues** & "
        "**low satisfaction** — targeting proactive retention & upsell recommended! 🚀"
    )
# PAGE 4 : SEGMENTATION & COHORT ANALYSIS
elif menu == "📌 Segmentation & Cohort Analysis":

    st.markdown(
        "<h1 style='color:#004e89; font-weight:700;'>👥 Segmentation & Cohort Analysis : Who Should Be Prioritized?</h1>",
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_clusters = st.multiselect(
            "Cluster (Customer Group)", sorted(df["cluster"].dropna().unique().tolist()), []
        )
    with col2:
        selected_contracts = st.multiselect(
            "Contract Type", sorted(df["contract"].dropna().unique().tolist()), []
        )
    with col3:
        selected_cohorts = st.multiselect(
            "Tenure Cohort", sorted(df["tenure_cohort"].dropna().unique().tolist()), []
        )

    # Copy & apply filter
    df_seg = df.copy()
    if selected_clusters:
        df_seg = df_seg[df_seg["cluster"].isin(selected_clusters)]
    if selected_contracts:
        df_seg = df_seg[df_seg["contract"].isin(selected_contracts)]
    if selected_cohorts:
        df_seg = df_seg[df_seg["tenure_cohort"].isin(selected_cohorts)]

    st.markdown("---")

    # CLUSTER PROFILING TABLE
    st.subheader("📊 Cluster Profiling Table")

    if df_seg.empty:
        st.warning("⚠️ No data found based on filter selection.")
        st.stop()

    # Tambahkan beberapa kolom tambahan & karakteristik singkat
    def get_characteristics(row):
        if row['churn_rate'] > 30:
            return "High Risk / Low Loyalty"
        elif row['avg_total_revenue'] > 5000:
            return "High Value / Loyal"
        else:
            return "Low Engagement / Stable"

    profile = df_seg.groupby("cluster").agg(
        customers=("customer_id", "count"),
        avg_tenure=("tenure", "mean"),
        avg_monthly_charges=("monthly_charges", "mean"),
        avg_total_revenue=("total_revenue", "mean"),
        avg_cltv=("cltv", "mean"),
        avg_gb_download=("avg_monthly_gb_download", "mean"),
        avg_services=("total_services_used", "mean"),
        churn_rate=("churn_value", "mean")
    ).reset_index()

    profile["churn_rate"] = profile["churn_rate"] * 100
    profile["characteristics"] = profile.apply(get_characteristics, axis=1)

    st.dataframe(
        profile.style.format({
            "avg_tenure": "{:.1f}",
            "avg_monthly_charges": "${:,.1f}",
            "avg_total_revenue": "${:,.1f}",
            "avg_cltv": "${:,.1f}",
            "avg_gb_download": "{:.1f} GB",
            "avg_services": "{:.1f}",
            "churn_rate": "{:.1f}%"
        })
    )

    st.markdown("---")

    # CHURN RATE BY CLUSTER
    st.subheader("📈 Churn Rate by Cluster")

    fig1 = px.bar(
        profile,
        x="cluster",
        y="churn_rate",
        text=profile["churn_rate"].map(lambda x: f"{x:.1f}%"),
        color="churn_rate",
        color_continuous_scale=["#1f4068", "#e63946"]
    )
    fig1.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # REVENUE LOST BY CLUSTER
    st.subheader("💸 Total Revenue Lost by Cluster")

    df_seg["revenue_lost"] = df_seg["total_revenue"] * df_seg["churn_value"]
    lost = df_seg.groupby("cluster")["revenue_lost"].sum().reset_index()

    fig_lost = px.bar(
        lost,
        x="cluster",
        y="revenue_lost",
        text=lost["revenue_lost"].map(lambda x: f"${x:,.0f}"),
        color="revenue_lost",
        color_continuous_scale="Reds"
    )
    fig_lost.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_lost, use_container_width=True)

    st.success("👉 Focus on segments with **high churn** and **high revenue loss**!")

    # CHURN RATE BY TENURE COHORT
    st.subheader("📈 Churn Rate by Tenure Cohort")

    # Pastikan tenure cohort ordered
    ordered_tenure = ["0–3", "4–6", "7–12", "13–24", "25+"]
    df_seg["tenure_cohort"] = pd.Categorical(
        df_seg["tenure_cohort"],
        categories=ordered_tenure,
        ordered=True
    )

    churn_tenure = (
        df_seg.groupby("tenure_cohort")["churn_value"]
        .mean()
        .mul(100)
        .reset_index()
    )

    fig_tenure = px.line(
        churn_tenure,
        x="tenure_cohort",
        y="churn_value",
        markers=True,
        text=churn_tenure["churn_value"].map(lambda x: f"{x:.1f}%"),
        title="Are Newer Customers More Likely to Churn?",
    )

    fig_tenure.update_traces(line=dict(width=3), marker=dict(size=10))
    fig_tenure.update_layout(
        yaxis_title="Churn Rate (%)",
        xaxis_title="Tenure Cohort",
        height=400,
        showlegend=False,
        coloraxis_showscale=False
    )

    # Styling warna merah untuk churn severity
    fig_tenure.update_traces(line_color="#e63946", marker_color="#e63946")

    st.plotly_chart(fig_tenure, use_container_width=True)

    st.markdown("---")
    # 📌 CLUSTER QUALITY CHECK — Silhouette Score
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    st.subheader("🎯 Cluster Quality & Visualization")

    # Hitung silhouette score pakai numeric features yang relevan
    numeric_cols = ["tenure", "total_revenue", "cltv", "monthly_charges", "total_services_used"]

    df_numeric = df_seg[numeric_cols]
    df_numeric = (df_numeric - df_numeric.min()) / (df_numeric.max() - df_numeric.min())

    sil_score = silhouette_score(df_numeric, df_seg["cluster"])

    st.metric("Silhouette Score", f"{sil_score:.3f}", 
              help="Semakin mendekati 1 semakin baik. <0.25 = weak cluster")

    if sil_score < 0.25:
        st.warning("⚠️ Clustering masih lemah. Pertimbangkan revisi jumlah cluster atau variabel 📌")
    else:
        st.success("🟢 Clustering cukup baik! Lanjut eksplorasi segmentasi 🚀")

    # ⭐ 2D Cluster Plot with PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_numeric)

    df_plot = pd.DataFrame({
        "PCA1": pca_result[:, 0],
        "PCA2": pca_result[:, 1],
        "cluster": df_seg["cluster"].astype(str)
    })

    fig_cluster = px.scatter(
        df_plot,
        x="PCA1",
        y="PCA2",
        color="cluster",
        title="Cluster Visualization (PCA 2D)",
        symbol="cluster",
        opacity=0.8,
        color_discrete_sequence=px.colors.qualitative.Bold,
        hover_data=["cluster"]
    )

    fig_cluster.update_layout(height=500)
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.markdown("---")


