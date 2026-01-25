import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import load_data

st.title("🔗 Relaties en multivariate analyse")
df = load_data()

# Check of data geladen is
if df.empty:
    st.error("Geen data beschikbaar.")
    st.stop()

year_pca = st.selectbox("Kies jaar analyse", sorted(df['jaar'].unique()))

# --- SCATTERPLOTS ---
c1, c2 = st.columns(2)
with c1:
    fig_scat1 = px.scatter(df[df['jaar']==year_pca], x="diepte_m", y="bedekking_pct", color="groeivorm", 
                           title="Diepte vs Bedekking")
    st.plotly_chart(fig_scat1, use_container_width=True)
with c2:
    # Aggregeren per locatie voor soortenrijkdom
    df_div = df[df['jaar']==year_pca].groupby('locatie_id').agg({
        'soort': 'nunique', 'doorzicht_m': 'mean'
    }).reset_index()
    fig_scat2 = px.scatter(df_div, x="doorzicht_m", y="soort", 
                           title="Doorzicht vs Soortenrijkdom")
    st.plotly_chart(fig_scat2, use_container_width=True)

# --- PCA ANALYSE ---
st.divider()
st.subheader("Multivariate clusteranalyse (PCA)")
st.caption("Clustering van meetpunten op basis van soortensamenstelling (bedekking).")

# --- EXPLAINER: HOE LEES IK DIT? ---
with st.expander("ℹ️ Uitleg: hoe interpreteer ik deze PCA plot?", expanded=False):
    st.markdown("""
    **Wat doet deze analyse?**
    Op elke locatie komen verschillende waterplanten voor in verschillende hoeveelheden. Omdat we tientallen soorten hebben, is het lastig om locaties direct met elkaar te vergelijken. 
    **PCA (Principal Component Analysis)** vat al deze soorten samen tot 2 hoofd-assen (PC1 en PC2) die de grootste verschillen in de data verklaren.
    
    **Hoe moet ik de grafiek lezen?**
    * **Elk punt is een meetlocatie.**
    * **Afstand zegt alles:** * Punten die **dicht bij elkaar** staan, hebben een **vergelijkbare vegetatie** (dezelfde soorten in dezelfde dichtheden).
        * Punten die **ver uit elkaar** staan, zijn qua plantengroei totaal verschillend.
    * **Clusters:** Als je groepjes punten ziet, zijn dit locaties die op elkaar lijken (bijv. 'diepe soortenarme locaties' vs 'ondiepe soortenrijke locaties').
        """)

# Data voorbereiden: Pivot table (Locaties x Soorten)
df_pca_source = df[df['jaar'] == year_pca]

# Check of er data is voor dit jaar
if df_pca_source.empty:
    st.warning("Geen data voor dit jaar.")
    st.stop()

pivot_df = df_pca_source.pivot_table(index='locatie_id', columns='soort', values='bedekking_pct', fill_value=0)

if len(pivot_df) > 5:
    # Standardiseren (belangrijk zodat soorten met enorme bedekking niet alles overheersen)
    x = StandardScaler().fit_transform(pivot_df)
    
    # PCA uitvoeren
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    
    # Resultaat in DataFrame
    pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    pca_df['locatie_id'] = pivot_df.index
    
    # Hoeveel variantie wordt verklaard? (Optioneel, goed voor interpretatie)
    explained_variance = pca.explained_variance_ratio_
    var_text = f"Verklaarde variantie: PC1 ({explained_variance[0]:.1%}) + PC2 ({explained_variance[1]:.1%})"
    
    # Koppelen aan meta-data (bijv. gemiddelde eco_score) voor kleur
    # We gebruiken 'mean' voor het geval er dubbele entries zouden zijn per locatie, hoewel locatie uniek zou moeten zijn in pivot
    meta = df_pca_source.groupby('locatie_id').mean(numeric_only=True).reset_index()
    pca_final = pd.merge(pca_df, meta, on='locatie_id')

    fig_pca = px.scatter(
        pca_final, 
        x='PC1', 
        y='PC2', 
        hover_data=['locatie_id'],
        color_continuous_scale='RdYlGn', 
        title=f"PCA vegetatiesamenstelling {year_pca}",
        labels={'PC1': f'PC1 ({explained_variance[0]:.1%})', 'PC2': f'PC2 ({explained_variance[1]:.1%})'}
    )
    
    # Zorg dat de assen 0 bevatten voor oriëntatie
    fig_pca.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
    fig_pca.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.5)
    
    st.plotly_chart(fig_pca, use_container_width=True)
    st.caption(f"NB.: {var_text}. Dit betekent dat deze 2D-weergave ongeveer {sum(explained_variance):.0%} van de totale verschillen in vegetatie samenvat.")
else:
    st.warning("Te weinig meetpunten (>5 nodig) om een betrouwbare clusteranalyse uit te voeren voor dit jaar.")