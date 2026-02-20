import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import load_data, RWS_GROEIVORM_CODES

st.title("🔗 Relaties en multivariate analyse")

df = load_data()

if df.empty:
    st.error("Geen data beschikbaar.")
    st.stop()

# -----------------------------
# Jaar selectie
# -----------------------------
years = sorted(df["jaar"].dropna().unique(), reverse=True)
year_pca = st.selectbox("Kies jaar analyse", years, index=0)

# Filter één keer, hergebruik overal
df_year = df[df["jaar"] == year_pca].copy()
if df_year.empty:
    st.warning("Geen data voor dit jaar.")
    st.stop()

# Maak numeric (1x)
for c in ["doorzicht_m", "diepte_m", "bedekking_pct", "totaal_bedekking_locatie"]:
    if c in df_year.columns:
        df_year[c] = pd.to_numeric(df_year[c], errors="coerce")

# Ratio: doorzicht / diepte
df_year["zicht_per_diepte"] = np.where(
    (df_year["diepte_m"].notna()) & (df_year["diepte_m"] > 0) & (df_year["doorzicht_m"].notna()),
    df_year["doorzicht_m"] / df_year["diepte_m"],
    np.nan,
)

# -----------------------------
# SCATTERPLOTS
# -----------------------------
c1, c2 = st.columns(2)

with c1:
    fig_scat1 = px.scatter(
        df_year,
        x="zicht_per_diepte",
        y="bedekking_pct",
        color="groeivorm",
        title="Doorzicht/Diepte vs Bedekking",
        labels={"zicht_per_diepte": "Doorzicht / Diepte (-)", "bedekking_pct": "Bedekking (%)"},
    )
    st.plotly_chart(fig_scat1, use_container_width=True)

with c2:
    # Optioneel: soortenrijkdom zuiverder maken (default UIT zodat functionaliteit niet verandert)
    st.caption("Soortenrijkdom kan optioneel gefilterd worden (type=Soort, excl. RWS codes).")
    strict_richness = st.checkbox("Soortenrijkdom: alleen echte soorten (excl. RWS codes)", value=False)

    if strict_richness and ("type" in df_year.columns):
        df_rich = df_year[(df_year["type"] == "Soort") & (~df_year["soort"].isin(RWS_GROEIVORM_CODES))].copy()
    else:
        df_rich = df_year

    df_div = (
        df_rich.groupby("locatie_id", as_index=False)
        .agg(
            soort=("soort", "nunique"),
            doorzicht_m=("doorzicht_m", "mean"),
            diepte_m=("diepte_m", "mean"),
        )
    )

    df_div["zicht_per_diepte"] = np.where(
        (df_div["diepte_m"].notna()) & (df_div["diepte_m"] > 0) & (df_div["doorzicht_m"].notna()),
        df_div["doorzicht_m"] / df_div["diepte_m"],
        np.nan,
    )

    fig_scat2 = px.scatter(
        df_div,
        x="zicht_per_diepte",
        y="soort",
        title="Doorzicht/Diepte vs Soortenrijkdom",
        labels={"zicht_per_diepte": "Doorzicht / Diepte (-)", "soort": "Soortenrijkdom (#)"},
    )
    st.plotly_chart(fig_scat2, use_container_width=True)

# -----------------------------
# PCA ANALYSE
# -----------------------------
st.divider()
st.subheader("Multivariate clusteranalyse (PCA)")
st.caption("Clustering van meetpunten op basis van soortensamenstelling (bedekking).")

with st.expander("ℹ️ Uitleg: hoe interpreteer ik deze PCA plot?", expanded=False):
    st.markdown(
        """
**Wat doet deze analyse?**
PCA vat de soorten-samenstelling samen tot 2 assen (PC1 en PC2) die het grootste deel van de variatie verklaren.

**Hoe lees ik de plot?**
- Elk punt = meetlocatie
- Dicht bij elkaar = vergelijkbare vegetatie
- Ver uit elkaar = sterk verschillend
"""
    )

# Optionele performanceknop: beperk aantal soorten in PCA (default = alle soorten)
use_top_n = st.checkbox("PCA: beperk tot Top-N soorten op totale bedekking (sneller bij veel soorten)", value=False)
top_n = st.slider("Top-N", min_value=10, max_value=200, value=50, step=10, disabled=not use_top_n)

# Bron voor PCA is df_year (niet opnieuw filteren)
df_pca_source = df_year.copy()

# Pivot table locaties x soorten (bedekking)
pivot_source = df_pca_source[["locatie_id", "soort", "bedekking_pct"]].copy()
pivot_source["bedekking_pct"] = pd.to_numeric(pivot_source["bedekking_pct"], errors="coerce").fillna(0.0)

if use_top_n:
    # Selecteer top N soorten op totale bedekking (vectorized)
    top_species = (
        pivot_source.groupby("soort", as_index=False)["bedekking_pct"].sum()
        .sort_values("bedekking_pct", ascending=False)
        .head(int(top_n))["soort"]
        .tolist()
    )
    pivot_source = pivot_source[pivot_source["soort"].isin(top_species)]

pivot_df = pivot_source.pivot_table(
    index="locatie_id",
    columns="soort",
    values="bedekking_pct",
    fill_value=0.0,
)

if len(pivot_df) <= 5:
    st.warning("Te weinig meetpunten (>5 nodig) om een betrouwbare clusteranalyse uit te voeren voor dit jaar.")
    st.stop()

# Standardiseren
x = StandardScaler().fit_transform(pivot_df)

# PCA
pca = PCA(n_components=2)
pc = pca.fit_transform(x)

pca_df = pd.DataFrame(pc, columns=["PC1", "PC2"])
pca_df["locatie_id"] = pivot_df.index

explained_variance = pca.explained_variance_ratio_
var_text = f"Verklaarde variantie: PC1 ({explained_variance[0]:.1%}) + PC2 ({explained_variance[1]:.1%})"

# Meta-data per locatie (numeric only)
meta = df_pca_source.groupby("locatie_id", as_index=False).mean(numeric_only=True)
pca_final = pca_df.merge(meta, on="locatie_id", how="left")

# Kies kleurvariabele (fix: eerder had je geen color=, waardoor color_continuous_scale geen effect had)
color_options = [c for c in ["totaal_bedekking_locatie", "doorzicht_m", "diepte_m"] if c in pca_final.columns]
color_var = st.selectbox("Kleur PCA-punten op (optioneel)", ["(geen)"] + color_options)

if color_var == "(geen)":
    fig_pca = px.scatter(
        pca_final,
        x="PC1",
        y="PC2",
        hover_data=["locatie_id"],
        title=f"PCA vegetatiesamenstelling {year_pca}",
        labels={"PC1": f"PC1 ({explained_variance[0]:.1%})", "PC2": f"PC2 ({explained_variance[1]:.1%})"},
    )
else:
    fig_pca = px.scatter(
        pca_final,
        x="PC1",
        y="PC2",
        hover_data=["locatie_id"],
        color=color_var,
        color_continuous_scale="RdYlGn",
        title=f"PCA vegetatiesamenstelling {year_pca}",
        labels={"PC1": f"PC1 ({explained_variance[0]:.1%})", "PC2": f"PC2 ({explained_variance[1]:.1%})"},
    )

fig_pca.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
fig_pca.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.5)
st.plotly_chart(fig_pca, use_container_width=True)

st.caption(
    f"NB.: {var_text}. Dit betekent dat deze 2D-weergave ongeveer {sum(explained_variance):.0%} van de totale verschillen in vegetatie samenvat."
)