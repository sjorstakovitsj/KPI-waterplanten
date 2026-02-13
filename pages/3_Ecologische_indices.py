import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import load_data, add_species_group_columns

st.set_page_config(layout="wide")
st.title("🌿 Ecologische indices")

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Selectie filters")

# 1. Project Filter
all_projects = sorted(df['Project'].dropna().unique())
selected_projects = st.sidebar.multiselect(
    "Selecteer project(en)", 
    options=all_projects, 
    default=all_projects
)

# Filter data op basis van project voor de volgende stap
df_project = df[df['Project'].isin(selected_projects)]

# 2. Waterlichaam Filter
all_bodies = sorted(df_project['Waterlichaam'].dropna().unique())
selected_bodies = st.sidebar.multiselect(
    "Selecteer waterlichaam",
    options=all_bodies,
    default=all_bodies
)

# Filter data op basis van waterlichaam
df_filtered_base = df_project[df_project['Waterlichaam'].isin(selected_bodies)]

# Uitsluiten van groepen (zoals EMSPTN) en toepassen van alle filters
df_species_only = df_filtered_base[
    (df_filtered_base['type'] == 'Soort')
].copy()

# --- BUBBLE PLOT ---
st.subheader("Relatie doorzicht vs bedekking")
with st.expander("ℹ️ Hoe komt deze bubble plot tot stand? (toelichting)"):
    st.markdown("""
### Wat stelt één bubble voor?
Elke **bubble staat voor één individuele plantensoort** (de wetenschappelijke naam in `soort`).  

---

### Welke data gaat er de plot in?
Na jouw selectie op **project** en **waterlichaam** worden alleen records met `type == 'Soort'` meegenomen.  
Daarmee sluit je groeivorm-/groepsregels (zoals RWS-verzamelgroepen) uit voor deze analyse.

---

### Stap 1 — Aggregatie per soort × jaar (uitmiddeling binnen het jaar)
De code maakt eerst een dataset `df_bubble` waarin per combinatie **(soort, jaar)** gemiddelden worden berekend:

- `doorzicht_m`: gemiddelde doorzicht (m)  
- `bedekking_pct`: gemiddelde bedekking (%)  
- `diepte_m`: gemiddelde diepte (m)

Dit gebeurt met:

- `groupby(['soort','jaar']).agg({ ... 'mean' ... })

**Gevolg:** als een soort in hetzelfde jaar op meerdere locaties of metingen voorkomt, wordt dat samengevat tot één punt per soort per jaar.)

---

### Stap 2 — Selectie van een jaarperiode + aggregatie per soort (uitmiddeling over jaren)
Vervolgens kies je met de slider een periode (bijv. 2018–2024).  
De dataset wordt op deze jaar-range gefilterd en daarna opnieuw geaggregeerd per **soort**:

- `groupby(['soort']).agg({ ... 'mean' ... })

**Belangrijk:** dit is een *gemiddelde van jaarlijkse gemiddelden* (“mean-of-means”).  
Dat betekent dat jaren met weinig metingen voor een soort ongeveer even zwaar kunnen meetellen als jaren met veel metingen (tenzij je dit later weegt).

---

### Wat betekenen de assen en bubble-grootte?
In de scatterplot wordt vervolgens geplot:

- **y-as:** `bedekking_pct` (gemiddelde bedekking van de soort over de gekozen periode)  
- **bubble size:** `diepte_m` (gemiddelde diepte; dieper water → grotere bubble)
- **x-as:**  
  - oorspronkelijk: `doorzicht_m` (gemiddeld doorzicht)
  - (indien aangepast): `doorzicht_diepte_ratio = doorzicht_m / diepte_m` (ratio doorzicht t.o.v. diepte)  

De soortnaam verschijnt in de tooltip doordat `hover_name="soort"` is ingesteld.

---

### Waarom worden sommige waarden “gefixt”?
De code past een kleine fix toe op `diepte_m` voor de bubble-grootte, zodat Plotly geen problemen krijgt met lege of niet-positieve waarden:
- lege/0/negatieve dieptes worden vervangen door een kleine minimale waarde (bijv. 0.1).

---

### Interpretatie in 10 seconden
- **Verder naar rechts** (bij ratio): relatief meer doorzicht t.o.v. diepte (gunstiger lichtklimaat).  
- **Hoger**: hogere gemiddelde bedekking voor die soort.  
- **Grotere bubble**: soort komt gemiddeld in dieper water voor.
""")

# 1. Aggregeren
df_bubble = df_species_only.groupby(['soort','jaar']).agg({
    'doorzicht_m': 'mean',
    'bedekking_pct': 'mean',
    'diepte_m': 'mean'
}).reset_index()

# 2. JAAR SELECTIE (RANGE SLIDER)
if not df['jaar'].empty:
    min_year = int(df['jaar'].min())
    max_year = int(df['jaar'].max())
    sel_years = st.slider("Selecteer periode", min_year, max_year, [min_year, max_year], key="ecol_bubble_period")

    # Filter op jaar-range
    df_bubble_range = df_bubble[
        (df_bubble['jaar'] >= sel_years[0]) & (df_bubble['jaar'] <= sel_years[1])
    ].copy()

    # Gemiddelde over de periode
    df_bubble_plot = df_bubble_range.groupby(['soort']).agg({
        'doorzicht_m': 'mean',
        'bedekking_pct': 'mean',
        'diepte_m': 'mean'
    }).reset_index()

    df_bubble_plot = df_bubble_range.groupby(['soort']).agg({
    'doorzicht_m': 'mean',
    'bedekking_pct': 'mean',
    'diepte_m': 'mean'
}).reset_index()

    # --- NIEUW: ratio doorzicht / diepte voor x-as ---
    # Zorg dat diepte niet 0 of negatief is om deling door 0 te voorkomen
    df_bubble_plot["diepte_safe"] = df_bubble_plot["diepte_m"].fillna(0.0)
    df_bubble_plot.loc[df_bubble_plot["diepte_safe"] <= 0, "diepte_safe"] = np.nan

    df_bubble_plot["doorzicht_diepte_ratio"] = df_bubble_plot["doorzicht_m"] / df_bubble_plot["diepte_safe"]

    # Optioneel: verwijder rijen waar ratio niet berekend kan worden
    df_bubble_plot = df_bubble_plot.dropna(subset=["doorzicht_diepte_ratio"])

    # Plotly fix voor NaNs in size (bubble size blijft diepte_m)
    df_bubble_plot['diepte_m'] = df_bubble_plot['diepte_m'].fillna(0.1)
    df_bubble_plot.loc[df_bubble_plot['diepte_m'] <= 0, 'diepte_m'] = 0.1


    # Plotly fix voor NaNs in size
    df_bubble_plot['diepte_m'] = df_bubble_plot['diepte_m'].fillna(0.1)
    df_bubble_plot.loc[df_bubble_plot['diepte_m'] <= 0, 'diepte_m'] = 0.1

# 2. JAAR SELECTIE (RANGE SLIDER)
if not df_bubble.empty:
    min_year = int(df_bubble['jaar'].min())
    max_year = int(df_bubble['jaar'].max())

    # Filter op jaar-range (LET OP: echte operatoren, geen &gt; &amp;)
    df_bubble_range = df_bubble[
        (df_bubble['jaar'] >= sel_years[0]) & (df_bubble['jaar'] <= sel_years[1])
    ].copy()

    # Gemiddelde over de periode per soort
    df_bubble_plot = df_bubble_range.groupby(['soort']).agg({
        'doorzicht_m': 'mean',
        'bedekking_pct': 'mean',
        'diepte_m': 'mean'
    }).reset_index()

    # --- Ratio doorzicht / diepte voor x-as ---
    df_bubble_plot["diepte_safe"] = df_bubble_plot["diepte_m"].fillna(0.0)
    df_bubble_plot.loc[df_bubble_plot["diepte_safe"] <= 0, "diepte_safe"] = np.nan
    df_bubble_plot["doorzicht_diepte_ratio"] = df_bubble_plot["doorzicht_m"] / df_bubble_plot["diepte_safe"]
    df_bubble_plot = df_bubble_plot.dropna(subset=["doorzicht_diepte_ratio"])

    # Bubble size fix (diepte_m)
    df_bubble_plot["diepte_m"] = df_bubble_plot["diepte_m"].fillna(0.1)
    df_bubble_plot.loc[df_bubble_plot["diepte_m"] <= 0, "diepte_m"] = 0.1

    if not df_bubble_plot.empty:
        fig_bubble = px.scatter(
            df_bubble_plot,
            x="doorzicht_diepte_ratio",
            y="bedekking_pct",
            size="diepte_m",
            hover_name="soort",
            size_max=40,
            title=f"Ecologische indices ({sel_years[0]} - {sel_years[1]})",
            labels={
                "doorzicht_diepte_ratio": "gem. doorzicht / gem. diepte (-)",
                "bedekking_pct": "gem. bedekking (%)",
                "diepte_m": "gem. diepte (m)"
            }
        )

        # --- Zones voor ratio doorzicht/diepte ---
        # Zone 1: 0.6–0.8 lichtgroen (transparant)
        fig_bubble.add_vrect(
            x0=0.6, x1=0.8,
            fillcolor="rgba(46, 204, 113, 0.16)",  # lichtgroen met alpha 0.16
            line_width=0,
            layer="below",
            annotation_text="OK (0.6–0.8)",
            annotation_position="top left"
        )

        # Zone 2: 0.8–1.0 donkerder groen (transparant maar duidelijker)
        fig_bubble.add_vrect(
            x0=0.8, x1=1.0,
            fillcolor="rgba(39, 174, 96, 0.24)",   # donkerder groen met alpha 0.24
            line_width=0,
            layer="below",
            annotation_text="Ideaal (≥0.8)",
            annotation_position="top left"
        )

        # (Optioneel) stippellijnen als extra houvast
        fig_bubble.add_vline(
            x=0.6,
            line_width=2,
            line_dash="dot",
            line_color="rgba(255, 165, 0, 0.85)",  # oranje
            annotation_text="Min 0.6",
            annotation_position="top left"
        )

        fig_bubble.add_vline(
            x=0.8,
            line_width=2,
            line_dash="dot",
            line_color="rgba(0, 100, 0, 0.90)",    # donker groen
            annotation_text="Streef 0.8",
            annotation_position="top left"
        )

        st.plotly_chart(fig_bubble, use_container_width=True)
    else:
        st.warning("Geen data gevonden voor deze filtercombinatie.")
else:
    st.warning("Geen data beschikbaar voor bubbleplot na filtering.")

# --- CATEGORIE HEATMAP (Trofie / Groeivorm / Soortgroep / KRW) ---
st.subheader("📊 Verdeling per jaar (heatmap)")

with st.expander("ℹ️ Uitleg: hoe wordt deze heatmap berekend?", expanded=False):
    st.markdown("""
### Wat laat deze heatmap zien?
De heatmap toont per **jaar** hoe de **samenstelling** (verdeling) eruitziet van een gekozen parameter:
- **Trofieniveau** (bij soorten)
- **Groeivormen** (bij groeivorm-/groepregels)
- **Soortgroepen** (afgeleid uit soortnamen via mapping)
- **KRW score-klasse** (bij soorten)

De rijen zijn categorieën (bijv. *Ondergedoken*, *chariden*, *Gunstig*), de kolommen zijn jaren, en elke cel is de **bijdrage** van die categorie in dat jaar.

---

## 1) Welke data gaat erin per keuze?
Afhankelijk van de parameter wordt eerst de brondata geselecteerd:

**A) Groeivormen**  
- Gebruikt regels met `type == "Groep"` (dus groeivorm-/verzamelgroepen).  
- Categoriekolom = `groeivorm`.  
- Bedekkingswaarde wordt numeriek gemaakt als `bedekking_num` op basis van `bedekking_pct`. 

**B) Soortgroepen**  
- Gebruikt regels met `type == "Soort"` en (als aanwezig) `Grootheid == "BEDKG"`.  
- Roept `add_species_group_columns()` aan om per soort een `soortgroep` toe te kennen én een numerieke bedekking `bedekkingsgraad_proc` te maken.  
- Categoriekolom = `soortgroep`, bedekkingswaarde = `bedekking_num` (afgeleid van `bedekkingsgraad_proc`).

**C) Trofieniveau**  
- Gebruikt `type == "Soort"` en (als aanwezig) `Grootheid == "BEDKG"`.  
- Categoriekolom = `trofisch_niveau`.  
- Bedekkingswaarde = `bedekking_num` op basis van `bedekking_pct`.

**D) KRW score**  
- Gebruikt `type == "Soort"` en (als aanwezig) `Grootheid == "BEDKG"`.  
- Categorie = `krw_class` als aanwezig, anders wordt `krw_cat` afgeleid uit `krw_score` met klassen (Gunstig/Neutraal/Ongewenst).  
- Bedekkingswaarde = `bedekking_num` op basis van `bedekking_pct`.

---

## 2) “Verdeling op basis van …” (de twee keuzes)

### Keuze 1 — **Records (aantal waarnemingen)**
Hier tel je per jaar hoeveel **regels/records** er vallen in elke categorie.

Technisch gebeurt dat met:
- `groupby([categorie, jaar]).size()` → dit geeft **aantallen** per categorie per jaar.

**Interpretatie:**  
Dit is een maat voor **hoe vaak** iets voorkomt in de dataset (frequentie van meetrecords), niet hoe “dominant” het ecologisch is.

> Let op: als in een jaar meer gemeten is (meer records), krijg je ook grotere aantallen — tenzij je normaliseert (zie punt 3).

---

### Keuze 2 — **Bedekking-gewogen (som bedekking)**
Hier tel je niet het aantal regels, maar je **somt de bedekking** op per categorie per jaar.

Technisch gebeurt dat met:
- `groupby([categorie, jaar])["bedekking_num"].sum()` → dit geeft de **totale bedekking** per categorie per jaar.

**Interpretatie:**  
Dit is een maat voor **dominantie/biomassa-indicatie** (in elk geval “hoeveel bedekking” er aan die categorie wordt toegeschreven), omdat categorieën met hoge bedekking zwaarder meetellen dan categorieën met lage bedekking.

> Dit is vaak ecologisch informatiever dan alleen aantallen records, zeker als je veel soorten met kleine bedekking hebt.

---

## 3) Wat betekent “normaliseren per jaar”?
Als **Normaliseer per jaar (100% verdeling)** aan staat, wordt elke jaarkolom omgerekend naar een **percentageverdeling** die optelt tot 100%.

Technisch gebeurt dat zo:
1. Per jaar wordt de kolomsom bepaald: `col_sums = heat_matrix.sum(axis=0)`  
2. Elke waarde wordt gedeeld door de jaartotaalsom: `heat_matrix.div(col_sums, axis=1)`  
3. Daarna × 100 zodat je percentages krijgt.

### Wat is het effect?
- **Zonder normalisatie**: de heatmap laat **absolute** aantallen of absolute som bedekking zien. Jaren met meer metingen of meer totale bedekking springen dan automatisch eruit.
- **Met normalisatie**: je kijkt naar **relatieve samenstelling** binnen elk jaar.  
  Een cel van 30% betekent: “30% van alle records (of 30% van de totale bedekking) in dát jaar hoort bij deze categorie.”

### Waarom is dit handig?
Normalisatie per jaar maakt jaren onderling vergelijkbaar als:
- er in sommige jaren veel meer of minder is gemeten (verschil in meetinspanning),
- of de totale bedekking per jaar sterk varieert.

Je ziet dan vooral **compositieverschuivingen** (bijv. “meer aandeel ondergedoken” of “meer aandeel ongunstig”), los van het totale volume.

---

## 4) Samenvatting in één zin
- **Records** = “hoe vaak komt een categorie voor in de data?”
- **Bedekking-gewogen** = “hoeveel bedekking draagt een categorie bij?”
- **Normaliseren per jaar** = “zet elk jaar om naar 0–100% zodat je vooral de *verdeling* ziet, niet de meetomvang.”
""")

# Keuzes: welke parameter tonen?
heatmap_param = st.selectbox(
    "Kies parameter voor heatmap",
    ["Trofieniveau", "Groeivormen", "Soortgroepen", "KRW score"]
)

# Keuzes: hoe berekenen we de verdeling?
heatmap_basis = st.radio(
    "Bereken verdeling op basis van",
    ["Records (aantal waarnemingen)", "Bedekking-gewogen (som bedekking)"],
    index=1, horizontal=True, key="heatmap_basis_choice"
)

# Normalisatie: verdeling per jaar naar 100%?
normalize_year = st.checkbox("Normaliseer per jaar (100% verdeling)", value=True)

# Gebruik dezelfde jaar-range als bubbleplot (of maak hem opnieuw als je wilt)
years = sorted(df_filtered_base["jaar"].dropna().unique())
if not years:
    st.info("Geen jaren beschikbaar voor heatmap.")
else:
    # --- bouw brondata afhankelijk van parameter ---
    if heatmap_param == "Groeivormen":
        # Groeivormen zitten als type='Groep' in de data
        df_h = df_filtered_base[df_filtered_base["type"] == "Groep"].copy()
        df_h = df_h.dropna(subset=["groeivorm", "jaar"])

        cat_col = "groeivorm"

        # Voor groeivormen is bedekking_pct de logische basis
        df_h["bedekking_num"] = pd.to_numeric(df_h["bedekking_pct"], errors="coerce").fillna(0).clip(lower=0)

    elif heatmap_param == "Soortgroepen":
        # Soortgroepen via mapping op individuele soorten
        df_species = df_filtered_base[df_filtered_base["type"] == "Soort"].copy()
        if "Grootheid" in df_species.columns:
            df_species = df_species[df_species["Grootheid"] == "BEDKG"].copy()

        df_h = add_species_group_columns(df_species)  # voegt 'soortgroep' + 'bedekkingsgraad_proc' toe
        df_h = df_h.dropna(subset=["soortgroep", "jaar"])

        cat_col = "soortgroep"

        # bedekkingsgraad_proc is numerieke bedekking uit utils-functie
        df_h["bedekking_num"] = pd.to_numeric(df_h["bedekkingsgraad_proc"], errors="coerce").fillna(0).clip(lower=0)

    elif heatmap_param == "Trofieniveau":
        df_h = df_filtered_base[df_filtered_base["type"] == "Soort"].copy()
        if "Grootheid" in df_h.columns:
            df_h = df_h[df_h["Grootheid"] == "BEDKG"].copy()

        df_h = df_h.dropna(subset=["trofisch_niveau", "jaar"])
        cat_col = "trofisch_niveau"

        df_h["bedekking_num"] = pd.to_numeric(df_h["bedekking_pct"], errors="coerce").fillna(0).clip(lower=0)

    else:  # "KRW score"
        df_h = df_filtered_base[df_filtered_base["type"] == "Soort"].copy()
        if "Grootheid" in df_h.columns:
            df_h = df_h[df_h["Grootheid"] == "BEDKG"].copy()

        # Gebruik krw_class als die er is; anders afleiden uit krw_score
        if "krw_class" in df_h.columns:
            df_h["krw_cat"] = df_h["krw_class"]
        else:
            df_h["krw_cat"] = pd.cut(
                df_h["krw_score"],
                bins=[0, 2, 4, 5],
                labels=["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)"],
                include_lowest=True
            )

        df_h = df_h.dropna(subset=["krw_cat", "jaar"])
        cat_col = "krw_cat"

        df_h["bedekking_num"] = pd.to_numeric(df_h["bedekking_pct"], errors="coerce").fillna(0).clip(lower=0)

    if df_h.empty:
        st.info("Geen data beschikbaar voor deze heatmap-keuze (na filters).")
    else:
        # --- aggregatie per jaar x categorie ---
        if heatmap_basis.startswith("Records"):
            df_agg = (
                df_h.groupby([cat_col, "jaar"])
                .size()
                .reset_index(name="waarde")
            )
        else:
            df_agg = (
                df_h.groupby([cat_col, "jaar"])["bedekking_num"]
                .sum()
                .reset_index(name="waarde")
            )

        # --- pivot naar matrix categorie x jaar ---
        heat_matrix = df_agg.pivot(index=cat_col, columns="jaar", values="waarde").fillna(0)

        # --- normaliseren per jaar (verdeling) ---
        if normalize_year:
            col_sums = heat_matrix.sum(axis=0).replace(0, np.nan)
            heat_matrix = heat_matrix.div(col_sums, axis=1).fillna(0) * 100  # percentage per jaar

        # --- optionele ordening voor bekende categorieën ---
        if heatmap_param == "KRW score":
            order = ["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)"]
            heat_matrix = heat_matrix.reindex([x for x in order if x in heat_matrix.index])

        if heatmap_param == "Trofieniveau":
            order = ["oligotroof", "mesotroof", "eutroof", "sterk eutroof", "brak", "marien", "kroos"]
            # behoud overige categorieën onderaan
            keep = [x for x in order if x in heat_matrix.index]
            rest = [x for x in heat_matrix.index if x not in keep]
            heat_matrix = heat_matrix.reindex(keep + rest)

        if heatmap_param == "Groeivormen":
            order = ["Ondergedoken", "Drijvend", "Emergent", "Draadalgen", "Kroos", "FLAB"]
            keep = [x for x in order if x in heat_matrix.index]
            rest = [x for x in heat_matrix.index if x not in keep]
            heat_matrix = heat_matrix.reindex(keep + rest)

        # --- plot ---
        title_suffix = " (%)" if normalize_year else ""
        fig = px.imshow(
            heat_matrix,
            color_continuous_scale="Viridis",
            aspect="auto",
            labels=dict(
                x="Jaar",
                y=heatmap_param,
                color=("Aandeel" + title_suffix) if normalize_year else "Waarde"
            ),
            title=f"Heatmap {heatmap_param} per jaar" + (" (genormaliseerd)" if normalize_year else "")
        )

        # -----------------------------
        # NIEUW: toon waarde in elke cel (wit, altijd zichtbaar)
        # -----------------------------
        if normalize_year:
            # Percentages: 1 decimaal + %
            text_matrix = heat_matrix.round(1).astype(str) + "%"
        else:
            if heatmap_basis.startswith("Records"):
                # Aantallen: als integer
                text_matrix = heat_matrix.round(0).astype(int).astype(str)
            else:
                # Bedekking-som: 1 decimaal (pas aan als je liever 0 of 2 decimals wilt)
                text_matrix = heat_matrix.round(1).astype(str)

        fig.update_traces(
            text=text_matrix.values,
            texttemplate="%{text}",
            textfont=dict(color="white", size=12),
            # center is default voor heatmap tekst; expliciet kan ook:
            # textposition="middle center"
        )

        # Optioneel: dwing tonen af (handig als Plotly soms wil verbergen)
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode="show")

        # Maak labels leesbaar
        fig.update_layout(height=650, yaxis=dict(side="left"))
        st.plotly_chart(fig, use_container_width=True)