# 2_Ruimtelijke_analyse.py
import streamlit as st
import pandas as pd
import folium
import numpy as np
from streamlit_folium import st_folium
from utils import load_data, add_species_group_columns, create_pie_map, create_map, get_sorted_species_list, EXCLUDED_SPECIES_CODES, RWS_GROEIVORM_CODES

st.set_page_config(layout="wide", page_title="Ruimtelijke analyse")

st.title("🗺️ Ruimtelijke analyse")
st.markdown("Vergelijk de vegetatieontwikkeling met diepte en doorzicht.")

# --- DATA INLADEN ---
df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filters")

if df.empty:
    st.error("Geen data geladen.")
    st.stop()

# 1. Jaar Filter
all_years = sorted(df['jaar'].dropna().unique(), reverse=True)
selected_year = st.sidebar.selectbox("Selecteer jaar", all_years)

# 2. Project Filter
all_projects = sorted(df['Project'].dropna().unique())
selected_projects = st.sidebar.multiselect(
    "Selecteer project(en)", 
    options=all_projects, 
    default=all_projects
)

# Filter basis dataset (Ruwe data met RWS codes)
df_filtered = df[
    (df['jaar'] == selected_year) & 
    (df['Project'].isin(selected_projects))
].copy()

if df_filtered.empty:
    st.warning("Geen data gevonden voor deze selectie.")
    st.stop()

# --- DATA VERRIJKING VOOR SOORTGROEPEN ---
# We maken een extra dataset aan waarin individuele soorten zijn ingedeeld in groepen
df_species_groups = add_species_group_columns(df_filtered)

st.sidebar.markdown("---")
st.sidebar.header("Kaartinstellingen")

# 3. KEUZE ANALYSE NIVEAU (NIEUWE FUNCTIONALITEIT)
# Hiermee splitsen we de dropdown om de UI schoon te houden.
analysis_level = st.sidebar.radio(
    "Kies analyseniveau",
    options=["groepen & aggregaties", "individuele soorten"]
)

selected_coverage_type = None

# A. Logica voor Groepen
if analysis_level == "groepen & aggregaties":

    # Optie A1: Algemene aggregaties
    opt_general = ["totale bedekking", "Groeivormen (pie)", "Trofieniveau", "KRW score", "soortgroepen"]

    # Optie A2: Taxonomische soortgroepen (uit verrijkte dataset)
    species_groups_list = sorted(df_species_groups['soortgroep'].dropna().unique())

    # NB: losse groeivormen NIET meer als selectiekeuze aanbieden
    all_options = opt_general

    selected_coverage_type = st.sidebar.selectbox("selecteer groep", options=all_options)

# B. Logica voor Individuele Soorten
else:
    # We halen de schone lijst met soorten op via de helper in utils.py
    # Dit toont alleen soorten die ook daadwerkelijk voorkomen in de gefilterde dataset (df_filtered)
    species_list = get_sorted_species_list(df_filtered)
    
    if not species_list:
        st.sidebar.warning("Geen individuele soorten gevonden in deze selectie.")
        st.stop()
        
    selected_coverage_type = st.sidebar.selectbox(
        "selecteer soort",
        options=species_list
    )

# 4. Kaartlaag Keuze
layer_mode = st.sidebar.radio(
    "kies kaartlaag",
    options=["Vegetatie", "Diepte", "Doorzicht"]
)

# --- DATA VOORBEREIDING VOOR KAART ---

# Stap 1: Maak een 'Base' dataframe met unieke locaties en abiotiek
# We gebruiken df_filtered omdat die alle bezochte locaties bevat (ook waar soorten 0% zijn)
df_locs = df_filtered.groupby(['locatie_id', 'Waterlichaam']).agg({
    'lat': 'first',
    'lon': 'first',
    'diepte_m': 'mean',
    'doorzicht_m': 'mean'
}).reset_index()

# Stap 2: Bepaal de vegetatiewaarde op basis van de selectie
df_veg_calc = pd.DataFrame()

# LOGICA SELECTIE:
if analysis_level == "groepen & aggregaties":
    
    if selected_coverage_type == "totale bedekking":
        # Gebruik WATPTN kolom
        df_veg_calc = df_filtered.groupby('locatie_id')['totaal_bedekking_locatie'].mean().reset_index()
        df_veg_calc.rename(columns={'totaal_bedekking_locatie': 'waarde_veg'}, inplace=True)

    elif selected_coverage_type == "KRW score":
        df_s = df_filtered[(df_filtered["type"] == "Soort") & (~df_filtered["soort"].isin(EXCLUDED_SPECIES_CODES))].copy()

        # Alleen rijen met score
        df_s = df_s.dropna(subset=["krw_score"])
        if df_s.empty:
            df_veg_calc = pd.DataFrame({"locatie_id": df_locs["locatie_id"].unique(), "waarde_veg": 0.0})
        else:
            # Gewicht = bedekking (als numeriek), fallback 1
            w = pd.to_numeric(df_s["bedekking_pct"], errors="coerce").fillna(1.0).clip(lower=0)
            df_s["w"] = w

            # Weighted mean per locatie: sum(score*w)/sum(w)
            agg = df_s.groupby("locatie_id").apply(
                lambda g: (g["krw_score"] * g["w"]).sum() / g["w"].sum() if g["w"].sum() > 0 else g["krw_score"].mean()
            )
            df_veg_calc = agg.reset_index().rename(columns={0: "waarde_veg"})

    elif selected_coverage_type == "Trofieniveau":
        df_s = df_filtered[(df_filtered["type"] == "Soort") & (~df_filtered["soort"].isin(RWS_GROEIVORM_CODES))].copy()
        df_s = df_s.dropna(subset=["trofisch_niveau"])
        if df_s.empty:
            df_veg_calc = pd.DataFrame({"locatie_id": df_locs["locatie_id"].unique(), "trofie_cat": "Onbekend"})
        else:
            df_s["bedekking_num"] = pd.to_numeric(df_s["bedekking_pct"], errors="coerce").fillna(1.0).clip(lower=0)

            # Som bedekking per locatie + trofie
            tmp = df_s.groupby(["locatie_id", "trofisch_niveau"])["bedekking_num"].sum().reset_index()

            # Kies dominante trofie per locatie
            idx = tmp.groupby("locatie_id")["bedekking_num"].idxmax()
            dom = tmp.loc[idx, ["locatie_id", "trofisch_niveau"]].rename(columns={"trofisch_niveau": "trofie_cat"})
            df_veg_calc = dom

    elif selected_coverage_type in species_groups_list:
        # Filter de VERRIJKTE dataset op taxonomische groep (sommeren want groep bestaat uit meerdere soorten)
        df_subset = df_species_groups[df_species_groups['soortgroep'] == selected_coverage_type]
        df_veg_calc = df_subset.groupby('locatie_id')['bedekkingsgraad_proc'].sum().reset_index()
        df_veg_calc.rename(columns={'bedekkingsgraad_proc': 'waarde_veg'}, inplace=True)

else: 
    # LOGICA VOOR INDIVIDUELE SOORTEN
    # We filteren de ruwe dataset op de specifieke soortnaam
    df_subset = df_filtered[df_filtered['soort'] == selected_coverage_type]
    
    # We nemen het gemiddelde (meestal is er maar 1 meting per locatie per jaar)
    df_veg_calc = df_subset.groupby('locatie_id')['bedekking_pct'].mean().reset_index()
    df_veg_calc.rename(columns={'bedekking_pct': 'waarde_veg'}, inplace=True)

# Stap 3: Merge alles samen
# Left join op locaties: zodat we OOK punten zien waar wel diepte is gemeten, maar de soort NIET voorkomt (waarde 0)
PIE_TYPES = ["KRW score", "Trofieniveau", "Groeivormen (pie)", "soortgroepen"]

# Voor pie-lagen: geen merge nodig; we gebruiken df_locs als basis
if analysis_level == "groepen & aggregaties" and layer_mode == "Vegetatie" and selected_coverage_type in PIE_TYPES:
    df_map_data = df_locs.copy()

    # Houd compatibiliteit met bestaande code (tabel/sortering verwacht vaak waarde_veg)
    if "waarde_veg" not in df_map_data.columns:
        df_map_data["waarde_veg"] = 0.0

else:
    # Normale (numerieke) kaartwaarden -> bestaande merge-logica
    # Zorg dat df_veg_calc altijd een locatie_id heeft; anders maak een lege placeholder
    if (df_veg_calc is None) or df_veg_calc.empty or ("locatie_id" not in df_veg_calc.columns):
        df_veg_calc = df_locs[["locatie_id"]].copy()
        df_veg_calc["waarde_veg"] = np.nan

    df_map_data = pd.merge(df_locs, df_veg_calc, on="locatie_id", how="left")
    df_map_data["waarde_veg"] = df_map_data["waarde_veg"].fillna(0)

# --- WEERGAVE ---

st.subheader(f"Kaartweergave: {layer_mode}")
if layer_mode == "Vegetatie":
    if analysis_level == "Individuele Soorten":
        st.info(f"Je bekijkt de verspreiding van de soort: **{selected_coverage_type}**")
    elif selected_coverage_type != "Totale Bedekking":
        st.info(f"Je bekijkt de verspreiding van de groep: **{selected_coverage_type}**")

# Legenda omschrijving
if layer_mode == "Vegetatie":
    st.caption("Legenda: Rood (0%) → Geel → Donkergroen (Hoge bedekking)")
elif layer_mode == "Diepte":
    st.caption("Legenda: Lichtblauw (Ondiep) → Donkerblauw (Diep)")
elif layer_mode == "Doorzicht":
    st.caption("Legenda: Bruin (Troebel) → Groen (Helder)")

# Map aanmaken
# Map aanmaken
if (
    layer_mode == "Vegetatie"
    and analysis_level == "groepen & aggregaties"
    and selected_coverage_type in ["KRW score", "Trofieniveau", "Groeivormen (pie)", "soortgroepen"]
):
    df_locs_for_map = df_locs.copy()

    # 1) Groeivormen (pie) -----------------------------------------------------
    if selected_coverage_type == "Groeivormen (pie)":
        # Groeivormen komen uit de RWS-groepregels (type='Groep')
        df_forms = df_filtered[df_filtered["type"] == "Groep"].copy()

        # Bedekking numeriek maken (voor pie op basis van % bedekking)
        df_forms["bedekking_num"] = pd.to_numeric(df_forms["bedekking_pct"], errors="coerce").fillna(0).clip(lower=0)

        # Som bedekking per locatie per groeivorm (pie-waarden)
        pivot = (
            df_forms.groupby(["locatie_id", "groeivorm"])["bedekking_num"]
            .sum()
            .unstack(fill_value=0)
        )

        # Optioneel: schaal terug als totalen per locatie > 100 (behoud relatieve verdeling)
        counts_by_loc = {}
        for loc, row in pivot.iterrows():
            d = row.to_dict()
            total = sum(d.values())
            if total > 100 and total > 0:
                factor = 100 / total
                d = {k: v * factor for k, v in d.items()}
            counts_by_loc[loc] = d

        # Kleuren groeivormen (jouw voorkeuren)
        color_map = {
            "Ondergedoken": "#2ca02c",   # groen
            "Emergent": "#ffd700",       # geel
            "Draadalgen": "#c2a5cf",     # lichtpaars
            "Drijvend": "#ff7f0e",       # oranje
            "FLAB": "#d62728",           # rood
            "Kroos": "#8c510a",          # bruin
        }
        order = ["Ondergedoken", "Drijvend", "Emergent", "Draadalgen", "Kroos", "FLAB"]

        map_obj = create_pie_map(
            df_locs_for_map,
            counts_by_loc=counts_by_loc,
            label="Groeivormen (% bedekking)",
            color_map=color_map,
            order=order,
            size_px=30,
            zoom_start=10,
            fixed_total=100,
            fill_gap=True,
            gap_color="transparent",
            # alleen als je dit in utils hebt toegevoegd:
            # empty_fill_color="transparent",
        )

    # 2) Trofieniveau (pie op records) ----------------------------------------
    elif selected_coverage_type == "Trofieniveau":
        df_species = df_filtered[
            (df_filtered["type"] == "Soort")
            & (~df_filtered["soort"].isin(EXCLUDED_SPECIES_CODES))
        ].copy()

        # Alleen records met trofisch niveau
        df_species = df_species.dropna(subset=["trofisch_niveau"])

        if df_species.empty:
            counts_by_loc = {}
        else:
            pivot = (
                df_species.groupby(["locatie_id", "trofisch_niveau"])
                .size()
                .unstack(fill_value=0)
            )
            counts_by_loc = {loc: row.to_dict() for loc, row in pivot.iterrows()}

        # Trofie kleuren (zoals eerder afgesproken)
        color_map = {
            "oligotroof": "#2ca02c",        # groen
            "mesotroof": "#1f77b4",         # blauw
            "eutroof": "#ff7f0e",           # oranje
            "sterk eutroof": "#d62728",     # rood
            "brak": "#ffd700",              # geel
            "marien": "#8c510a",            # bruin
            "kroos": "#7f7f7f",             # grijs (optioneel)
            "Onbekend": "#999999",
        }
        order = ["oligotroof", "mesotroof", "eutroof", "sterk eutroof", "brak", "marien", "kroos", "Onbekend"]

        map_obj = create_pie_map(
            df_locs_for_map,
            counts_by_loc=counts_by_loc,
            label="Trofieniveau (records)",
            color_map=color_map,
            order=order,
            size_px=30,
            zoom_start=10
        )

    # 3) KRW score (pie op records) -------------------------------------------
    elif selected_coverage_type == "KRW score":
        df_species = df_filtered[
            (df_filtered["type"] == "Soort")
            & (~df_filtered["soort"].isin(EXCLUDED_SPECIES_CODES))
        ].copy()

        # Gebruik bestaande krw_class als die er is; anders afleiden uit krw_score
        if "krw_class" in df_species.columns:
            df_species["krw_cat"] = df_species["krw_class"]
        else:
            df_species["krw_cat"] = pd.cut(
                df_species["krw_score"],
                bins=[0, 2, 4, 5],
                labels=["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)"],
                include_lowest=True
            )

        df_species = df_species.dropna(subset=["krw_cat"])

        if df_species.empty:
            counts_by_loc = {}
        else:
            pivot = (
                df_species.groupby(["locatie_id", "krw_cat"])
                .size()
                .unstack(fill_value=0)
            )
            counts_by_loc = {loc: row.to_dict() for loc, row in pivot.iterrows()}

        color_map = {
            "Gunstig (1-2)": "#2ca02c",   # groen
            "Neutraal (3-4)": "#ff7f0e",  # oranje
            "Ongewenst (5)": "#d62728"    # rood
        }
        order = ["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)"]

        map_obj = create_pie_map(
            df_locs_for_map,
            counts_by_loc=counts_by_loc,
            label="KRW-score (records)",
            color_map=color_map,
            order=order,
            size_px=30,
            zoom_start=10
        )

    # 4) Soortgroepen (pie op bedekking, gedeeltelijk gevuld t.o.v. 100) -------
    else:  # selected_coverage_type == "soortgroepen"
        # df_species_groups is al aangemaakt via add_species_group_columns(df_filtered) in je script
        df_sg = df_species_groups.copy()

        # Bedekking numeriek maken
        df_sg["bedekking_num"] = pd.to_numeric(df_sg["bedekkingsgraad_proc"], errors="coerce").fillna(0).clip(lower=0)

        # Som bedekking per locatie en soortgroep
        pivot = (
            df_sg.groupby(["locatie_id", "soortgroep"])["bedekking_num"]
            .sum()
            .unstack(fill_value=0)
        )

        # Schaal terug als totalen > 100 (behoud relatieve verdeling)
        counts_by_loc = {}
        for loc, row in pivot.iterrows():
            d = row.to_dict()
            total = sum(d.values())
            if total > 100 and total > 0:
                factor = 100 / total
                d = {k: v * factor for k, v in d.items()}
            counts_by_loc[loc] = d

        # Kleuren: voldoende onderscheid (kwalitatieve, contrasterende palette)
        # LET OP: jouw soortgroepnamen zijn lowercase (o.a. "chariden", "iseotiden", etc.)
        color_map = {
            "chariden": "#1b9e77",
            "iseotiden": "#7570b3",          # in utils staat deze spelling zo
            "parvopotamiden": "#d95f02",
            "magnopotamiden": "#66a61e",
            "myriophylliden": "#e7298a",
            "vallisneriiden": "#e6ab02",
            "elodeiden": "#a6761d",
            "stratiotiden": "#1f78b4",
            "pepliden": "#b2df8a",
            "batrachiiden": "#fb9a99",
            "nymphaeiden": "#cab2d6",
            "haptofyten": "#fdbf6f",
            "Kenmerkende soort (N2000)": "#000000",
            "Overig / Individueel": "#999999"
        }

        # Volgorde (optioneel): alleen tonen wat in pivot voorkomt kan ook, maar dit houdt het consistent
        order = [
            "chariden", "iseotiden", "parvopotamiden", "magnopotamiden", "myriophylliden",
            "vallisneriiden", "elodeiden", "stratiotiden", "pepliden", "batrachiiden",
            "nymphaeiden", "haptofyten", "Kenmerkende soort (N2000)", "Overig / Individueel"
        ]

        map_obj = create_pie_map(
            df_locs_for_map,
            counts_by_loc=counts_by_loc,
            label="Soortgroepen (% bedekking)",
            color_map=color_map,
            order=order,
            size_px=30,
            zoom_start=10,
            fixed_total=100,
            fill_gap=True,
            gap_color="transparent",
            # belangrijk als je 0% volledig transparant wilt:
            #empty_fill_color="transparent"
        )

else:
    # Bestaande functionaliteit behouden
    map_obj = create_map(df_map_data, layer_mode, label_veg=selected_coverage_type)

st_folium(map_obj, height=600, width=None)

# --- Extra kolommen per locatie: KRW-score en Trofieniveau ---
df_species = df_filtered[
    (df_filtered["type"] == "Soort") &
    (~df_filtered["soort"].isin(RWS_GROEIVORM_CODES))
].copy()

# 1) KRW-score per locatie (gewogen gemiddelde op basis van bedekking)
df_krw = df_species.dropna(subset=["krw_score"]).copy()
if not df_krw.empty:
    df_krw["bedekking_num"] = pd.to_numeric(df_krw["bedekking_pct"], errors="coerce").fillna(1.0).clip(lower=0)

    krw_loc = (
        df_krw.groupby("locatie_id")
        .apply(lambda g: (g["krw_score"] * g["bedekking_num"]).sum() / g["bedekking_num"].sum()
               if g["bedekking_num"].sum() > 0 else g["krw_score"].mean())
        .reset_index(name="krw_score_loc")
    )
else:
    krw_loc = pd.DataFrame(columns=["locatie_id", "krw_score_loc"])

# 2) Trofieniveau per locatie (dominant: hoogste som bedekking)
df_trof = df_species.dropna(subset=["trofisch_niveau"]).copy()
if not df_trof.empty:
    df_trof["bedekking_num"] = pd.to_numeric(df_trof["bedekking_pct"], errors="coerce").fillna(1.0).clip(lower=0)

    tmp = (
        df_trof.groupby(["locatie_id", "trofisch_niveau"])["bedekking_num"]
        .sum()
        .reset_index()
    )

    idx = tmp.groupby("locatie_id")["bedekking_num"].idxmax()
    trof_loc = tmp.loc[idx, ["locatie_id", "trofisch_niveau"]].rename(
        columns={"trofisch_niveau": "trofieniveau_loc"}
    )
else:
    trof_loc = pd.DataFrame(columns=["locatie_id", "trofieniveau_loc"])

# Merge de locatie-samenvattingen in df_map_data
df_map_data = (
    df_map_data
    .merge(krw_loc, on="locatie_id", how="left")
    .merge(trof_loc, on="locatie_id", how="left")
)

# --- TABEL ---
st.divider()
with st.expander(f"Toon data voor {selected_coverage_type}"):

    df_display = df_map_data.copy()

    # Veilige sortering
    if "waarde_veg" in df_display.columns:
        df_display = df_display.sort_values(by="waarde_veg", ascending=False)
    elif "krw_score_loc" in df_display.columns:
        df_display = df_display.sort_values(by="krw_score_loc", ascending=True)

    # Kolommen voor weergave
    cols = ["locatie_id", "Waterlichaam", "waarde_veg", "krw_score_loc", "trofieniveau_loc", "diepte_m", "doorzicht_m"]
    cols = [c for c in cols if c in df_display.columns]

    # Dynamische formattering van waarde_veg
    if selected_coverage_type == "KRW score":
        waarde_cfg = st.column_config.NumberColumn("waarde_veg (KRW)", format="%.2f")
    elif selected_coverage_type == "Trofieniveau":
        waarde_cfg = st.column_config.TextColumn("waarde_veg")
    else:
        waarde_cfg = st.column_config.NumberColumn(f"{selected_coverage_type} (%)", format="%.1f%%")

    st.dataframe(
        df_display[cols],
        use_container_width=True,
        column_config={
            "waarde_veg": waarde_cfg,
            "krw_score_loc": st.column_config.NumberColumn("KRW score (locatie)", format="%.2f"),
            "trofieniveau_loc": st.column_config.TextColumn("Trofieniveau (dominant)"),
            "diepte_m": st.column_config.NumberColumn("Diepte (m)", format="%.2f"),
            "doorzicht_m": st.column_config.NumberColumn("Doorzicht (m)", format="%.2f"),
        }
    )