import streamlit as st
import pandas as pd
from faker import Faker
import random
from pycaret.clustering import setup, create_model, assign_model
import numpy as np

# ---------------------------------------------
# KONFIGURACJA STRONY
# ---------------------------------------------
st.set_page_config(
    page_title="Find Friends – Analiza",
    layout="wide"
)

fake = Faker("pl_PL")


# ---------------------------------------------
# FUNKCJA: GENEROWANIE DANYCH (100 PROFILI)
# ---------------------------------------------
def generate_data():
    data = []
    for i in range(1, 101):
        record = {
            "client_id": i,
            "miasto": fake.city(),
            "pytanie_1": random.randint(1, 5),
            "pytanie_2": random.randint(1, 5),
            "pytanie_3": random.randint(1, 5),
            "pytanie_4": random.randint(1, 5),
            "pytanie_5": random.randint(1, 5)
        }
        data.append(record)

    return pd.DataFrame(data)


# ---------------------------------------------
# DYNAMICZNE NAZWY KLASTRÓW
# ---------------------------------------------
cluster_names_list = [
    "Innowatorzy", "Analitycy", "Eksploratorzy", "Mediatorzy",
    "Stabilizatorzy", "Twórcy Adaptacyjni", "Systemowcy", "Wizjonerzy"
]


def generate_cluster_descriptions(n_clusters):
    descriptions = {}
    for i in range(n_clusters):
        name = cluster_names_list[i]
        descriptions[name] = f"{name}: grupa wyróżniająca się charakterystycznym stylem odpowiedzi i unikalnym profilem psychologicznym."
    return descriptions


# ---------------------------------------------
# SIDEBAR – NAWIGACJA
# ---------------------------------------------
st.sidebar.title("Nawigacja")

page = st.sidebar.radio(
    "Wybierz widok:",
    ["Generowanie danych", "Klasteryzacja", "Raport klastrów", "Profil indywidualny"]
)


# ---------------------------------------------
# STRONA 1 – GENEROWANIE DANYCH
# ---------------------------------------------
if page == "Generowanie danych":
    st.title("Generowanie danych (100 profili)")

    if st.button("Generuj dane"):
        st.session_state["data"] = generate_data()
        st.success("Wygenerowano 100 profili!")

    if "data" in st.session_state:
        st.subheader("Podgląd danych")
        st.dataframe(st.session_state["data"].head(10))


# ---------------------------------------------
# STRONA 2 – KLASTERYZACJA
# ---------------------------------------------
elif page == "Klasteryzacja":
    st.title("Klasteryzacja danych")

    if "data" not in st.session_state:
        st.warning("Najpierw wygeneruj dane.")
    else:
        df = st.session_state["data"]

        n_clusters = st.slider("Liczba klastrów:", 2, 8, 4)

        if st.button("Uruchom klasteryzację"):
            clustering_setup = setup(
                df.drop(columns=["client_id", "miasto"]),
                # silent=True,
                verbose=False
            )
            model = create_model("kmeans", num_clusters=n_clusters)
            clustered_df = assign_model(model)

            # Dodanie ID i miasta z oryginalnego df
            clustered_df["client_id"] = df["client_id"]
            clustered_df["miasto"] = df["miasto"]

            st.session_state["clustered"] = clustered_df
            st.session_state["cluster_info"] = generate_cluster_descriptions(n_clusters)

            st.success("Klasteryzacja zakończona!")

        if "clustered" in st.session_state:
            st.subheader("Wyniki klasteryzacji (podgląd)")
            st.dataframe(st.session_state["clustered"].head(10))


# -------------------------------------------      st.write("---")


# ---------------------------------------------
# STRONA 4 – PROFIL INDYWIDUALNY
# ---------------------------------------------
elif page == "Profil indywidualny":
    st.title("Profil indywidualny")

    if "clustered" not in st.session_state:
        st.warning("Najpierw przeprowadź klasteryzację.")
    else:
        df = st.session_state["clustered"]
        info = st.session_state["cluster_info"]

        selected_id = st.number_input(
            "Wybierz client_id:", min_value=1, max_value=100, step=1
        )

        osoba = df[df["client_id"] == selected_id]

        if not osoba.empty:
            cluster_idx = int(osoba["Cluster"].values[0])
            cluster_name = cluster_names_list[cluster_idx]
            opis = info[cluster_name]

            st.subheader(f"Klient ID: {selected_id}")
            st.write(f"Miasto: {osoba['miasto'].values[0]}")
            st.write(f"Przypisany klaster: **{cluster_name}**")
            st.write(opis)

            st.info("To tylko model edukacyjny — nie jest to diagnoza psychologiczna.")
            