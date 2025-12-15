import streamlit as st
import pandas as pd
from faker import Faker
import random
from pycaret.clustering import setup, create_model, assign_model

# -------------------------------------------------
# KONFIGURACJA STRONY
# -------------------------------------------------
st.set_page_config(
    page_title="ClusterCraft – Analiza klastrów",
    layout="wide"
)

fake = Faker("pl_PL")

# -------------------------------------------------
# GENEROWANIE DANYCH
# -------------------------------------------------
def generate_data():
    data = []
    for i in range(1, 101):
        data.append({
            "client_id": i,
            "miasto": fake.city(),
            "pytanie_1": random.randint(1, 5),
            "pytanie_2": random.randint(1, 5),
            "pytanie_3": random.randint(1, 5),
            "pytanie_4": random.randint(1, 5),
            "pytanie_5": random.randint(1, 5),
        })
    return pd.DataFrame(data)

# -------------------------------------------------
# NAZWY I OPISY KLASTRÓW
# -------------------------------------------------
cluster_names_list = [
    "Innowatorzy",
    "Analitycy",
    "Eksploratorzy",
    "Mediatorzy",
    "Stabilizatorzy",
    "Twórcy Adaptacyjni",
    "Systemowcy",
    "Wizjonerzy"
]

def generate_cluster_descriptions(n_clusters):
    descriptions = {}
    for i in range(n_clusters):
        name = cluster_names_list[i] if i < len(cluster_names_list) else f"Klastr {i}"
        descriptions[name] = (
            f"{name}: grupa wyróżniająca się charakterystycznym "
            f"stylem odpowiedzi i spójnym profilem cech."
        )
    return descriptions

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("Nawigacja")

page = st.sidebar.radio(
    "Wybierz widok:",
    [
        "Generowanie danych",
        "Klasteryzacja",
        "Raport klastrów",
        "Profil indywidualny"
    ]
)

# -------------------------------------------------
# STRONA 1 – GENEROWANIE DANYCH
# -------------------------------------------------
if page == "Generowanie danych":
    st.title("Generowanie danych (100 profili)")

    if st.button("Generuj dane"):
        st.session_state["data"] = generate_data()
        st.success("Wygenerowano 100 profili!")

    if "data" in st.session_state:
        st.subheader("Podgląd danych")
        st.dataframe(st.session_state["data"].head(10))

# -------------------------------------------------
# STRONA 2 – KLASTERYZACJA
# -------------------------------------------------
elif page == "Klasteryzacja":
    st.title("Klasteryzacja danych")

    if "data" not in st.session_state:
        st.warning("Najpierw wygeneruj dane.")
    else:
        df = st.session_state["data"]

        n_clusters = st.slider("Liczba klastrów:", 2, 8, 4)

        if st.button("Uruchom klasteryzację"):
            setup(
                df.drop(columns=["client_id", "miasto"]),
                verbose=False
            )

            model = create_model("kmeans", num_clusters=n_clusters)
            clustered_df = assign_model(model)

            clustered_df["Cluster"] = clustered_df["Cluster"].str.replace("Cluster ", "").astype(int)
            clustered_df["client_id"] = df["client_id"]
            clustered_df["miasto"] = df["miasto"]

            st.session_state["clustered"] = clustered_df
            st.session_state["cluster_info"] = generate_cluster_descriptions(n_clusters)

            st.success("Klasteryzacja zakończona!")

        if "clustered" in st.session_state:
            st.subheader("Podgląd wyników klasteryzacji")
            st.dataframe(st.session_state["clustered"].head(10))

# -------------------------------------------------
# STRONA 3 – RAPORT KLASTRÓW
# -------------------------------------------------
elif page == "Raport klastrów":
    st.title("Raport klastrów – pełny przegląd")

    if "clustered" not in st.session_state:
        st.warning("Najpierw przeprowadź klasteryzację.")
    else:
        df = st.session_state["clustered"]
        cluster_info = st.session_state["cluster_info"]

        st.write("## Raport wszystkich klastrów")
        st.write("---")

        for cluster_id in sorted(df["Cluster"].unique()):
            cluster_name = (
                cluster_names_list[int(cluster_id)]
                if int(cluster_id) < len(cluster_names_list)
                else f"Klastr {cluster_id}"
            )

            st.subheader(f"Klastr {cluster_id} – {cluster_name}")
            st.write(cluster_info.get(cluster_name, ""))

            cluster_df = df[df["Cluster"] == cluster_id]

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Liczebność:**")
                st.write(len(cluster_df))

            with col2:
                st.write("**Najczęstsze miasta:**")
                st.write(cluster_df["miasto"].value_counts().head(5))

            st.write("**Średnie odpowiedzi:**")
            st.dataframe(
                cluster_df[[f"pytanie_{i}" for i in range(1, 6)]]
                .mean()
                .round(2)
                .to_frame("średnia")
                .T
            )

            st.write("---")

        # -------- ZBIORCZE PORÓWNANIE --------
        st.write("## Zbiorcze porównanie klastrów")

        summary_rows = []

        for cluster_id in sorted(df["Cluster"].unique()):
            cluster_df = df[df["Cluster"] == cluster_id]
            cluster_name = (
                cluster_names_list[int(cluster_id)]
                if int(cluster_id) < len(cluster_names_list)
                else f"Klastr {cluster_id}"
            )

            row = {
                "Klaster": cluster_id,
                "Nazwa klastra": cluster_name,
                "Liczebność": len(cluster_df)
            }

            for i in range(1, 6):
                row[f"Śr. pytanie {i}"] = round(
                    cluster_df[f"pytanie_{i}"].mean(), 2
                )

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)

# -------------------------------------------------
# STRONA 4 – PROFIL INDYWIDUALNY
# -------------------------------------------------
elif page == "Profil indywidualny":
    st.title("Profil indywidualny")

    if "clustered" not in st.session_state:
        st.warning("Najpierw przeprowadź klasteryzację.")
    else:
        df = st.session_state["clustered"]
        cluster_info = st.session_state["cluster_info"]

        selected_id = st.number_input(
            "Podaj numer klienta:",
            min_value=1,
            max_value=100,
            step=1
        )

        person = df[df["client_id"] == selected_id]

        if person.empty:
            st.error("Nie znaleziono takiego klienta.")
        else:
            person = person.iloc[0]
            cluster_id = int(person["Cluster"])
            cluster_name = (
                cluster_names_list[cluster_id]
                if cluster_id < len(cluster_names_list)
                else f"Klastr {cluster_id}"
            )

            st.subheader(f"Klient {selected_id}")
            st.write(f"Miasto: **{person['miasto']}**")
            st.write(f"Należy do klastra: **{cluster_name}**")

            st.write("### Profil odpowiedzi")
            st.dataframe(
                pd.DataFrame(
                    person[[f"pytanie_{i}" for i in range(1, 6)]]
                    .to_dict(),
                    index=["odpowiedź"]
                )
            )

            st.write("### Opis klastra")
            st.info(cluster_info.get(cluster_name, ""))