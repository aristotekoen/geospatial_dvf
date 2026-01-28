"""
Test DVF aggregation logic with realistic edge cases.

Based on the DVF multilignes example:
- Mutation 2013P00181
- 2 dispositions (180k€ and 317k€)
- Multiple parcels (C294, KT33, KT34)
- Multiple locals (Appartement, Maison, Dépendance)
- Multiple nature_culture per parcel (Sol, Jardin, Agrément Sport, Agrément Chasse)
"""

import tempfile
from pathlib import Path

import polars as pl
import pytest


def create_sample_dvf_data() -> pl.DataFrame:
    """Create realistic DVF sample data based on the multilignes example.
    
    Mutation 2013P00181:
    - Disposition 2: Appartement 75m² on C294 (180k€)
    - Disposition 3: Maison 120m² + Appartement 55m² + Dépendance 30m² on KT34, Jardin on KT33 (317k€)
    
    Each local appears multiple times due to different nature_culture.
    This tests the edge case where multiple property types (Maison + Appartement) are in the same disposition.
    """
    data = {
        # Disposition 2 - Appartement on C294
        "id_mutation": [
            "2013P00181",  # Appart - Jardin
        ],
        "date_mutation": ["2013-05-15"],
        "numero_disposition": [2],
        "nature_mutation": ["Vente"],
        "valeur_fonciere": [180000.0],
        "adresse_numero": ["12"],
        "adresse_suffixe": [None],
        "adresse_nom_voie": ["RUE DE LA PAIX"],
        "adresse_code_voie": ["0001"],
        "code_postal": ["75001"],
        "code_commune": ["75101"],
        "nom_commune": ["PARIS 1ER"],
        "code_departement": ["75"],
        "ancien_code_commune": [None],
        "ancien_nom_commune": [None],
        "id_parcelle": ["750010000C0294"],
        "ancien_id_parcelle": [None],
        "numero_volume": [None],
        "lot1_numero": ["1"],
        "lot1_surface_carrez": [75.0],
        "lot2_numero": [None],
        "lot2_surface_carrez": [None],
        "lot3_numero": [None],
        "lot3_surface_carrez": [None],
        "lot4_numero": [None],
        "lot4_surface_carrez": [None],
        "lot5_numero": [None],
        "lot5_surface_carrez": [None],
        "nombre_lots": [1],
        "code_type_local": ["2"],
        "type_local": ["Appartement"],
        "surface_reelle_bati": [75.0],
        "nombre_pieces_principales": [3],
        "code_nature_culture": ["J"],
        "nature_culture": ["Jardin"],
        "code_nature_culture_speciale": [None],
        "nature_culture_speciale": [None],
        "surface_terrain": [1368.0],
        "longitude": [2.3388],
        "latitude": [48.8634],
    }
    
    base_df = pl.DataFrame(data)
    
    # Build more rows for the complex mutation
    rows = []
    
    # Disposition 2: Appartement on C294 (only 1 nature_culture: Jardin)
    rows.append({
        "id_mutation": "2013P00181",
        "date_mutation": "2013-05-15",
        "numero_disposition": 2,
        "nature_mutation": "Vente",
        "valeur_fonciere": 180000.0,
        "adresse_numero": "12",
        "adresse_suffixe": None,
        "adresse_nom_voie": "RUE DE LA PAIX",
        "adresse_code_voie": "0001",
        "code_postal": "75001",
        "code_commune": "75101",
        "nom_commune": "PARIS 1ER",
        "code_departement": "75",
        "ancien_code_commune": None,
        "ancien_nom_commune": None,
        "id_parcelle": "750010000C0294",
        "ancien_id_parcelle": None,
        "numero_volume": None,
        "lot1_numero": "1",
        "lot1_surface_carrez": 75.0,
        "lot2_numero": None,
        "lot2_surface_carrez": None,
        "lot3_numero": None,
        "lot3_surface_carrez": None,
        "lot4_numero": None,
        "lot4_surface_carrez": None,
        "lot5_numero": None,
        "lot5_surface_carrez": None,
        "nombre_lots": 1,
        "code_type_local": "2",
        "type_local": "Appartement",
        "surface_reelle_bati": 75.0,
        "nombre_pieces_principales": 3,
        "code_nature_culture": "J",
        "nature_culture": "Jardin",
        "code_nature_culture_speciale": None,
        "nature_culture_speciale": None,
        "surface_terrain": 1368.0,
        "longitude": 2.3388,
        "latitude": 48.8634,
    })
    
    # Disposition 3: Jardin on KT33 (no local)
    rows.append({
        "id_mutation": "2013P00181",
        "date_mutation": "2013-05-15",
        "numero_disposition": 3,
        "nature_mutation": "Vente",
        "valeur_fonciere": 317000.0,
        "adresse_numero": "14",
        "adresse_suffixe": None,
        "adresse_nom_voie": "RUE DE LA PAIX",
        "adresse_code_voie": "0001",
        "code_postal": "75001",
        "code_commune": "75101",
        "nom_commune": "PARIS 1ER",
        "code_departement": "75",
        "ancien_code_commune": None,
        "ancien_nom_commune": None,
        "id_parcelle": "75001KT33",
        "ancien_id_parcelle": None,
        "numero_volume": None,
        "lot1_numero": None,
        "lot1_surface_carrez": None,
        "lot2_numero": None,
        "lot2_surface_carrez": None,
        "lot3_numero": None,
        "lot3_surface_carrez": None,
        "lot4_numero": None,
        "lot4_surface_carrez": None,
        "lot5_numero": None,
        "lot5_surface_carrez": None,
        "nombre_lots": 0,
        "code_type_local": None,
        "type_local": None,  # No local - just land
        "surface_reelle_bati": None,
        "nombre_pieces_principales": None,
        "code_nature_culture": "J",
        "nature_culture": "Jardin",
        "code_nature_culture_speciale": None,
        "nature_culture_speciale": None,
        "surface_terrain": 1368.0,
        "longitude": 2.3390,
        "latitude": 48.8635,
    })
    
    # Disposition 3: Dépendance on KT34 (3 nature_culture)
    for nature in ["Sol", "Agrément Sport", "Agrément Chasse"]:
        rows.append({
            "id_mutation": "2013P00181",
            "date_mutation": "2013-05-15",
            "numero_disposition": 3,
            "nature_mutation": "Vente",
            "valeur_fonciere": 317000.0,
            "adresse_numero": "14",
            "adresse_suffixe": None,
            "adresse_nom_voie": "RUE DE LA PAIX",
            "adresse_code_voie": "0001",
            "code_postal": "75001",
            "code_commune": "75101",
            "nom_commune": "PARIS 1ER",
            "code_departement": "75",
            "ancien_code_commune": None,
            "ancien_nom_commune": None,
            "id_parcelle": "75001KT34",
            "ancien_id_parcelle": None,
            "numero_volume": None,
            "lot1_numero": None,
            "lot1_surface_carrez": None,
            "lot2_numero": None,
            "lot2_surface_carrez": None,
            "lot3_numero": None,
            "lot3_surface_carrez": None,
            "lot4_numero": None,
            "lot4_surface_carrez": None,
            "lot5_numero": None,
            "lot5_surface_carrez": None,
            "nombre_lots": 0,
            "code_type_local": "3",
            "type_local": "Dépendance",
            "surface_reelle_bati": 30.0,
            "nombre_pieces_principales": 0,
            "code_nature_culture": "S" if nature == "Sol" else "AG",
            "nature_culture": nature,
            "code_nature_culture_speciale": "SPORT" if nature == "Agrément Sport" else ("CHASSE" if nature == "Agrément Chasse" else None),
            "nature_culture_speciale": nature if "Agrément" in nature else None,
            "surface_terrain": 1000.0 if nature == "Sol" else (800.0 if nature == "Agrément Sport" else 3633.0),
            "longitude": 2.3391,
            "latitude": 48.8636,
        })
    
    # Disposition 3: Maison on KT34 (3 nature_culture - same as Dépendance)
    for nature in ["Sol", "Agrément Sport", "Agrément Chasse"]:
        rows.append({
            "id_mutation": "2013P00181",
            "date_mutation": "2013-05-15",
            "numero_disposition": 3,
            "nature_mutation": "Vente",
            "valeur_fonciere": 317000.0,
            "adresse_numero": "14",
            "adresse_suffixe": None,
            "adresse_nom_voie": "RUE DE LA PAIX",
            "adresse_code_voie": "0001",
            "code_postal": "75001",
            "code_commune": "75101",
            "nom_commune": "PARIS 1ER",
            "code_departement": "75",
            "ancien_code_commune": None,
            "ancien_nom_commune": None,
            "id_parcelle": "75001KT34",
            "ancien_id_parcelle": None,
            "numero_volume": None,
            "lot1_numero": None,
            "lot1_surface_carrez": None,
            "lot2_numero": None,
            "lot2_surface_carrez": None,
            "lot3_numero": None,
            "lot3_surface_carrez": None,
            "lot4_numero": None,
            "lot4_surface_carrez": None,
            "lot5_numero": None,
            "lot5_surface_carrez": None,
            "nombre_lots": 0,
            "code_type_local": "1",
            "type_local": "Maison",
            "surface_reelle_bati": 120.0,
            "nombre_pieces_principales": 5,
            "code_nature_culture": "S" if nature == "Sol" else "AG",
            "nature_culture": nature,
            "code_nature_culture_speciale": "SPORT" if nature == "Agrément Sport" else ("CHASSE" if nature == "Agrément Chasse" else None),
            "nature_culture_speciale": nature if "Agrément" in nature else None,
            "surface_terrain": 1000.0 if nature == "Sol" else (800.0 if nature == "Agrément Sport" else 3633.0),
            "longitude": 2.3391,
            "latitude": 48.8636,
        })
    
    # Disposition 3: Appartement on KT34 (3 nature_culture - same parcel as Maison)
    # This tests the case where multiple property types are in the same disposition
    for nature in ["Sol", "Agrément Sport", "Agrément Chasse"]:
        rows.append({
            "id_mutation": "2013P00181",
            "date_mutation": "2013-05-15",
            "numero_disposition": 3,
            "nature_mutation": "Vente",
            "valeur_fonciere": 317000.0,
            "adresse_numero": "14",
            "adresse_suffixe": "bis",
            "adresse_nom_voie": "RUE DE LA PAIX",
            "adresse_code_voie": "0001",
            "code_postal": "75001",
            "code_commune": "75101",
            "nom_commune": "PARIS 1ER",
            "code_departement": "75",
            "ancien_code_commune": None,
            "ancien_nom_commune": None,
            "id_parcelle": "75001KT34",
            "ancien_id_parcelle": None,
            "numero_volume": None,
            "lot1_numero": "3",
            "lot1_surface_carrez": 55.0,
            "lot2_numero": None,
            "lot2_surface_carrez": None,
            "lot3_numero": None,
            "lot3_surface_carrez": None,
            "lot4_numero": None,
            "lot4_surface_carrez": None,
            "lot5_numero": None,
            "lot5_surface_carrez": None,
            "nombre_lots": 1,
            "code_type_local": "2",
            "type_local": "Appartement",
            "surface_reelle_bati": 55.0,
            "nombre_pieces_principales": 2,
            "code_nature_culture": "S" if nature == "Sol" else "AG",
            "nature_culture": nature,
            "code_nature_culture_speciale": "SPORT" if nature == "Agrément Sport" else ("CHASSE" if nature == "Agrément Chasse" else None),
            "nature_culture_speciale": nature if "Agrément" in nature else None,
            "surface_terrain": 1000.0 if nature == "Sol" else (800.0 if nature == "Agrément Sport" else 3633.0),
            "longitude": 2.3391,
            "latitude": 48.8636,
        })
    
    # Add another mutation for diversity (simple case)
    rows.append({
        "id_mutation": "2013P00182",
        "date_mutation": "2013-06-20",
        "numero_disposition": 1,
        "nature_mutation": "Vente",
        "valeur_fonciere": 250000.0,
        "adresse_numero": "5",
        "adresse_suffixe": None,
        "adresse_nom_voie": "AVENUE DES CHAMPS",
        "adresse_code_voie": "0002",
        "code_postal": "75008",
        "code_commune": "75108",
        "nom_commune": "PARIS 8E",
        "code_departement": "75",
        "ancien_code_commune": None,
        "ancien_nom_commune": None,
        "id_parcelle": "75008ABC123",
        "ancien_id_parcelle": None,
        "numero_volume": None,
        "lot1_numero": "2",
        "lot1_surface_carrez": 50.0,
        "lot2_numero": None,
        "lot2_surface_carrez": None,
        "lot3_numero": None,
        "lot3_surface_carrez": None,
        "lot4_numero": None,
        "lot4_surface_carrez": None,
        "lot5_numero": None,
        "lot5_surface_carrez": None,
        "nombre_lots": 1,
        "code_type_local": "2",
        "type_local": "Appartement",
        "surface_reelle_bati": 50.0,
        "nombre_pieces_principales": 2,
        "code_nature_culture": "S",
        "nature_culture": "Sol",
        "code_nature_culture_speciale": None,
        "nature_culture_speciale": None,
        "surface_terrain": 0.0,
        "longitude": 2.3100,
        "latitude": 48.8700,
    })
    
    return pl.DataFrame(rows)


def test_sample_data_structure():
    """Test that sample data has correct structure."""
    # Arrange
    df = create_sample_dvf_data()
    
    # Act
    mutations = df.select("id_mutation").unique()
    dispositions = df.filter(pl.col("id_mutation") == "2013P00181").select("numero_disposition").unique()
    
    print("\n=== Sample DVF Data ===")
    print(f"Total rows: {len(df)}")
    print(df.select(["id_mutation", "numero_disposition", "id_parcelle", "type_local", "nature_culture", "surface_reelle_bati"]))
    
    # Assert
    assert len(mutations) == 2, "Should have 2 mutations"
    assert len(dispositions) == 2, "Mutation 2013P00181 should have 2 dispositions"


def test_remove_duplicate_lines_logic():
    """Test the remove_duplicate_lines logic from V2."""
    # Arrange
    df = create_sample_dvf_data()
    group_cols = ["id_mutation", "numero_disposition", "id_parcelle", "nature_mutation"]
    
    print("\n=== Before remove_duplicate_lines ===")
    print(f"Total rows: {len(df)}")
    kt34_before = df.filter(pl.col("id_parcelle") == "75001KT34")
    print(f"Rows for KT34: {len(kt34_before)}")
    print(kt34_before.select(["type_local", "nature_culture", "surface_reelle_bati"]))
    
    # Act
    first_culture = df.group_by(group_cols).agg(
        pl.col("nature_culture").first().alias("first_nature_culture")
    )
    df_dedup = df.join(first_culture, on=group_cols, how="left")
    df_dedup = df_dedup.filter(pl.col("nature_culture") == pl.col("first_nature_culture"))
    df_dedup = df_dedup.drop("first_nature_culture")
    
    print("\n=== After remove_duplicate_lines ===")
    print(f"Total rows: {len(df_dedup)}")
    kt34 = df_dedup.filter(pl.col("id_parcelle") == "75001KT34")
    print(f"Rows for KT34: {len(kt34)}")
    print(kt34.select(["type_local", "nature_culture", "surface_reelle_bati"]))
    
    # Assert
    # Should have 3 rows for KT34: 1 Dépendance + 1 Maison + 1 Appartement (all with first nature_culture)
    assert len(kt34) == 3, f"KT34 should have 3 rows after dedup, got {len(kt34)}"


def test_filter_maison_appartement():
    """Test filtering to keep only Maison/Appartement."""
    # Arrange
    df = create_sample_dvf_data()
    group_cols = ["id_mutation", "numero_disposition", "id_parcelle", "nature_mutation"]
    
    # Apply remove_duplicate_lines first
    first_culture = df.group_by(group_cols).agg(
        pl.col("nature_culture").first().alias("first_nature_culture")
    )
    df_dedup = df.join(first_culture, on=group_cols, how="left")
    df_dedup = df_dedup.filter(pl.col("nature_culture") == pl.col("first_nature_culture"))
    df_dedup = df_dedup.drop("first_nature_culture")
    
    # Act
    df_filtered = df_dedup.filter(pl.col("type_local").is_in(["Maison", "Appartement"]))
    
    print("\n=== After filtering Maison/Appartement ===")
    print(f"Total rows: {len(df_filtered)}")
    print(df_filtered.select(["id_mutation", "numero_disposition", "type_local", "surface_reelle_bati"]))
    
    # Assert
    # Expected: 
    # - Mutation 2013P00181: 1 Appartement (disp 2) + 1 Maison + 1 Appartement (disp 3)
    # - Mutation 2013P00182: 1 Appartement (disp 1)
    assert len(df_filtered) == 4, f"Should have 4 rows, got {len(df_filtered)}"


def test_surface_calculation():
    """Test that surface_batie_totale is calculated correctly."""
    # Arrange
    df = create_sample_dvf_data()
    group_cols = ["id_mutation", "numero_disposition", "id_parcelle", "nature_mutation"]
    
    # Apply full pipeline
    first_culture = df.group_by(group_cols).agg(
        pl.col("nature_culture").first().alias("first_nature_culture")
    )
    df_dedup = df.join(first_culture, on=group_cols, how="left")
    df_dedup = df_dedup.filter(pl.col("nature_culture") == pl.col("first_nature_culture"))
    df_dedup = df_dedup.drop("first_nature_culture")
    
    # Filter Maison/Appartement
    df_filtered = df_dedup.filter(pl.col("type_local").is_in(["Maison", "Appartement"]))
    
    # Act
    surface_totals = df_filtered.group_by(["id_mutation", "numero_disposition"]).agg(
        pl.col("surface_reelle_bati").sum().alias("surface_batie_totale"),
        pl.col("valeur_fonciere").first().alias("prix"),
        pl.col("type_local").first()
    ).with_columns(
        (pl.col("prix") / pl.col("surface_batie_totale")).alias("prix_m2")
    ).sort(["id_mutation", "numero_disposition"])
    
    print("\n=== Surface and Price Calculation ===")
    print(surface_totals)
    
    disp2 = surface_totals.filter(
        (pl.col("id_mutation") == "2013P00181") & 
        (pl.col("numero_disposition") == 2)
    )
    disp3 = surface_totals.filter(
        (pl.col("id_mutation") == "2013P00181") & 
        (pl.col("numero_disposition") == 3)
    )
    expected_prix_m2_disp3 = 317000.0 / 175.0  # ~1811.43
    
    # Assert
    assert disp2["surface_batie_totale"][0] == 75.0, "Disposition 2 should have 75m²"
    assert abs(disp2["prix_m2"][0] - 2400.0) < 0.01, "Disposition 2 should be 2400 €/m²"
    assert disp3["surface_batie_totale"][0] == 175.0, f"Disposition 3 should have 175m² (120+55), got {disp3['surface_batie_totale'][0]}"
    assert abs(disp3["prix_m2"][0] - expected_prix_m2_disp3) < 0.01, f"Disposition 3 should be {expected_prix_m2_disp3:.2f} €/m²"
    
    print("\n✅ All surface and price calculations are correct!")


def test_with_final_functions():
    """Test using actual final version functions on sample data saved to temp CSV."""
    # Arrange
    df = create_sample_dvf_data()
    
    with pl.Config(tbl_rows=-1):  # Show all rows
        print("\n=== Full Sample DVF Data (12 rows) ===")
        print(df.select(["id_mutation", "numero_disposition", "id_parcelle", "type_local", "nature_culture", "surface_reelle_bati"]))
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = Path(f.name)
        df.write_csv(temp_path)
    
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from process_dvf import (
            fill_nature_culture_nulls,
            remove_duplicate_lines,
            drop_unwanted_values,
            compute_total_surface_and_price,
            add_dependency,
            reduce_data,
        )
        
        # Act
        df_lazy = pl.scan_csv(temp_path).with_columns([
            pl.col("date_mutation").str.to_date("%Y-%m-%d"),
        ])
        
        df_lazy = fill_nature_culture_nulls(df_lazy)
        df_lazy = remove_duplicate_lines(df_lazy)
        df_lazy = add_dependency(df_lazy)
        df_lazy = drop_unwanted_values(df_lazy)
        df_lazy = compute_total_surface_and_price(df_lazy)
        df_lazy = reduce_data(df_lazy)
        
        result = df_lazy.collect()
        
        print("\n=== Final Pipeline Result ===")
        print(f"Total aggregated transactions: {len(result)}")
        print(result.select(["id_mutation", "numero_disposition", "type_local", "surface_batie_totale", "valeur_fonciere"]))
        
        disp3 = result.filter(
            (pl.col("id_mutation") == "2013P00181") & 
            (pl.col("numero_disposition") == 3)
        )
        
        # Assert
        assert len(result) == 3, f"Should have 3 aggregated transactions, got {len(result)}"
        assert disp3["surface_batie_totale"][0] == 175.0, f"Final: Disposition 3 should have 175m² (120+55), got {disp3['surface_batie_totale'][0]}"
        
        print("\n✅ Final pipeline produces correct results!")
        
    finally:
        temp_path.unlink()


if __name__ == "__main__":
    print("=" * 80)
    print("DVF AGGREGATION TESTS")
    print("=" * 80)
    
    test_sample_data_structure()
    test_remove_duplicate_lines_logic()
    test_filter_maison_appartement()
    test_surface_calculation()
    
    print("\n" + "=" * 80)
    print("Testing with final version functions...")
    print("=" * 80)
    test_with_final_functions()
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✅")
    print("=" * 80)
