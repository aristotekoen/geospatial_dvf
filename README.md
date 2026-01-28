# DVF Interactive Price Map

**Live Demo:** [https://pub-52ee07814bfa467bad60b77c80d53dd3.r2.dev/index.html](https://pub-52ee07814bfa467bad60b77c80d53dd3.r2.dev/index.html)

![alt text](image.png) 

An interactive map visualizing real estate price estimates from the French DVF (Demandes de Valeurs Foncières) database, covering transactions from 2020 to June 2025. The map displays median price per square meter aggregated at six geographic levels, with property type filtering (All, Apartments, Houses).

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start Guide](#quick-start-guide)
3. [Tech Stack](#tech-stack)
4. [Data Sources](#data-sources)
5. [Description of the DVF Dataset](#description-of-the-dvf-dataset)
6. [Pipeline Overview](#pipeline-overview)
7. [DVF Processing Method](#dvf-processing-method)
8. [Outlier Removal](#outlier-removal)
9. [Time-Adjusted Median Price](#time-adjusted-median-price)
10. [Aggregation and Spatial Join](#aggregation-and-spatial-join)
11. [Map Generation](#map-generation)
12. [Hosting and Deployment](#hosting-and-deployment)

---

## Introduction

This project develops an interactive map visualizing real estate price estimates (aggregates) from the DVF database between 2020 and June 2025, by property type, at six levels of geographic granularity:

1. **Country** (France)
2. **Region**
3. **Department**
4. **Commune**
5. **IRIS** (neighborhood-level statistical zones)
6. **Parcels** (building plots)

The price estimate used for aggregations is the **median price per square meter**, chosen for its robustness to outliers. In addition to the raw median computed over the entire 2020–2025 period, we also compute and display a **time-adjusted median price** that normalizes historical transactions to current market values.

The map transitions between aggregation levels as the user zooms in, from the least granular (France) to the most granular (parcels). At each level:

- Polygons are colored according to the median price per square meter using fixed color scale from 1.5k EUR to 8k EUR
- Hovering over a polygon displays: median price, mean price, number of transactions, and interquartile range (Q1–Q3)
- Polygons with fewer than 5 transactions are displayed in light gray (except at parcel level where we color them based on the scale)
- Polygons with no recorded transactions are displayed in dark gray (except at parcel level)

A panel displays the top 10 cities ranked by number of transactions, with their median price per square meter.

---

## Quick Start Guide

### Prerequisites

- **Python 3.10+**
- **uv** (Python package manager) — [Installation guide](https://docs.astral.sh/uv/getting-started/installation/)
- **tippecanoe** (for PMTiles generation) — [Installation guide](https://github.com/felt/tippecanoe#installation)
- ~20GB disk space for raw data and intermediate files

### Installation

```bash
# Clone the repository
git clone https://github.com/aristotekoen/geospatial_dvf.git
cd geospatial_dvf

# Install dependencies using uv
uv sync
```

### Running the Full Pipeline

```bash
# Run the complete pipeline (download → process → aggregate → geometries → parcels → convert)
uv run pipeline.py --all

# Or run individual steps:
uv run pipeline.py --download    # Download DVF, INSEE, and geometry data
uv run pipeline.py --process     # Process DVF transactions
uv run pipeline.py --aggregate   # Compute price aggregates
uv run pipeline.py --geometries  # Join aggregates with administrative boundaries
uv run pipeline.py --parcels     # Generate parcel-level PMTiles from cadastre
uv run pipeline.py --convert     # Convert GeoJSON to PMTiles
```

### Serving the Map Locally

```bash
# Start the local map server
uv run pipeline.py --serve
```

### Deploying to Cloudflare R2 (Optional)

To host the map on Cloudflare R2:

1. Copy `.env.example` to `.env` and fill in your R2 credentials:

```bash
cp .env.example .env
```

2. Configure your `.env` file:

```dotenv
R2_ACCOUNT_ID=your-account-id
R2_ACCESS_KEY_ID=your-access-key-id
R2_SECRET_ACCESS_KEY=your-secret-access-key
R2_BUCKET_NAME=your-bucket-name
```

3. Upload the map files:

```bash
uv run upload_to_r2.py
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Data Processing** | Polars | High-performance DataFrame operations for DVF processing |
| **Spatial Operations** | GeoPandas, Shapely | Spatial joins, geometry manipulation |
| **Geometry Storage** | GeoJSON, GeoPackage | Intermediate geometry formats |
| **Vector Tiles** | PMTiles | Efficient single-file vector tile format for web maps |
| **Tile Conversion** | tippecanoe | Converting GeoJSON to optimized PMTiles |
| **Map Rendering** | MapLibre GL JS | Client-side vector map rendering |
| **Map Hosting** | Cloudflare R2 | Static file hosting with CDN |
| **Local Server** | FastAPI + Uvicorn | Development server for local testing |
| **Dependency Management** | uv | Fast Python package management |

### Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Raw Data      │     │   Processing    │     │   Map Output    │
│                 │     │                 │     │                 │
│ • DVF (CSV)     │────▶│ • Clean/Filter  │────▶│ • PMTiles       │
│ • INSEE refs    │     │ • Aggregate     │     │ • JSON data     │
│ • Geometries    │     │ • Spatial join  │     │ • index.html    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  Cloudflare R2  │
                                               │  (Static Host)  │
                                               └─────────────────┘
```

---

## Data Sources

| Data | Source | Description |
|------|--------|-------------|
| **DVF** | [data.gouv.fr](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres-geolocalisees) | Geolocalized real estate transactions (2020–2025) |
| **Administrative Boundaries** | [IGN Admin Express](https://geoservices.ign.fr/adminexpress) | Regions, departments, communes (GeoPackage) |
| **IRIS Contours** | [IGN CONTOURS-IRIS](https://geoservices.ign.fr/contoursiris) | Neighborhood-level statistical zones |
| **Cadastre Parcels** | [cadastre.data.gouv.fr](https://cadastre.data.gouv.fr/) | Building plot geometries per commune |
| **INSEE COG** | [INSEE](https://www.insee.fr/fr/information/2560452) | Official geographic code mapping (departments ↔ regions) |

We use the **geolocalized format** of DVF which includes latitude/longitude coordinates, enabling spatial joins with IRIS zones that are not otherwise available in the base DVF dataset.

---

## Description of the DVF Dataset

### Mutation

The data is structured as **mutations**. Each mutation corresponds to a group of one or more **transactions**.

### Disposition

Within a mutation, there exist **dispositions**. In most cases, there is one disposition per mutation. However, when multiple dispositions exist within a mutation, each disposition is associated with its own price. Thus, to obtain the total price of a mutation with multiple dispositions, one must sum the property values (valeurs foncières) of all dispositions.

### Parcel

Each mutation is composed of property values that can be grouped into dispositions. These property values belong to **parcels** (parcelles).

### Land Use Type (Nature de Culture)

Parcels constitute an administrative geographic division corresponding to land that may or may not be developed, classified by their **land use type** (nature de culture) and **special land use type** (nature de culture spéciale).

Each parcel may have multiple land use types and multiple special land use types.

In the DVF data, the land surface area is provided for each land use type and special land use type combination. Thus, the total land surface of a parcel corresponds to the sum of the land surfaces for each land use/special land use group.

In some cases, the land use type is not specified, yet a land surface area may still be present.

### Property Type (Type de Local)

On a parcel, there may be multiple properties classified by the variable "Type Local". These properties can take the following values:

- Apartment (Appartement)
- House (Maison)
- Outbuilding (Dépendance)
- Commercial premises (Locaux Commerciaux)

### Data Structure

Based on the above information, a mutation is represented as follows:

- If the mutation comprises n dispositions, then the mutation will be duplicated into n rows
- If disposition m ∈ [1, n) comprises n' parcels, then the disposition is duplicated into n' rows
- If the parcel contains x property types, then the parcel in question is duplicated into x rows
- If a parcel contains y land use types, then each property of the parcel is duplicated into y rows
- If for a given land use type a parcel is subdivided into z special land use types, then each row with that land use type within the mutation is duplicated into z rows

> **Example:** A mutation with 2 dispositions, 3 parcels, 1 property on one parcel and another parcel with 2 properties, two land use types within one of the 3 parcels, and 2 special land use types for one of the parcels, will have 8 rows.

### Conclusion: Data Difficult to Exploit at the Property Level

The structure of DVF data poses several challenges when mutations are complex:

- A mutation may contain multiple parcels that are not located in the same place
- A mutation with a single disposition may contain multiple apartments, houses, land plots, and outbuildings, all grouped under the same mutation and under a single price. In this case, it is impossible to determine the price of each individual property within a mutation
- Limited information about the properties comprising the mutation (construction date, property condition, energy performance certificate, heating system, balconies, pools, terraces, parking, orientation, etc.)
- To estimate a property, we need one property per row. **Multi-row mutations pose processing challenges that often must be addressed on a case-by-case basis.** For example, for a mutation with multiple dispositions containing multiple properties per disposition, we cannot treat each property as a separate asset since the displayed price corresponds to the sum of all properties. For a mutation with multiple properties, we could sum the surfaces and number of rooms; however, we could not classify it as either a house or apartment, and a model would not handle a synthetic property with 100 rooms and 1000 m² as well as typical properties. A generalized method has been designed, but certain cases will not be handled optimally.

The method designed to aggregate the properties to a single row is described below in the [dvf-processing-section](#dvf-processing-method) 

---

## Pipeline Overview

The pipeline consists of seven steps:

| Step | Command | Description |
|------|---------|-------------|
| 1 | `--download` | Download DVF data, INSEE references, administrative boundaries, IRIS contours, and cadastre parcels |
| 2 | `--process` | Process DVF transactions: clean, filter, aggregate multi-row mutations, remove outliers, spatial join with IRIS |
| 3 | `--aggregate` | Compute price statistics at all geographic levels (country, region, department, commune, IRIS, parcel) |
| 4 | `--geometries` | Join aggregates with administrative boundary geometries, export to GeoJSON |
| 5 | `--parcels` | Generate parcel-level PMTiles from cadastre data with price attributes |
| 6 | `--convert` | Convert commune and IRIS GeoJSON files to PMTiles format |
| 7 | `--serve` | Start local development server |

### Step 1: Download (`--download`)

Downloads all required data sources:

- **DVF data**: Geolocalized real estate transactions from data.gouv.fr (3.3GB compressed CSV, 20M rows )
- **INSEE references**: Department-to-region mapping tables (COG 2025)
- **Admin Express**: Administrative boundaries for regions, departments, and communes from IGN (GeoPackage format)
- **CONTOURS-IRIS**: Neighborhood-level statistical zone geometries from IGN
- **Cadastre parcels**: Building plot geometries downloaded per commune from cadastre.data.gouv.fr (~35,000 communes)

Files are downloaded only if they don't already exist. Use `--force` to re-download.

### Step 2: Process (`--process`)

Processes raw DVF transactions into a clean, analysis-ready dataset:

1. **Load and parse** the raw CSV with proper type handling
2. **Aggregate multi-row mutations** into single rows (see [DVF Processing Method](#dvf-processing-method))
3. **Filter** to residential properties (apartments and houses) with valid prices and surfaces
4. **Add derived columns**: price per m², region codes
5. **Remove outliers** using hard thresholds and IQR method per commune (see [Outlier Removal](#outlier-removal))
6. **Compute time-adjusted prices** normalized to 2025 market values (see [Time-Adjusted Median Price](#time-adjusted-median-price))
7. **Spatial join with IRIS geometries** to assign IRIS codes to each property using lat/lon coordinates

Output: `data/processed/dvf_processed.parquet` (~350MB, ~4M transactions)

### Step 3: Aggregate (`--aggregate`)

Computes price statistics at all geographic levels:

- Groups transactions by geographic unit (country, region, department, commune, IRIS, parcel)
- Computes for each group and property type (All, Maison, Appartement):
  - Transaction count
  - Mean, median, Q25, Q75 price per m²
  - Time-adjusted median price per m²
- Generates aggregates for multiple time spans: all years, 2023-2025/06, 2024-2025/06, 2025+
- Produces the top 50 cities JSON file for the map top 10 cities panel

Output: `data/aggregates/{time_span}/agg_{level}.parquet`

### Step 4: Geometries (`--geometries`)

Joins aggregated statistics with administrative boundary geometries:

1. Loads boundary polygons from Admin Express (regions, departments, communes) and CONTOURS-IRIS
2. Joins aggregate statistics by geographic code
3. Simplifies geometries to reduce file size while preserving visual quality
4. Exports to GeoJSON format for each level

Output: `map/data/{level}.geojson`

### Step 5: Parcels (`--parcels`)

Generates parcel-level vector tiles from cadastre data:

1. Loads parcel-level price aggregates
2. Processes department by department to manage memory
3. For each commune in the department: loads cadastre GeoJSON, joins with parcel aggregated price data, writes enriched GeoJSON
4. Converts to MBTiles and then PMTiles using tippecanoe and pmtiles with appropriate zoom levels

This step is memory-intensive and processes around 35,000 communes it runs in around 30 mins. Output: `map/data/parcels.pmtiles`

### Step 6: Convert (`--convert`)

Converts commune and IRIS GeoJSON files to PMTiles format:

- Uses tippecanoe to convert GeoJSON to PMTiles
- Sets appropriate min/max zoom levels for each layer

Output: `map/data/communes.pmtiles`, `map/data/iris.pmtiles`

### Step 7: Serve (`--serve`)

Starts a local development server:

- Serves the `map/` directory via FastAPI + Uvicorn
- Available at `http://localhost:8765`

---

## DVF Processing Method

### Two Cases

- **Simple mutations** (1 row): No processing required
- **Multi-row mutations**: Processing required to transform into a single row

### Multi-Row Aggregation Method

#### 1. Fill Null Land Use Values

Fill null values in land use columns (`nature_culture`, `nature_culture_speciale`) with "unknown" to ensure consistent grouping.

#### 2. Removal of Duplicate Rows per Mutation

1. Group by: Mutation, disposition, parcel, **mutation nature**
2. For each group, identify the first land use type (`nature_culture`)
3. Keep only rows where the land use type matches the first value of the group
4. This removes duplicates created during the duplication by land use type and special land use type

#### 3. Creation of a Dependency Variable

1. Group by: Mutation, disposition, parcel, land use type, special land use type, mutation nature
2. If the group contains a property type "Dépendance" (outbuilding), set `has_dependency = True`

#### 4. Removal of Unwanted Records

Filter to keep only relevant transactions:

- Mutation nature must be in: Sale (Vente), Off-plan sale (VEFA), Auction (Adjudication)
- Property type must be: Apartment (Appartement) or House (Maison)
- Property value must be > 100€
- Built surface must be > 0 and not null
- Latitude and longitude must be present (for spatial join)

#### 5. Calculation of Total Built Surface and Price

1. Group by mutation and disposition
2. Compute total built surface: sum of `surface_reelle_bati` across the group
3. Compute mean property value across the group (theoretically all property_values/valeur_fonciere should be equal within a group but by security we aggregate with mean() instead of first())
4. Calculate individual property price within the mutation-disposition group proportionally to surface:
   
   `prix_de_vente = (valeur_fonciere / surface_batie_totale) × surface_reelle_bati`

#### 6. Data Reduction

1. Group by mutation and disposition
2. Aggregation:
   - **First value** (constant within group): date, mutation nature, property value/valeur_fonciere, address, postal code, commune, department, coordinates, dependency flag, total built surface
   - **List** (preserved for potential later use): parcel IDs, built surfaces, land use types, prix_de_vente
   - **Sum**: number of main rooms

> **Result:** A dataset with one row per mutation-disposition

### Additional Processing Steps

After multi-row aggregation, the following steps are applied:

#### 7. Add Price per Square Meter

Compute `prix_m2 = valeur_fonciere / surface_batie_totale` and create a unique key (`cle_principale`) from mutation ID and disposition number.

#### 8. Extract First Parcel ID

Extract the first parcel ID from the list to use as `id_parcelle_unique` for parcel-level joins.

#### 9. Add Region Information

Join with INSEE department-to-region mapping to add `code_region` and `nom_region` columns.

#### 10. Outlier Removal

See [Outlier Removal](#outlier-removal) section below.

#### 11. Time-Adjusted Price Calculation

Compute time-adjusted price per m² normalized to 2025 market values. See [Time-Adjusted Median Price](#time-adjusted-median-price) section.

#### 12. Spatial Join with IRIS

Perform point-in-polygon spatial join using lat/lon coordinates to assign each transaction to its IRIS zone (`code_iris`, `nom_iris`).

---

## Outlier Removal

The DVF database contains many outliers such as transactions with an iconic price, surfaces of 1 sqm. Also as we mentioned before, mutations with multiple rows might correspond to multiple properties as part of a complex transaction and the total surface or number of rooms corresponding to this group would not be representative of a classic singe property transaction. Thus the surface and number of rooms aggregation can introduce some abnormal values for rare complex transactions.

We can see this clearly when looking at some statistics (computed for a previous DVF project I worked on with the DVF 2024 millesime) on three departments (Creuse (23), Gironde (33), Paris(75)):


Describe of price, price per sqm, log of price per sqm, total surface over the three departments shows abnormal max and min values.

![alt text](<Untitled 4.png>)

Heavily compressed boxplots of the log price per square meter for each of the three departments. 

![alt text](<Untitled 5.png>)

Very long tailed distributions of the log of the price per sqm. 
![alt text](<Untitled 10.png>)

That is why we need to manage these outliers and we do this in a two step process:

### 1. Hard Threshold Filtering

First we remove values we consider abnormal with respect to domain knowledge country wide. 

Removal of aberrant values by imposing business constraints. Rows satisfying the following conditions are removed:

| Variable | Minimum | Maximum |
|----------|---------|---------|
| Built surface | 5 m² | 1,000 m² |
| Property value | 10,000 € | 10,000,000 € |
| Price per m² | 400 €/m² | 30,000 €/m² |
| Number of rooms | 1 | 20 |

### 2. IQR Method per Commune

Here we apply the IQR method to remove outliers per Commune. We group by commune in order to consider outliers in a more homogeneous market and avoid removing properties only in tensed markets (like Paris) or in very rural areas (Creuse) where the price per sqm would be low. If the commune has less than 10 properties we do not perform the IQR method and keep it as is.

We could push this method further by performing the IQR method on finer groups taking into account transaction year, property type and start at a more granular geographical layer (IRIS) however this would lead to many groups with insufficient data to compute quantiles and would require to dynamically move to less granular levels for these. For the sake on simplicity we do it at the commune level and assume that they are sufficiently homogeneous to suppress outliers although we should keep in mind that this could be further improved.  

For a given variable, the interquartile range is defined as the absolute difference between the first and third quartiles:

$$IQR = Q_{0.75} - Q_{0.25}$$

where $Q_j$ is the j-th quantile.

A value x is considered extreme for a given variable if:

$$x \leq Q_{0.25} - 1.5 \times IQR$$

$$x \geq Q_{0.75} + 1.5 \times IQR$$


**Method applied:**

- Variables considered: built surface, number of rooms, property value, price per m²
- Group properties by commune
- For each group: if the number of properties is less than 10, no filtering is applied (calculating a quantile on fewer than 10 rows is unreliable)
- If the number of properties in the group exceeds 10:
  - Calculate min and max values according to the IQR method for each variable
  - Remove rows that, for at least one variable, fall outside the acceptable range

---

## Time-Adjusted Median Price

To account for market price evolution over the 2020–2025 period, we compute a **time-adjusted price per square meter** that normalizes all historical transactions to **2025** (the reference year).

### Method

For each transaction, we first compute the median price per m² for each combination of **(department, property type, year)**. This creates a lookup table of market conditions by location, property type, and time period.

The adjustment factor for a transaction is the ratio between:
- The median price in the **reference year (2025)** for the same department and property type
- The median price in the **transaction year** for the same department and property type

The adjusted price is then computed as:

$$\text{prix\_m2\_ajusté} = \text{prix\_m2} \times \frac{\text{median}_{(\text{dept}, \text{type}, 2025)}}{\text{median}_{(\text{dept}, \text{type}, \text{year}_{\text{transaction}})}}$$


### Implementation

1. Extract the transaction year from each transaction date
2. Compute median price per m² for each **(department, property type, year)** combination
3. Build adjustment factors: ratio of 2025 median to transaction year median for each (department, property type) pair
4. For (department, property type) combinations without 2025 data, use the latest available year's median as fallback
5. Apply adjustment factor to each transaction's price per m²

This allows comparing transactions across different years on an equivalent basis, providing a more accurate estimate of current market values.

---

## Aggregation and Spatial Join

### Geographic Levels

Price statistics are computed at six levels of geographic granularity:

| Level | Key Column | Description |
|-------|------------|-------------|
| Country | - | Single aggregate for France |
| Region | `code_region` | 18 metropolitan regions |
| Department | `code_departement` | 101 departments |
| Commune | `code_commune` | ~35,000 municipalities |
| IRIS | `code_iris` | ~50,000 neighborhood zones |
| Parcel | `id_parcelle` | ~2M Individual building plots |

### Statistics Computed

For each geographic level and property type (All, Maison, Appartement):

- Number of transactions
- Mean price per m²
- Median price per m²
- First quartile (Q25) price per m²
- Third quartile (Q75) price per m²
- Time-adjusted median price per m²

### Spatial Join with IRIS

Since the base DVF data does not include IRIS codes, we perform a spatial join using the geolocalized coordinates:

1. Load IRIS polygon geometries (CONTOURS-IRIS GeoPackage)
2. Convert DVF points (lat/lon) to the same coordinate reference system (EPSG:2154 Lambert-93)
3. Perform point-in-polygon spatial join to assign each transaction to its IRIS zone
4. Process in chunks to manage memory usage

### Geometry Join

Aggregated statistics are joined with administrative boundary geometries:

1. Load boundary geometries (Admin Express GeoPackage)
2. Join aggregates by geographic code
3. Export to GeoJSON with simplified geometries for web display

---

## Map Generation

### File Formats by Level

The map uses different formats depending on the data volume at each geographic level:

| Level | Format | Size | Reason |
|-------|--------|------|--------|
| Country | GeoJSON | 0.8 MB | Small, single polygon |
| Regions | GeoJSON | 1.6 MB | 18 polygons, manageable size |
| Departments | GeoJSON | 4.6 MB | 101 polygons, still efficient as GeoJSON |
| Communes | PMTiles | 210 MB | ~35,000 polygons, requires tiling |
| IRIS | PMTiles | 980 MB | ~50,000 polygons, requires tiling |
| Parcels | PMTiles | 966 MB | ~2.4M polygons, requires tiling |

### PMTiles Format

For larger layers (communes, IRIS, parcels), the map uses [PMTiles](https://github.com/protomaps/PMTiles), a single-file archive format for vector tiles that enables efficient serving.

### Conversion Process

GeoJSON files are converted to PMTiles using [tippecanoe](https://github.com/felt/tippecanoe):

```bash
tippecanoe -o output.pmtiles \
  --maximum-zoom=14 \
  --minimum-zoom=0 \
  --simplification=10 \
  --force \
  input.geojson
```

### Parcel Generation

Parcels are processed separately due to their volume (~2.4 million):

1. Download cadastre GeoJSON files per commune
2. Join with parcel-level aggregates
3. Process department by department to manage memory
4. Generate PMTiles per department
5. Merge into final parcel layer

### Map Interface

The web interface provides:

- **Zoom-based layer switching**: Automatically transitions between aggregation levels
- **Property type filter**: Toggle between All, Apartments, Houses
- **Color scale**:  Color mapping from Green (blue) to high (red) prices passing by Yellow.
- **Hover information**: Displays detailed statistics for each polygon
- **Top cities panel**: Shows the 10 largest cities by transaction volume

---

## Hosting and Deployment

### Local Development

The map can be served locally using the built-in FastAPI server:

```bash
uv run pipeline.py --serve
```

This serves the `map/` directory at `http://localhost:8080`.

### Cloudflare R2 Deployment

For hosting, the map is deployed to Cloudflare R2:

1. Configure R2 credentials in `.env`
2. Run the upload script: `uv run upload_to_r2.py`
3. Enable public access on the R2 bucket
4. Access via the public R2 URL

Files uploaded:
- `index.html` — Map interface
- `data/*.pmtiles` — Vector tile archives
- `data/*.geojson` — Pre-computed statistics
- `data/*.json` — Pre-computed statistics


---

## License

This project uses open data from the French government. The DVF data is published under the [Licence Ouverte / Open License](https://www.etalab.gouv.fr/licence-ouverte-open-licence/).

## Acknowledgments

- [data.gouv.fr](https://www.data.gouv.fr/) for DVF and cadastre data
- [IGN](https://www.ign.fr/) for administrative boundary geometries
- [INSEE](https://www.insee.fr/) for geographic code references
- [MapLibre](https://maplibre.org/) for the mapping library
- [PMTiles](https://protomaps.com/docs/pmtiles) for the vector tile format
