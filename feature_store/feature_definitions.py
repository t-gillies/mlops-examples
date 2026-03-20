from pathlib import Path
from datetime import timedelta
from feast.types import Float64, Int64
from feast import (
    Entity,
    FeatureView,
    FileSource,
    Field,
    ValueType,
    FeatureService,
)
from feast.data_format import ParquetFormat


REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = REPO_ROOT / "data" / "features" / "current"

features_source = FileSource(
    name="features_snapshot",
    path=str((SNAPSHOT_DIR / "features.parquet").resolve()),
    timestamp_field="event_timestamp",
    file_format=ParquetFormat(),
)

targets_source = FileSource(
    name="targets_snapshot",
    path=str((SNAPSHOT_DIR / "targets.parquet").resolve()),
    timestamp_field="event_timestamp",
    file_format=ParquetFormat(),
)

patient = Entity(name = 'patient_id', value_type = ValueType.INT64, description = 'ID of the patient')

features_fv = FeatureView(
    name = "features_df_feature_view",
    ttl = timedelta(seconds = 86400*2),
    entities = [patient],
    schema = [
        Field(name = "mean radius", dtype = Float64),
        Field(name = "mean texture", dtype = Float64),
        Field(name = "mean perimeter", dtype = Float64),
        Field(name = "mean area", dtype = Float64),
        Field(name = "mean smoothness", dtype = Float64),
        Field(name = "mean compactness", dtype = Float64),
        Field(name = "mean concave points", dtype = Float64),
        Field(name = "mean symmetry", dtype = Float64),
        Field(name = "mean fractal dimension", dtype = Float64),
        Field(name = "radius error", dtype = Float64),
        Field(name = "texture error", dtype = Float64),
        Field(name = "perimeter error", dtype = Float64),
        Field(name = "area error", dtype = Float64),
        Field(name = "smoothness error", dtype = Float64),
        Field(name = "compactness error", dtype = Float64),
        Field(name = "concavity error", dtype = Float64),
        Field(name = "concave points error", dtype = Float64),
        Field(name = "symmetry error", dtype = Float64),
        Field(name = "fractal dimension error", dtype = Float64),
        Field(name = "worst radius", dtype = Float64),
        Field(name = "worst texture", dtype = Float64),
        Field(name = "worst perimeter", dtype = Float64),
        Field(name = "worst area", dtype = Float64),
        Field(name = "worst smoothness", dtype = Float64),
        Field(name = "worst compactness", dtype = Float64),
        Field(name = "worst concavity", dtype = Float64),
        Field(name = "worst concave points", dtype = Float64),
        Field(name = "worst symmetry", dtype = Float64),
        Field(name = "worst fractal dimension", dtype = Float64),
    ],
    source = features_source
)

target_fv = FeatureView(
    name = "target_df_feature_view",
    ttl = timedelta(seconds = 86400*2),
    entities = [patient],
    schema = [
        Field(name = "target", dtype = Int64)
    ],
    source = targets_source
)

patient_features = FeatureService(
    name="patient_features",
    features=[features_fv],
)
