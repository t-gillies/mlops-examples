# This is an example feature definition file

from datetime import timedelta
from feast.types import Float32, Float64, Int64
from feast import (
    Entity,
    FeatureView,
    Field,
    ValueType,
    FeatureService
)

from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource

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
    source = PostgreSQLSource(table="public.features_df", timestamp_field="event_timestamp")
)

target_fv = FeatureView(
    name = "target_df_feature_view",
    ttl = timedelta(seconds = 86400*2),
    entities = [patient],
    schema = [
        Field(name = "target", dtype = Int64)
    ],
    source = PostgreSQLSource(table="public.target_df", timestamp_field="event_timestamp")
)

patient_features = FeatureService(
    name="patient_features",
    features=[features_fv],
)