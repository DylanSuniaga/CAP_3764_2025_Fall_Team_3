import pandas as pd
def load_dataset_and_encode(df_path, ids_path):
    df = pd.read_csv(df_path)
    ids = pd.read_csv(ids_path)
    df_encoded = df.copy()
    mappings = {}
    for col in df_encoded.select_dtypes(exclude="number").columns:
        df_encoded[col], uniques = pd.factorize(df_encoded[col])
        mappings[col] = dict(enumerate(uniques))
    return df, ids, df_encoded, mappings