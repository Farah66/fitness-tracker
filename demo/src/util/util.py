def normalize_dataset(data_table, columns):
    dt_norm = data_table.copy()
    for col in columns:
        dt_norm[col] = (dt_norm[col] - dt_norm[col].mean()) / dt_norm[col].std()
    return dt_norm