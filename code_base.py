def categorize_columns(df, columns=[], used_uniq=3):
    df = df.copy()
    for column in columns:
        uniq = data.groupby(column).size().sort_values()
        mask = [df[column]==uniq_value for uniq_value in list(uniq.tail(used_uniq).index)]
        df[column] = np.where(np.logical_or.reduce((mask)), df[column], 'OTHER')
    return  df

def prepare_data(data):
    columns = data.select_dtypes(include='object').columns.values
    print(columns)
    prep_data = categorize_columns(data, columns, used_uniq=5)
    prep_data = pd.get_dummies(prep_data, columns=columns, prefix = columns)
    float_columns = data.select_dtypes(include='float64').columns.values
    scaler = StandardScaler()
    prep_data[float_columns] = scaler.fit_transform(prep_data[float_columns])
    return prep_data