import pandas as pd
import os


def load_data(base_path, limit_days=None):
    data_files = os.listdir(base_path)
    df = pd.DataFrame()
    for day, filename in enumerate(data_files):
        if limit_days is not None and day == limit_days:
            break
        file_dir = os.path.join(base_path, filename)
        candles = pd.read_csv(file_dir, parse_dates=[0]).set_index("Date")
        if not df.empty and df.index[-1] == candles.index[0]:
            candles = candles.drop([candles.index[0]], axis=0)
        df = pd.concat([df, candles], axis=0)

    assert df.duplicated().sum() == 0
    assert not df.isnull().sum().any(), f"df contains NaN"
    return df
