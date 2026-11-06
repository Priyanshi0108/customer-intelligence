from pathlib import Path              
import pandas as pd


REQUIRED_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
}


def load_dataset(file_path:str | Path) ->pd.DataFrame:

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    if file_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(file_path)

    elif file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)

    else:
        raise ValueError("Only CSV and XLSX files are supported")     


    missing_column = REQUIRED_COLUMNS - set(df.columns)   


    if missing_column:
        raise ValueError(
            f"Missing columns : {missing_column}"
        )


    
    return df


