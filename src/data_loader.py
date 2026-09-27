"""Validated CSV and Excel loading helpers."""

from __future__ import annotations
import csv
import io

from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd


class DataLoadError(ValueError):
    """Raised when an uploaded dataset cannot be safely loaded."""


class EmptyDatasetError(DataLoadError):
    """Raised when a file contains no usable tabular data."""


class UnsupportedFileError(DataLoadError):
    """Raised for formats that are not supported by the loader."""


def _read_bytes(source: str | Path | BinaryIO | bytes | bytearray) -> tuple[bytes, str]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            return path.read_bytes(), path.name
        except OSError as exc:
            raise DataLoadError("The selected file could not be read.") from exc

    filename = getattr(source, "name", "upload.csv")
    if isinstance(source, (bytes, bytearray)):
        return bytes(source), str(filename)
    try:
        if hasattr(source, "getvalue"):
            contents = source.getvalue()
        else:
            position = source.tell() if hasattr(source, "tell") else None
            if hasattr(source, "seek"):
                source.seek(0)
            contents = source.read()
            if position is not None and hasattr(source, "seek"):
                source.seek(position)
        return bytes(contents), str(filename)
    except (OSError, TypeError, ValueError) as exc:
        raise DataLoadError("The uploaded file could not be read.") from exc


def _unique_column_names(columns: list[Any]) -> list[str]:
    names: list[str] = []
    used: set[str] = set()
    for index, value in enumerate(columns, start=1):
        base = str(value).strip() if value is not None else ""
        base = base or f"column_{index}"
        candidate = base
        suffix = 2
        while candidate.casefold() in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        names.append(candidate)
        used.add(candidate.casefold())
    return names


def load_dataset(
    source: str | Path | BinaryIO | bytes | bytearray,
    *,
    max_size_mb: int = 50,
) -> pd.DataFrame:
    """Load a CSV/XLSX/XLS file without modifying its source.

    Blank header cells and duplicate headers receive deterministic unique names.
    The configured size limit is checked before parsing to keep uploads bounded.
    """
    contents, filename = _read_bytes(source)
    if not contents:
        raise EmptyDatasetError("The file is empty.")
    if len(contents) > max_size_mb * 1024 * 1024:
        raise DataLoadError(f"The file exceeds the {max_size_mb} MB upload limit.")

    extension = Path(filename).suffix.casefold()
    if extension not in {".csv", ".xlsx", ".xls"}:
        raise UnsupportedFileError("Supported file formats are CSV, XLSX, and XLS.")

    try:
        if extension == ".csv":
            frame = None
            for encoding in ("utf-8-sig", "cp1252", "latin-1"):
                try:
                    text = contents.decode(encoding)
                    try:
                        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
                    except csv.Error:
                        dialect = csv.excel
                    headers = next(csv.reader(io.StringIO(text, newline=""), dialect))
                    frame = pd.read_csv(
                        BytesIO(contents),
                        encoding=encoding,
                        header=0,
                        names=_unique_column_names(headers),
                        low_memory=False,
                    )
                    break
                except UnicodeDecodeError:
                    continue
                except StopIteration as exc:
                    raise EmptyDatasetError("The file does not contain a header row or data.") from exc
            if frame is None:
                raise DataLoadError("The CSV encoding could not be recognized.")
        else:
            engine = "openpyxl" if extension == ".xlsx" else None
            headers = pd.read_excel(BytesIO(contents), header=None, nrows=1, engine=engine)
            if headers.empty:
                raise EmptyDatasetError("The file does not contain a header row or data.")
            frame = pd.read_excel(BytesIO(contents), engine=engine)
            frame.columns = _unique_column_names(headers.iloc[0].tolist())
    except DataLoadError:
        raise
    except pd.errors.EmptyDataError as exc:
        raise EmptyDatasetError("The file does not contain a header row or data.") from exc
    except ImportError as exc:
        raise DataLoadError("Excel support is unavailable. Install openpyxl for XLSX or xlrd for XLS files.") from exc
    except Exception as exc:
        raise DataLoadError("The file could not be parsed. Check that it is a valid CSV or Excel workbook.") from exc

    if frame.shape[1] == 0:
        raise EmptyDatasetError("The file does not contain any columns.")
    if frame.shape[0] == 0:
        raise EmptyDatasetError("The file contains column headers but no data rows.")

    frame.columns = _unique_column_names(list(frame.columns))
    return frame