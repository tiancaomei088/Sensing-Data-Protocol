from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

from .bfee_reader import BfeeReader
from .xrf_reader import XrfReader
from .elder_reader import ElderReader
from sdp.structure import CSIData

# import future reader here


_READER_REGISTRY = {
    'widar': BfeeReader,
    'gait': BfeeReader,
    'xrf55': XrfReader,
    'elderAL': ElderReader
}


def get_reader_class(dataset: str):
    """
    return the proper reader class according to dataset
    """
    reader_cls = _READER_REGISTRY.get(dataset)

    if reader_cls is None:
        raise ValueError(f"not supported dataset: {dataset}")

    return reader_cls


def _process_file(reader, file_path):
    """
    process function for concurrent reading
    """
    try:
        data = reader.read_file(str(file_path))
        return file_path.name, data, None
    except Exception as e:
        return file_path.name, None, str(e)


def load_data(file_path: str, dataset: str) -> List[CSIData]:
    input_path = Path(file_path)
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"invalid file path: {input_path}")
    files = [f for f in input_path.rglob("*") if f.is_file() and "truth" not in f.name]
    if not files:
        raise IOError(f"no file in folder: {input_path}")

    reader_class = get_reader_class(dataset)
    reader = reader_class()
    csi_data_list = []

    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(_process_file, reader, file_path): file_path for file_path in files}
        for future in as_completed(futures):
            file_name, data, err = future.result()
            if err is None:
                csi_data_list.extend(data) if isinstance(data, List) else csi_data_list.append(data)
                print(f"√ processed: {file_name}\n")
            else:
                print(f"× unable to process {file_name}: {err}\n")
    return csi_data_list
