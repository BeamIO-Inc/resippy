import csv
import numpy as np


class CsvUtils:
    @classmethod
    def single_line_header_csv_to_dict(cls,
                                       fname,
                                       delimiter=","
                                       ):
        header = None
        data = None
        with open(fname) as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader) # skip header
            data = [row for row in reader]
            data = np.asarray(data)
        csv_dict = {}
        for i, header_name in enumerate(header):
            csv_dict[header_name] = data[:, i]
        return csv_dict