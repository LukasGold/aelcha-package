from pathlib import Path

from aelcha.core import process_file
from aelcha.user_interface import read_file_selection

if __name__ == "__main__":
    file_path = Path("File_Selection.xlsx")
    file_selection = read_file_selection(file_path)
    for row in file_selection.selection_rows:
        if row.process_file:
            process_file(row, file_selection.configuration)
