from pathlib import Path

from aelcha.user_interface import read_file_selection

if __name__ == "__main__":
    file_selection = read_file_selection(Path("File_Selection.xlsx"))
