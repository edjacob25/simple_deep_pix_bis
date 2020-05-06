from configparser import ConfigParser
from pathlib import Path
from shutil import rmtree, copy
from random import choice


def main():
    config = ConfigParser()
    config.read("config.ini")
    data_folder = config["ROUTES"]["base_data"]
    initial_path = Path(data_folder) / "Test_files"
    folder_destination = Path(data_folder) / "test"
    if folder_destination.exists():
        rmtree(folder_destination)

    folder_destination.mkdir()

    files = [x.name for x in initial_path.iterdir()]
    final_files = set([x.split("__")[0] for x in files])
    for file in final_files:
        possible_files = [x for x in files if x.startswith(file)]
        chosen = choice(possible_files)
        source = initial_path / chosen
        destination = (folder_destination / file).with_suffix(".png")
        # print(f"Moving {source} to {destination}")
        copy(source, destination)

    print(f"Initial files: {len(files)} -> Final files: {len(final_files)}")


if __name__ == '__main__':
    main()
