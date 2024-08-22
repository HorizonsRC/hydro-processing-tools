"""Copies files from template into site folders for batch processing."""

import os
import shutil

import pandas as pd
import ruamel.yaml

# file locations
template_base = ".\\template\\"
destination_base = ".\\output_dump\\"
site_config = pd.read_csv("batch_config.csv")
annalist = "analyst_name"
dsn_name = "batch_dsn.dsn"
batch_name = "batch_runner.bat"

run_files = []
dsn_file_list = []

# for each site
for site_index in site_config.index:
    # find base files
    base_files_to_copy = [
        os.path.join(template_base, f)
        for f in os.listdir(template_base)
        if os.path.isfile(os.path.join(template_base, f))
    ]
    # for each measurement at the site
    for measurement in site_config.loc[site_index].list_of_measurements.split(";"):
        # find measurement specific files
        files_to_copy = base_files_to_copy + [
            os.path.join(template_base, measurement, f)
            for f in os.listdir(os.path.join(template_base, measurement))
            if os.path.isfile(os.path.join(template_base, measurement, f))
        ]

        site_destination = os.path.join(
            destination_base,
            measurement,
            site_config.loc[site_index].site_name,
            str(site_config.loc[site_index].batch_no),
        )
        # make sure it exists
        os.makedirs(site_destination, exist_ok=True)
        # copy files over
        for file in files_to_copy:
            shutil.copy2(file, site_destination)

        for file in os.listdir(site_destination):
            file = os.path.join(site_destination, file)
            ext = os.path.splitext(file)[-1].lower()

            if ext in [".hts", ".accdb"]:
                # rename files
                path, file_suffix = os.path.split(file)
                new_file_name = (
                    str(site_config.loc[site_index].batch_no)
                    + "_"
                    + str(site_config.loc[site_index].site_code)
                    + "_"
                    + file_suffix
                )
                os.rename(file, os.path.join(path, new_file_name))

            if ext in [".yaml"]:
                # add in relevant info

                yaml = ruamel.yaml.YAML()
                with open(file) as fp:
                    data = yaml.load(fp)
                    data["site"] = site_config.loc[site_index].site_name
                    data["from_date"] = site_config.loc[site_index].from_date
                    data["to_date"] = site_config.loc[site_index].to_date
                    data["frequency"] = site_config.loc[site_index].frequency
                    data["analyst_name"] = annalist
                    data["standard_measurement_name"] = measurement
                    dsn_file_list.append(
                        os.path.join(site_destination, data["export_file_name"])
                    )

                with open(file, "w") as fp:
                    yaml.dump(data, fp)

            if ext in [".py"]:
                # prep for running
                run_files += [file]


def remove_prefix_dots(string):
    """Removes any "."s at the start of a string."""
    if len(string) == 0:
        return string
    elif string[0] == ".":
        return remove_prefix_dots(string[1:])
    else:
        return string


def make_dsn(file_list, file_path):
    """Makes the hilltop dsn."""
    with open(file_path, "w") as dsn:
        dsn.write("[Hilltop]\n")
        dsn.write("Style=Merge\n")
        for index, file_name in enumerate(file_list):
            dsn.write(f'File{index + 1}="{os.path.abspath(file_name)}"\n')


def make_batch(file_list, file_path):
    """Makes run script."""
    with open(file_path, "w") as runner:
        for file_name in file_list:
            runner.write(f'pushd "{os.path.abspath(os.path.split(file_name)[0])}"\n')
            runner.write("dir\n")
            runner.write(f'start python ".\\{os.path.split(file_name)[1]}"\n')


make_dsn(dsn_file_list, os.path.join(destination_base, dsn_name))
make_batch(run_files, os.path.join(destination_base, batch_name))

"""
base_dir = os.getcwd()
# run the scripts
for file in run_files:
    print(file)
    os.chdir(os.path.split(file)[0])
    importlib.import_module(
        remove_prefix_dots(os.path.splitext(file)[0].replace("\\", "."))
    )
    os.chdir(base_dir)
"""
