import os
import pandas as pd
import numpy as np
from datetime import datetime
import xarray as xr


def clean_dataframe(df):
    # Remove useless columns
    df = df[[col for col in df.columns if ".1" not in str(col)]]
    # Replace characters with nan
    df = df.replace("x", np.nan).replace(" ", np.nan).replace(":", np.nan)
    df = df.replace("u", np.nan).replace("<3", np.nan).replace("<NA>", np.nan)
    # Convert to float
    df = df.convert_dtypes(float, convert_floating=False, convert_integer=True)

    return df


class ONS:

    def __init__(self) -> None:
        # Create folders
        for folder in ["data", "graph"]:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def get_list_datasets(self):
        list_file = os.listdir("data/")
        list_file = [file for file in list_file if ".xlsx" in file and not "~" in file]
        self.list_file = sorted(list_file)
        return self.list_file

    def get_dataset(self, file):
        version = int(file.split("_v")[1].split(".")[0])
        if version < 4:
            dict_file_info = {
                0: {
                    "Table": [4, 5],
                    "skipfooter": 11,
                    "header": [3, 4],
                    "index_col": 0,
                },
                1: {
                    "Table": [1, 2],
                    "skipfooter": 13,
                    "header": [3, 4],
                    "index_col": 0,
                },
                2: {
                    "Table": [5, 6, 7],
                    "skipfooter": 13,
                    "header": [3],
                    "index_col": None,
                },
                3: {
                    "Table": [5, 6, 7],
                    "skipfooter": 15,
                    "header": [3],
                    "index_col": None,
                },
            }
            table_number = dict_file_info[version]["Table"]
            skipfooter = dict_file_info[version]["skipfooter"]
            header = dict_file_info[version]["header"]
            index_col = dict_file_info[version]["index_col"]
            list_df = []
            list_df_name = [
                "Deaths involving COVID-19",
                "Non-COVID-19 deaths",
                "All causes",
            ]
            for i, table in enumerate(table_number):
                df = pd.read_excel(
                    f"data/{file}",
                    sheet_name=f"Table {table}",
                    header=header,
                    skipfooter=skipfooter,
                    index_col=index_col,
                )
                df = clean_dataframe(df)
                # display(df.tail())
                if "Age-group" and "Vaccination status" in df.columns:
                    df = df.pivot_table(
                        index="Month", columns=["Age-group", "Vaccination status"]
                    )
                df = pd.concat([df], axis=1, keys=[list_df_name[i]])
                list_df.append(df)
            df = pd.concat(list_df, axis=1)
            # Dealing with data as an index
            match version:
                case 0 | 1:
                    df = df.drop("Week number", axis=1, level=2)
                    df.columns.names = ["death", "vax_status", "variable"]
                    df.index.name = "date"
                case 2 | 3:
                    df.index = [
                        datetime.strptime(dt + " 2021", "%B %Y") for dt in df.index
                    ]
                    df.index.name = "date"
                    df.columns.names = ["death", "variable", "age_group", "vax_status"]
            df = df.astype(float)
            df = df.rename(
                columns={
                    "Rate per 100,000 population": "ASMR",
                    "Age-standardised mortality rate / 100,000 person-years": "ASMR",
                    "Age-standardised mortality rate per 100,000": "ASMR",
                    "Age-standardised mortality rate per 100,000 person-years": "ASMR",
                }
            )
            df = df.rename(
                columns={
                    "Deaths 21 days or more after first dose": "First dose, at least 21 days ago",
                    "Deaths within 21 days of first dose": "First dose, less than 21 days ago",
                    "21 days or more after first dose": "First dose, at least 21 days ago",
                    "21 days or more after second dose": "Second dose, at least 21 days ago",
                    "21 days or more after third dose or booster": "Third dose or booster, at least 21 days ago",
                }
            )
            ds = df.unstack().to_xarray()
        else:
            df = pd.read_excel(
                f"data/{file}", sheet_name="Table 2", header=[3], skipfooter=0
            )
            df["Month"] = df.Month.apply(lambda x: x.strip())
            df["Month"] = pd.to_datetime(df.Month, format="%B").dt.month
            df["day"] = 1
            df["Date"] = pd.to_datetime(df[["Year", "Month", "day"]])
            df = df.drop(["Year", "Month", "day"], axis=1)
            df = clean_dataframe(df)
            df = df.rename(
                columns={
                    "Rate per 100,000 population": "ASMR",
                    "Age-standardised mortality rate / 100,000 person-years": "ASMR",
                    "Age-standardised mortality rate per 100,000": "ASMR",
                    "Age-standardised mortality rate per 100,000 person-years": "ASMR",
                    'Count of deaths':'Number of deaths'
                }
            )
            ds = (
                df.set_index(
                    ["Cause of Death", "Age group", "Vaccination status", "Date"]
                )
                .stack()
                .to_xarray()
            )
            ds = ds.rename(
                {
                    "Cause of Death": "death",
                    "Age group": "age_group",
                    "Vaccination status": "vax_status",
                    "Date": "date",
                    "level_4": "variable",
                }
            )
        ds = ds.astype(float)
        return ds

    def get_full_dataset(self, force=False):
        filepath = "data/ONS_dataset.nc"
        if not os.path.exists(filepath) or force:
            list_ds = []
            for file in self.list_file:
                print(file)
                ds = self.get_dataset(file)
                list_ds.append(ds)

            list_version = [file.split("_")[1].split(".")[0] for file in self.list_file]
            ds = xr.concat(
                list_ds,
                dim="version",
            )
            ds = ds.assign_coords(coords={"version": list_version})
            ds.to_netcdf(filepath)
        else:
            ds = xr.open_dataarray(filepath)
        return ds

    def group_vax_status(self, ds, binary=False):
        list_vaccin_status = ds.vax_status.values
        if binary:
            dict_vac_groups = {
                "Unvaccinated": ["Unvaccinated"],
                "Vaccinated 1+": [
                    col for col in list_vaccin_status if col != "Unvaccinated"
                ],
            }
        else:
            dict_vac_groups = {
                "Unvaccinated": ["Unvaccinated"],
                "Vaccinated 1 dose": [
                    col for col in list_vaccin_status if "first" in col.lower()
                ],
                "Vaccinated 2 doses": [
                    col for col in list_vaccin_status if "second" in col.lower()
                ],
                "Vaccinated 3 doses": [
                    col for col in list_vaccin_status if "third" in col.lower()
                ],
                "Vaccinated 4 doses": [
                    col for col in list_vaccin_status if "fourth" in col.lower()
                ],
            }
        list_vac_groups = [key for key in dict_vac_groups]
        ds = ds.rename({"vax_status": "Sub_vax_status"})
        ds = xr.concat(
            [
                ds.sel(Sub_vax_status=dict_vac_groups[vac_groups])
                for vac_groups in list_vac_groups
            ],
            dim="vax_status",
        )
        ds = ds.assign_coords(coords={"vax_status": list_vac_groups})
        ds = ds.sum(dim="Sub_vax_status")
        return ds


