import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("Lobelletal.SOMdata/maize.alldat.nolatlon.csv")

stats_by_county = data.groupby("FIPS")["TRUEYIELD"].agg(["mean", "std"])
