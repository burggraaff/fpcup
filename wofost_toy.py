"""
Playing around with the PCSE implementation of WOFOST.
"""
import numpy as np
from matplotlib import pyplot as plt, dates as mdates
import pandas as pd

import pcse
# from progressbar import printProgressBar

# Initialise model
wofost_object = pcse.start_wofost(grid=31031, crop=1, year=2000, mode="wlp")

# Run model and get output
wofost_object.run_till_terminate()
output = wofost_object.get_output()
df = pd.DataFrame(output)

# Show output
summary_output = wofost_object.get_summary_output()
msg = "Reached maturity at {DOM} with total biomass {TAGP} kg/ha and a yield of {TWSO} kg/ha."
print(msg.format(**summary_output[0]))

fig, ax1 = plt.subplots()
ax1.plot(df["day"], df["TWSO"], color="black")
ax1.plot(df["day"], df["TAGP"], color="red")
ax1.set_title("WOFOST growth curve - toy example \nWater-limited winter wheat")
ax1.set_xlabel("Time")
ax1.set_ylabel("Biomass, Yield [kg/ha]")
ax1.set_ylim(ymin=0)
# ax1.set_xlim(df["day"].min(), df["day"].max())

ax2 = ax1.twinx()
ax2.plot(df["day"], df["LAI"], color="blue")
ax2.set_ylabel("Leaf area index (LAI) [m²/m²]", color="blue")
ax2.tick_params(axis="y", colors="blue")
ax2.set_ylim(ymin=0)

plt.savefig("results/wofost_toy_wlp.pdf")
plt.show()
plt.close()
