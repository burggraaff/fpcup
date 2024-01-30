"""
Names of various land types and crops in Dutch (following the BRP) and English.
"""
brp_categories_NL2EN = {'Grasland': "Grassland",
                        'Bouwland': "Arable land",
                        'Natuurterrein': "Nature reserve",
                        'Braakland': "Fallow land",
                        'Overige': "Other",}

brp_categories_colours = {"Grassland": "#1b9e77",
                          "Arable land": "#d95f02",
                          "Nature reserve": "#ddaa33",
                          "Fallow land": "#aa3377",
                          "Other": "#bbbbbb"}

barley = {'Gerst, winter-': "Barley (winter)",
          'Gerst, zomer-': "Barley (spring)",}

maize = {'Maïs, snij-': "Maize (green)",
         'Maïs, korrel-': "Maize (grain)",
         'Maïs, corncob mix': "Maize (mix)",
         'Maiskolvesilage': "Maize (silage)",
         'Maïs, suiker-': "Maize (sweet)",
         'Maïs, energie-': "Maize (energy)"}

oat = {'Haver': "Oat",
       'Naakte haver': "Oat (hulless)"}

rye = {'Rogge (geen snijrogge)': "Rye (green)",}

sorghum = {'Soedangras/Sorghum': "Sorghum",}

soy = {'Sojabonen': "Soy beans",}

spelt = {'Spelt': "Spelt"}

wheat = {'Tarwe, winter-': "Wheat (winter)",
         'Tarwe, zomer-': "Wheat (spring)",}

brp_crops_NL2EN = {**barley, **maize, **oat, **rye, **sorghum, **soy, **spelt, **wheat}

brp_crops_colours = {"Barley": "#1b9e77",
                     "Maize": "#e6ab02",
                     "Wheat": "#d95f02",
                     "Oat": "#7570b3",
                     "Rye": "#66a61e",
                     "Sorghum": "#e7298a",
                     "Soy": "#a6761d",
                     "Spelt": "#666666"}
