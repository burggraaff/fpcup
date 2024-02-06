"""
Names of various land types and crops in Dutch (following the BRP) and English.
"""
brp_categories_NL2EN = {'Grasland': "grassland",
                        'Bouwland': "cropland",
                        'Natuurterrein': "nature reserve",
                        'Braakland': "fallow land",
                        'Landschapselement': "landscape element",
                        'Overige': "other",}

brp_categories_colours = {"grassland": "#1b9e77",
                          "cropland": "#d95f02",
                          "nature reserve": "#ddaa33",
                          "fallow land": "#aa3377",
                          "landscape element": "#228833",
                          "other": "#bbbbbb"}

barley = {'Gerst, winter-': "barley (winter)",
          'Gerst, zomer-': "barley (spring)",}

maize = {'Maïs, snij-': "maize (green)",
         'Mais, snij-': "maize (green)",
         'Maïs, korrel-': "maize (grain)",
         'Mais, korrel-': "maize (grain)",
         'Maïs, corncob mix': "maize (mix)",
         'Mais, corncob mix': "maize (mix)",
         'Maiskolvesilage': "maize (silage)",
         'Maïs, suiker-': "maize (sweet)",
         'Mais, suiker-': "maize (sweet)",
         'Maïs, energie-': "maize (energy)",
         'Mais, energie-': "maize (energy)",}

oat = {'Haver': "oat",
       'Naakte haver': "oat (hulless)",
       'Japanse haver': "oat (black)",}

rye = {'Rogge (geen snijrogge)': "rye",
       'Snijrogge': "rye (green)",}

sorghum = {'Soedangras/Sorghum': "sorghum",}

soy = {'Sojabonen': "soy beans",}

spelt = {'Spelt': "spelt"}

wheat = {'Tarwe, winter-': "wheat (winter)",
         'Tarwe, zomer-': "wheat (spring)",}

brp_crops_NL2EN = {**barley, **maize, **oat, **rye, **sorghum, **soy, **spelt, **wheat}

brp_crops_colours = {"barley": "#1b9e77",
                     "maize": "#e6ab02",
                     "wheat": "#d95f02",
                     "oat": "#7570b3",
                     "rye": "#66a61e",
                     "sorghum": "#e7298a",
                     "soy": "#a6761d",
                     "spelt": "#666666"}
