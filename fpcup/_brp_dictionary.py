"""
Names of various land types and crops in Dutch (following the BRP) and English.
"""
brp_categories_NL2EN = {'Grasland': "Grassland",
                        'Bouwland': "Cropland",
                        'Natuurterrein': "Nature reserve",
                        'Braakland': "Fallow land",
                        'Landschapselement': "Landscape element",
                        'Overige': "Other",}

brp_categories_colours = {"Grassland": "#1b9e77",
                          "Cropland": "#d95f02",
                          "Nature reserve": "#ddaa33",
                          "Fallow land": "#aa3377",
                          "Landscape element": "#228833",
                          "Other": "#bbbbbb"}

barley = {'Gerst, winter-': "Barley (winter)",
          'Gerst, zomer-': "Barley (spring)",}

maize = {'Maïs, snij-': "Maize (green)",
         'Mais, snij-': "Maize (green)",
         'Maïs, korrel-': "Maize (grain)",
         'Mais, korrel-': "Maize (grain)",
         'Maïs, corncob mix': "Maize (mix)",
         'Mais, corncob mix': "Maize (mix)",
         'Maiskolvesilage': "Maize (silage)",
         'Maïs, suiker-': "Maize (sweet)",
         'Mais, suiker-': "Maize (sweet)",
         'Maïs, energie-': "Maize (energy)",
         'Mais, energie-': "Maize (energy)",}

oat = {'Haver': "Oat",
       'Naakte haver': "Oat (hulless)",
       'Japanse haver': "Oat (black)",}

rye = {'Rogge (geen snijrogge)': "Rye",
       'Snijrogge': "Rye (green)",}

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
