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

sorghum = {'Soedangras/Sorghum': "sorghum",}

soy = {'Sojabonen': "soy",}

wheat = {'Tarwe, winter-': "wheat (winter)",
         'Tarwe, zomer-': "wheat (spring)",}

brp_crops_NL2EN = {**barley, **maize, **sorghum, **soy, **wheat}

brp_crops_colours = {"barley": "#1b9e77",
                     "maize": "#e6ab02",
                     "sorghum": "#e7298a",
                     "soy": "#a6761d",
                     "wheat": "#d95f02",}
