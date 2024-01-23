"""
Names of various vrops in Dutch (following the BRP) and English.
"""
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

brp_dictionary = {**barley, **maize, **oat, **rye, **sorghum, **soy, **spelt, **wheat}
