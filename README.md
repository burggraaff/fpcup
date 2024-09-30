# Consistent retrieval of crop yields using a data assimilation platform
**Work in progress**

This repository contains the code for a project run by the Institute of Environmental Sciences (CML) at Leiden University, subcontracted from the Netherlands Space Office (NSO) within the [FPCUP framework](https://www.copernicus-user-uptake.eu/).
The aim of the project was to develop a data product improving the retrieval and prediction of crop yields by assimilating existing models (WOFOST) and satellite data (Copernicus).
Please note that it is currently under very active development and not yet ready to be used by others.

### Installation
The module is most easily installed using `pip`.
This requires first cloning the repository, then going into it and running `pip install .`.

Some data will need to be downloaded separately.
From the [PCSE tutorial notebooks](https://github.com/ajwdewit/pcse_notebooks), download the soil data (`data/soil/ec*.soil`) and move these into [data/soil](data/soil).
From [PDOK](https://service.pdok.nl/rvo/brpgewaspercelen/atom/v1_0/basisregistratie_gewaspercelen_brp.xml), download the original BRP files and move these into [data/brp](data/brp); these are then processed using the [process_brp.py script](process_brp.py).

The provincial basemaps are included in the repository but may not load properly when importing into a different working directory.
This can be solved temporarily by working in the `fpcup` directory or copying the files over to your working directory.
A permanent fix would be to package the data into the module.
To generate the basemaps, download data from [PDOK](https://service.pdok.nl/brt/topnl/atom/top10nl.xml) and process them using the [generate_basemaps.py script](generate_basemaps.py).
