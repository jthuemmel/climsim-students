# Glossary

This glossary explains the terms of atmospheric physics in this repository, and the names of the variables.
You do not need more physics than this to do the work.

Each entry is short. If an entry uses a term in *italics*, that term has its own entry.

## 1. The simulation

**ClimSim.**
A public data set for machine learning (Yu et al. 2023).
It contains the results of a climate simulation with the model E3SM-MMF.

**Climate model.**
A computer program that calculates the weather of the full Earth for many years.
It divides the atmosphere into *columns* and calculates the change of each column from one *timestep* to the next.

**E3SM-MMF.**
The climate model that made the ClimSim data.
In this model, each *column* contains a second, small model with a much finer grid.
The small model calculates the clouds and the rain in the column. A radiation calculation follows it.
The two calculations together use most of the computer time of the simulation.

**The task.**
For each *column* and each *timestep*, the data contains the state of the column before these two calculations (the inputs) and their effect (the targets).
A neural network learns to predict the targets from the inputs.
A network that does this correctly can replace the two calculations, and the climate model becomes much faster.

**Real geography.**
The simulation has the continents, the mountains, and the oceans of the Earth.

**Low resolution.**
The grid of this data has 384 *columns*. One column is approximately 1,100 km wide.

**Timestep.**
The interval between two states of the simulation: 20 minutes, which is 1,200 seconds.
One day has 72 timesteps.

**Calendar.**
Each year of the simulation has 365 days. There are no leap years.
The years have the numbers 1 to 9. They are not years of history.

## 2. The dimensions of the data

**Column (`ncol`).**
A vertical part of the atmosphere above one area of the surface, from the surface to the top of the atmosphere.
The data has 384 columns. `ncol` is the number of a column, from 0 to 383.
The columns do not form a regular grid of latitude and longitude.

**`lat`, `lon`, `area`.**
The latitude and the longitude of the centre of a column, in degrees, and the area of the column.
The unit of `area` is the area on a sphere with a radius of 1. Thus the sum of all areas is 4π.
The columns do not all have the same area. The paper scores use `area` as a weight.

**Level (`level`).**
A horizontal layer of a *column*. Each column has 60 levels.
Level 0 is at the top of the atmosphere, at a height of approximately 65 km.
Level 59 is adjacent to the surface.
In height, the levels are thin near the surface and thick near the top.

**Profile.**
A variable that has one value for each *level* of a *column*, for example the temperature.
A profile gives 60 values for one row of the data.

**Scalar.**
A variable that has one value for a *column*, for example the surface pressure.

**Row.**
One *column* at one time. The model predicts one row independently of all other rows.

## 3. The inputs

| Name | Term | Unit | Values |
| --- | --- | --- | --- |
| `state_t` | Air temperature | K | 60 |
| `state_q0001` | *Specific humidity* | kg/kg | 60 |
| `state_ps` | *Surface pressure* | Pa | 1 |
| `pbuf_SOLIN` | *Insolation* | W/m2 | 1 |
| `pbuf_LHFLX` | Surface *latent heat flux* | W/m2 | 1 |
| `pbuf_SHFLX` | Surface *sensible heat flux* | W/m2 | 1 |

The total is 124 values for one row.

## 4. The targets

| Name | Term | Unit | Values |
| --- | --- | --- | --- |
| `ptend_t` | Heating *tendency* | K/s | 60 |
| `ptend_q0001` | Moistening *tendency* | kg/kg/s | 60 |
| `cam_out_NETSW` | Net *shortwave* *flux* at the surface | W/m2 | 1 |
| `cam_out_FLWDS` | Downward *longwave* flux at the surface | W/m2 | 1 |
| `cam_out_PRECSC` | Snow rate | m/s | 1 |
| `cam_out_PRECC` | Rain rate | m/s | 1 |
| `cam_out_SOLS` | Downward *visible* *direct* solar flux at the surface | W/m2 | 1 |
| `cam_out_SOLL` | Downward *near-infrared* direct solar flux at the surface | W/m2 | 1 |
| `cam_out_SOLSD` | Downward visible *diffuse* solar flux at the surface | W/m2 | 1 |
| `cam_out_SOLLD` | Downward near-infrared diffuse solar flux at the surface | W/m2 | 1 |

The total is 128 values for one row.
The eight `cam_out` targets are values at the surface. They are not negative.

## 5. The additional variables of `python fetch_data.py --full`

| Name | Term | Unit | Values |
| --- | --- | --- | --- |
| `state_q0002` | Cloud liquid *mixing ratio* | kg/kg | 60 |
| `state_q0003` | Cloud ice mixing ratio | kg/kg | 60 |
| `state_u` | *Zonal* wind speed | m/s | 60 |
| `state_v` | *Meridional* wind speed | m/s | 60 |
| `pbuf_ozone` | Ozone *volume mixing ratio* | mol/mol | 60 |
| `pbuf_TAUX`, `pbuf_TAUY` | Zonal and meridional *surface stress* | N/m2 | 1 each |
| `pbuf_COSZRS` | Cosine of the *solar zenith angle* | no unit | 1 |
| `cam_in_ASDIR`, `cam_in_ASDIF` | *Albedo* for *visible* light, *direct* and *diffuse* | no unit | 1 each |
| `cam_in_ALDIR`, `cam_in_ALDIF` | Albedo for *near-infrared* light, direct and diffuse | no unit | 1 each |
| `cam_in_LWUP` | Upward *longwave* *flux* from the surface | W/m2 | 1 |
| `cam_in_ICEFRAC` | The part of the column area that is sea ice | 0 to 1 | 1 |
| `cam_in_LANDFRAC` | The part of the column area that is land | 0 to 1 | 1 |
| `cam_in_OCNFRAC` | The part of the column area that is open ocean | 0 to 1 | 1 |
| `cam_in_SNOWHLAND` | Snow depth on land, as *liquid water equivalent* | m | 1 |
| `cam_in_SNOWHICE` | Snow depth on sea ice | m | 1 |
| `ptend_q0002`, `ptend_q0003` | *Tendency* of cloud liquid and of cloud ice | kg/kg/s | 60 each |
| `ptend_u`, `ptend_v` | Tendency of the zonal and of the meridional wind | m/s2 | 60 each |

The paper names the albedos `ASDIR` and `ASDIF` "shortwave" and the albedos `ALDIR` and `ALDIF` "longwave".
In these four names, "shortwave" is visible sunlight and "longwave" is near-infrared sunlight.

## 6. Terms of physics

**Albedo.**
The part of the sunlight that a surface reflects. The value is between 0 and 1.
Snow has a high albedo. The ocean has a low albedo.

**Diffuse.**
Sunlight that the air or the clouds scatter before it arrives at the surface. It comes from all directions of the sky.

**Direct.**
Sunlight that arrives at the surface in a straight line from the sun.

**Flux.**
A flow of energy through an area, in watts for each square metre (W/m2).

**Insolation.**
The *flux* of sunlight that arrives at the top of the atmosphere above the *column*.
It is zero at night.

**Latent heat flux.**
The *flux* of energy from the surface into the atmosphere as water vapour.
Water that evaporates takes energy from the surface. The energy is released when the vapour becomes cloud water.

**Liquid water equivalent.**
The depth of the water that you get if you melt the snow.

**Longwave.**
The heat radiation that the surface, the air, and the clouds emit.

**Meridional.**
In the direction from south to north. A positive meridional wind blows to the north.

**Mixing ratio.**
The mass of a substance in one kilogram of air, in kg/kg.

**Moistening.**
An increase of the *specific humidity*. A negative moistening *tendency* is a decrease.

**Near-infrared.**
The part of the sunlight with a wavelength of more than 0.7 micrometres. You cannot see it.

**Sensible heat flux.**
The *flux* of energy from the surface into the atmosphere as heat, because the surface and the air have different temperatures.

**Shortwave.**
The radiation of the sun: *visible* and *near-infrared* light together.

**Net shortwave flux at the surface.**
The sunlight that the surface absorbs: the sunlight that arrives, minus the sunlight that the surface reflects.

**Solar zenith angle.**
The angle between the sun and the vertical.
Its cosine is 1 when the sun is directly above the *column*. Its cosine is 0 when the sun is at the horizon.

**Specific humidity.**
The mass of water vapour in one kilogram of air, in kg/kg.

**Surface pressure.**
The pressure of the air at the surface, in pascals. It is the weight of the full *column* of air on one square metre.

**Surface stress.**
The force of friction between the wind and the surface on one square metre, in N/m2.

**Tendency.**
The rate of change of a variable in time.
In this data, a tendency is (the value after the calculation − the value before the calculation) / 1,200 seconds.
The heating tendency is the tendency of the temperature. The moistening tendency is the tendency of the *specific humidity*.

**Visible.**
The part of the sunlight with a wavelength of less than 0.7 micrometres.

**Volume mixing ratio.**
The number of molecules of a gas for each molecule of air, in mol/mol.

**Zonal.**
In the direction from west to east. A positive zonal wind blows to the east.

## 7. The levels and the conversion to W/m2

**`hyai`, `hybi`, `P0`.**
Constants that give the pressure at the 61 boundaries of the 60 *levels*:
pressure = `P0` × `hyai` + *surface pressure* × `hybi`.
`P0` is 100,000 Pa.
Near the top, the pressure of a level is constant. Near the surface, the pressure of a level follows the surface pressure.

**Mass of a level.**
The mass of the air in one *level* above one square metre: the pressure difference between its two boundaries, divided by g.
g is the acceleration of gravity, 9.81 m/s2.

**Conversion to W/m2.**
The ClimSim paper gives all errors in W/m2. Then you can compare the errors of different *levels* and different targets.
- Heating *tendency* × the heat capacity of air × the mass of the level.
- *Moistening* tendency × the heat of evaporation of water × the mass of the level.
- Rain rate and snow rate × the heat of evaporation of water × the density of water.
- The radiation *fluxes* are in W/m2 already.

`score.py` does this conversion.

## 8. The scores

**R2.**
1 − (the sum of the squared errors) / (the sum of the squared differences between the truth and its mean).
An R2 of 1 is a perfect prediction. An R2 of 0 is as good as a prediction of the mean. A bad prediction has a negative R2.

**MAE, MSE.**
The mean of the absolute errors, and the mean of the squared errors.

**Kaggle score.**
The mean R2 of the 128 targets in their physical units. Each *level* of a *profile* is one target.
For each target, the R2 uses all *rows* of the test data together.
The Kaggle competition gave a weight of zero to levels 0 to 11 of the moistening tendency. These 12 targets count as 1.

**Paper scores.**
The MAE, the MSE, and the R2 in W/m2.
The MAE and the MSE use the area of each *column* as a weight.
The R2 is calculated along time in each column. The result is the mean of the columns.
A column where the target is constant in time has no R2.

**CRPS.**
A score for a prediction that is a distribution, not one value. A smaller CRPS is better.
For a prediction that is one value, the CRPS is equal to the MAE.

**PIT histogram.**
A check of a distribution. It counts where the truth is in the predicted distribution.
If the histogram is flat, the predicted distribution has the correct width.
