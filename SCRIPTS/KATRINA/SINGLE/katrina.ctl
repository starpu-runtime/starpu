dset ^katrinaout
dtype netcdf
undef -888
TITLE WRF Output Grid: Time, bottom_top, south_north, west_east
pdef 45 70 LCCR 25.0 -85.0 24.5 23.5 0 0 -85.0 30000 30000
xdef 44 linear 265.02 0.4536
ydef 69 linear  16.43 0.2402
zdef  27 linear 1 1
tdef   2000 linear 28aug2005 30mn 
vars 9 
P=>p      25  t,z,y,x  Pressure
T=>t      25  t,z,y,x  perturbation potential temperature (theta-t0)
U=>u      25  t,z,y,x  Wind speed U
V=>v      25  t,z,y,x  Wind speed V
U10=>u10   0  t,y,x    Wind at 10m U 
V10=>v10   0  t,y,x    Wind at 10m V 
HGT=>hgt   0  t,y,x    Terrain height 
T2=>t2     0  t,y,x    Temperature at 2m 
LANDMASK=>l     0  t,y,x    landmask 
endvars
