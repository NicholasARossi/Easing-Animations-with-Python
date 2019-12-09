@Echo Off
@For /F "tokens=1,2,3,4 delims=:,. " %%A in ('echo %time%') do @(
Set Hour=%%A
Set Min=%%B
Set Sec=%%C
Set mSec=%%D
Set All=%%A%%B%%C%%D
)
ren map.shp map.sh1
ren map.dbf map.db1
@For %%a in ("*.shp") do copy %%a "map.shp"
@For %%a in ("*.dbf") do copy %%a "map.dbf"
ren map.sh1 map_original_%All%.shp
ren map.db1 map_original_%All%.dbf
