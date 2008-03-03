'open katrina'

'enable print toto'

i=1
j=1000

'set dbuff on'

while (i <= 48)
 'set t 'i
 'set grads off'
 'set gxout shaded'
 'set clevs 0 20 40 60 80 100 120 140 160 180 200 220 240 260 280'
 'set mpdset hires'
 'd sqrt(u10*u10+v10*v10)*6'
 'run cbarn'
 'set gxout vector'
 'set ccolor 0'
 'd u10*1.944;v10*1.944'
 'draw title WIND SPEED  #' i ''
 'print'
 'printim toto'j'.gif gif'
 'swap'
 'clear'
  
 i=i+1
 j=j+1
endwhile
