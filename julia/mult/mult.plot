set output "comparison.pdf"
set term pdf
plot "julia_native.dat" w l,"cstarpu.dat" w l,"julia_generatedc.dat" w l,"julia_calllib.dat" w l
quit
