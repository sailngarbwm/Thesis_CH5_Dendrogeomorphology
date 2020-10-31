library('SDAR')
library(readxl)

#data("saltarin")

#strat.data =strata(saltarin)

help(plot)

# sdar_test = read_excel('NPR_wtp_stratcol.xlsx')


# sdar_test$rock_type = 'sedimentary'
# strat.test = strata(sdar_test)

#help(plot.strata)

# plot(strat.test, datum = 'top',
#      data.units = 'meters',d.scale= 1000, d.barscale = 10,
#      file.name = 'NPR_wtp')


#help("read_excel")
#plot(strat.data, datum = 'top')
#help(SDAR)

make.strat.col = function(xlsfile, outfile, d.scaler = 1000, d.barscaler=10){
  
  sdar_test = read_excel(xlsfile)
  
  
  # sdar_test$rock_type = 'sedimentary'
  strat.test = strata(sdar_test)
  
  #help(plot.strata)
  
  plot(strat.test, datum = 'top',
       data.units = 'meters',d.scale= d.scaler, d.barscale = d.barscaler,
       file.name = outfile)
  
  
}


#make.strat.col('NPR_wtp_stratcol.xlsx', 'NPR_outt2')

make.strat.col('NPR_US_stratcol.xlsx', 'NPR_out_BCAS3')

make.strat.col('NPR_GPS5_stratcol.xlsx', 'NPR_out_GPS5')


make.strat.col('Mary_gps181_stratcol.xlsx', 'Mary_out_GPS181')

