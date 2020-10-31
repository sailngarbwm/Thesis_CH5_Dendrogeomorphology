library('raster')
library('sf')

matcher = read.csv('zonal_stats_file_match.csv',stringsAsFactors=FALSE)

head(matcher)



zonalStats = function(name,raster_file, Shapefile){
  print(paste('running', name, Shapefile, raster_file))
  
  outfile = paste0(name, '_merged_buff_V2_RZonal.geojson')
  
  if (file.exists(outfile) == FALSE) { 
    lidar = raster(raster_file)
    
    shapefile = st_read(Shapefile)
    # shapefile
    mean = extract(lidar, shapefile, fun = mean)
    shapefile['_mean']= mean
    
    max = extract(lidar, shapefile, fun = max)
    shapefile['_max']= max
    
    min = extract(lidar, shapefile, fun = min)
    shapefile['_min']= min
    
    stdev = extract(lidar, shapefile, fun = sd)
    shapefile['_stdev']= stdev
    
    outfile = paste0(name, '_merged_buff_V2_RZonal.geojson')
    
    st_write(shapefile, outfile)
  } else { 
    print('already done')
    }
  
  
}

apply(matcher[c('name', 'raster_file', 'Shapefile')],1, FUN = function(x){zonalStats(x[1],x[2], x[3])})


