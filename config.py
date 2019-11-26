from pathlib import Path


class Config(object):
    """
    this object stores allt eh directories we need to work with this data
    will be lazy and just chuck the path into a dictionary for each
    """
    def __init__(self): 
        self.data_dir = Path(r'F:\Brisbane_Pine GIS QGIS_2018')
        self.bris_dirs = {'lidar': r'F:\Brisbane_Pine GIS QGIS_2018\Brisbane GIS data\UB_3Raster_Mash_Up_V6_DEM_Smoother.tif',
                        'thalwegs':r'F:\Brisbane_Pine GIS QGIS_2018\Brisbane GIS data\Brisbane basemap\dbh_shapefiles\bris_thalweg.geojson',
                        'banks':r'F:\Brisbane_Pine GIS QGIS_2018\Brisbane GIS data\Brisbane basemap\dbh_shapefiles\bris_banks.geojson',
                        'points':r'F:\Brisbane_Pine GIS QGIS_2018\Brisbane GIS data\Brisbane basemap\dbh_shapefiles\Brisbane_waypoints_merge.geojson',
        }
        self.kobble_dirs = {'lidar': r'F:\Brisbane_Pine GIS QGIS_2018\Brisbane GIS data\Lidar_Pine\2014_UpperPine_MinorBasinMosaics\2014_DEM_UpperPine_MinorBasin_mga56.tif',
                            'thalwegs':r'F:\Brisbane_Pine GIS QGIS_2018\Brisbane GIS data\Kobble_Creek_2015\kobble shapefiles\kobble_thalweg.geojson',
                            'banks':r'F:\Brisbane_Pine GIS QGIS_2018\Brisbane GIS data\Kobble_Creek_2015\kobble shapefiles\kobble_banks.geojson',
                            'points':r'F:\Brisbane_Pine GIS QGIS_2018\Brisbane GIS data\Kobble_Creek_2015\kobble shapefiles\kobble_points.geojson'
                            }

        self.pine_dirs = {'lidar': r'F:\Brisbane_Pine GIS QGIS_2018\Brisbane GIS data\Lidar_Pine\2014_UpperPine_MinorBasinMosaics\2014_DEM_UpperPine_MinorBasin_mga56.tif',
                            'thalwegs':r'F:\Brisbane_Pine GIS QGIS_2018\Brisbane GIS data\Lidar_Pine\Pine shapefiles\NPR_thalweg.geojson',
                            'banks':r'F:\Brisbane_Pine GIS QGIS_2018\Brisbane GIS data\Lidar_Pine\Pine shapefiles\NPR_banks.geojson',
                            'points':r'F:F:\Brisbane_Pine GIS QGIS_2018\Brisbane GIS data\Lidar_Pine\Pine shapefiles\NPR_merged_points.geojson'
                            }

        self.mary_dirs = {'lidar':r'F:\Brisbane_Pine GIS QGIS_2018\Mary River imagery\Mary LiDAR\2014_sunshine_coast_merge.2tif.tif',
                          'thalwegs':r'F:\Brisbane_Pine GIS QGIS_2018\Mary River imagery\shapefiles\mr_thalweg.geojson',
                          'banks':r'F:\Brisbane_Pine GIS QGIS_2018\Mary River imagery\shapefiles\mr_banklines.geojson',
                          'points':r'F:\Brisbane_Pine GIS QGIS_2018\Mary River imagery\shapefiles\mary_waypoints_joined_reach.geojson',

                         }
        
        
