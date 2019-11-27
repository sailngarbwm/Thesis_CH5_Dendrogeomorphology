from pathlib import Path


class Config(object):
    """
    this object stores allt eh directories we need to work with this data
    will be lazy and just chuck the path into a dictionary for each
    """
    def __init__(self): 
        self.data_dir = Path(r'F:\Brisbane_Pine GIS QGIS_2018')
        self.brispath = Path(r'F:\Brisbane_Pine GIS QGIS_2018\Brisbane GIS data\Brisbane basemap\dbh_shapefiles')
        self.bris_dirs = {'folder':r'F:\Brisbane_Pine GIS QGIS_2018\Brisbane GIS data\Brisbane basemap\dbh_shapefiles',
                        'thalwegs':'bris_thalweg.geojson',
                        'banks':r'bris_banks.geojson',
                        'points':r'Brisbane_waypoints_merge.geojson',
                        'lidar1': 'bris_dem_1.asc',
                        'lidar2': 'bris_dem_2.asc'
        }
        self.kobble_dirs = {'folder':r'F:\Brisbane_Pine GIS QGIS_2018\Brisbane GIS data\Kobble_Creek_2015\kobble shapefiles',
                            'lidar': r'kobble_lidar.asc',
                            'thalwegs':'kobble_thalweg.geojson',
                            'banks':r'kobble_banks.geojson',
                            'points':r'kobble_points.geojson'
                            }

        self.pine_dirs = {'folder':r'F:\Brisbane_Pine GIS QGIS_2018\Brisbane GIS data\Lidar_Pine\Pine shapefiles',
                        'lidar': 'pine_lidar_clip.asc',
                            'thalwegs':'NPR_thalweg.geojson',
                            'banks':r'NPR_banks.geojson',
                            'points':r'NPR_data_points.geojson'
                            }

        self.mary_dirs = {'folder':r'F:\Brisbane_Pine GIS QGIS_2018\Mary River imagery\shapefiles',
                          'lidar1':r'mary_lidar_1.asc',
                          'lidar2':r'mary_lidar_2.asc',
                          'thalwegs':'mr_thalweg.geojson',
                          'banks':r'mr_banklines.geojson',
                          'points':r'mary_waypoints_joined_reach.geojson'

                         }
        
        
