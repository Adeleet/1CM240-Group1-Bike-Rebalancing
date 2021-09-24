import math

# This function translates coordinates (latitudes, longitudes) to a distance in kilometers
def calcDistance(lat1_, lon1_, lat2_, lon2_):

    R = 6373.0

    lat1 = math.radians(lat1_)
    lon1 = math.radians(lon1_)
    lat2 = math.radians(lat2_)
    lon2 = math.radians(lon2_)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

