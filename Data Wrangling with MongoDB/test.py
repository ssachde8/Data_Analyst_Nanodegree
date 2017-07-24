OSMFILE = 'phoenix_osm_dataset/phoenix_osm_dataset'

import audit_file
other_city = audit_file.audit_city(OSMFILE)
print(len(other_city))