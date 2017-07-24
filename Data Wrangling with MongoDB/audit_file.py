# -*- coding: utf-8 -*-
"""
Your task in this exercise has two steps:

- audit the OSMFILE and change the variable 'mapping' to reflect the changes needed to fix
    the unexpected street types to the appropriate ones in the expected list.
    You have to add mappings only for the actual problems you find in this OSMFILE,
    not a generalized solution, since that may and will depend on the particular area you are auditing.
- write the update_name function, to actually fix the street name.
    The function takes a string with street name as an argument and should return the fixed name
    We have provided a simple test so that you see what exactly is expected
"""
import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

OSMFILE = "data/phoenix_osm_dataset"
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

expected = ["Avenue", "Boulevard", "Commons", "Court", "Drive", "Lane", "Parkway", 
                         "Place", "Road", "Square", "Street", "Trail"]

# UPDATE THIS VARIABLE
mapping = {'Ave'  : 'Avenue',
           'Blvd' : 'Boulevard',
           'Dr'   : 'Drive',
           'Ln'   : 'Lane',
           'Pkwy' : 'Parkway',
           'Rd'   : 'Road',
           'Rd.'   : 'Road',
           'St'   : 'Street',
           'street' :"Street",
           'Ct'   : "Court",
           'Cir'  : "Circle",
           'Cr'   : "Court",
           'ave'  : 'Avenue',
           'Hwg'  : 'Highway',
           'Hwy'  : 'Highway',
           'Sq'   : "Square"
		   }


def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)


def audit_zipcode(invalid_zip, zipcode):
	# check if first two characters are digits
	temp_digit = zipcode[:2]
	if not temp_digit.isdigit():
		invalid_zip[temp_digit].add(zipcode)
	elif temp_digit != 85:
		invalid_zip[temp_digit].add(zipcode)

		
def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")

	
def is_zipcode(elem):
    return (elem.attrib['k'] == "addr:postcode")

	
def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])

    return street_types


	
def audit_zip(osmfile):
	osm_file = open(osmfile, "r")
	invalid_zip = defaultdict(set)
	for event, elem in ET.iterparse(osm_file, events=("start",)):
		if elem.tag == "node" or elem.tag == "way":
			for tag in elem.iter("tag"):
				if is_zipcode(tag):
					audit_zipcode(invalid_zip, tag.attrib['v'])
	return invalid_zip
	

def update_name(name, mapping):
    after = []
    # Split name string to test each part of the name;
    # Replacements may come anywhere in the name.
    for part in name.split(" "):
        # Check each part of the name against the keys in the correction dict
        if part in mapping.keys():
            # If exists in dict, overwrite that part of the name with the dict value for it.
            part = mapping[part]
        # Assemble each corrected piece of the name back together.
        after.append(part)
    # Return all pieces of the name as a string joined by a space.
    return " ".join(after)

    #     for w in mapping.keys():
    #         if w in name:
    #             if flag:
    #                 continue
    #             # Replace abbrev. name in string with full name value from the mapping dict.
    #             name = name.replace(w, mapping[w], 1)
    #             # If St., flag to not check again in this string looking for St since new 'Street' will contain St
    #             # re.compile() might be better
    #             if w == "St.":
    #                 flag = True

    return name

	
def update_zip(zipcode):
    zipcode.strip()
    if "AZ" in zipcode:
        zipcode = zipcode.replace("AZ", "")
    if len(zipcode) < 5:
        zipcode = None
    if zipcode is not None and len(zipcode) == 5:
        if zipcode < 85001 or zipcode > '86556':
            zipcode = None
    return zipcode

	
def test():
    st_types = audit(OSMFILE)
    assert len(st_types) == 3
    pprint.pprint(dict(st_types))

    for st_type, ways in st_types.iteritems():
        for name in ways:
            better_name = update_name(name, mapping)
            print(name, "=>", better_name)
            if name == "West Lexington St.":
                assert better_name == "West Lexington Street"
            if name == "Baldwin Rd.":
                assert better_name == "Baldwin Road"


if __name__ == '__main__':
    test()