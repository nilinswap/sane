import re
import time
#regex_st = r'([^a-z]R)(oa)(d)'
regex_st = r'road'
comp_pat = re.compile(regex_st,re.I)
check_st = "  199PRoadroad  12/11/1996 "
mat_obj = comp_pat.match(check_st)
srch_obj = comp_pat.search(check_st)
fa_obj = comp_pat.findall( check_st )

#r'[.*/'(P.*/').*/1'

"""st = srch_obj.group(1)
new_regex_st = eval("r'"+st+"'")
new_comp_pat = re.compile(new_regex_st)
#new_srch_obj = new_comp_pat.search(check_st)
print(new_comp_pat.sub('',check_st))
"""
"""
if mat_obj:
  print("match, yeah")
else:
  print("match, no")
"""

def func(match_obj):
	print(match_obj.group(1))
	if match_obj.group(2):
		return match_obj.group(1) + '_' + match_obj.group(2).lower()






print( 'here', re.sub( regex_st, '' , check_st))


if srch_obj:
  print("search, yeah")
  #print("search group is :", srch_obj.group(1))
  print("all are :", fa_obj)
  print("in span :", srch_obj.group())
else:
  print("search, no")

#print("now subsituting '' in place of 0")
#print(comp_pat.sub('',check_st))
#print("search group is : ", srch_obj.group())
