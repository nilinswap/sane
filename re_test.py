import re
import time
regex_st = r"\[.*?('P.*?').*?('P.*?').*?\]"

comp_pat = re.compile(regex_st)
check_st = "['Pello','pello','Pllo']"
mat_obj = comp_pat.match(check_st)
srch_obj = comp_pat.search(check_st) 
fa_obj = comp_pat.findall( check_st )
"""
if mat_obj:
  print("match, yeah")
else:
  print("match, no")
"""
if srch_obj:
  print("search, yeah")
  print("search group is :", srch_obj.group())
  print("all are :", fa_obj)
else:
  print("search, no")

iter_obj = comp_pat.finditer( check_st )
for mat in iter_obj:
  print(mat.span())
#print("search group is : ", srch_obj.group())
