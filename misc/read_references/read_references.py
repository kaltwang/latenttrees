import fileinput
import re
from pyparsing import nestedExpr

def print_list(list, prefix=""):
    if isinstance(list, str):
        print(prefix + ":" + list)
    else:
        for num, elem in enumerate(list):
            print_list(elem, prefix + "-{}".format(num))
            
def connect_list(list, output, depth=()):
    if isinstance(list, str):
        output.append((depth, list))
    else: 
        for num, elem in enumerate(list):
            connect_list(elem, output, depth=depth+(num,))
            
def clean_list(list):
    if isinstance(list, str):
        return list
    elif len(list) > 1:
        for item in list:
            item = clean_list(item)
        return list
    elif len(list) == 1:
        return list[0]
    else:
        return None
        
def concat_connected_tuples(connected):
    depth_last = (-1,)
    concat = []
    for depth, s in connected:
        if depth[:-1] == depth_last[:-1]:
            concat[-1].append(s)
        else:
            concat.append([s])
        depth_last = depth
    return concat
 
def check_elem(elem):
    if elem[0] == '\\item':
        return (True, False)
    elif elem[0] == '\\selectlanguage':
        return (True, True)
    elif elem[-1] == '\\textcolor':
        return (True, True)
    elif elem[-1] == '\\foreignlanguage':
        return(True, True)
    elif elem[0] == '\\rmfamily\\color':
        return (True, True)    
      
    # return skip_this, skip_next
    return (False, False)
 
def process_concat(concat):
    result = []
    skip_next = False
    for elem in concat:
        if not skip_next:
            skip_this, skip_next = check_elem(elem)
            if not skip_this:
                result.append(elem)
        else:
            skip_next = False
    return result
    
def divide_in_authorsyeartitle_and_joutnalpages(processed):
    divider = -1
    for num, elem in enumerate(processed):
        if (len(elem) == 1) and (elem[0] == '\\textit'):
            divider = num
            break
    
    if divider == -1:
        for num, elem in enumerate(processed):
            if (len(elem) == 1) and (elem[0] == '\\textit'):
                divider = num
                break
    
    
    if divider != -1:
        part1 = processed[:divider]
        part2 = processed[(divider+1):]
    else:
        for num, elem in enumerate(processed):
            if elem[-1] == '\\textit':
                divider = num
                break
        if divider != -1:
            part1 = processed[:(divider+1)]
            part2 = processed[(divider+1):]
            del part1[-1][-1] # last element is the \\textit
        else:               
            part1 = processed
            part2 = []
    
    part1 = concatenate_lists(part1)
    part2 = concatenate_lists(part2)
    return part1, part2
    
def concatenate_lists(list):
    if len(list) <= 0:
        return list
        
    result = list[0]
    for elem in list[1:]:
        result += elem
    return result

regex_pages = re.compile(r"(\d+)\-{1,2}(\d+)")
lambda_pages = lambda x: (int(x[0]), int(x[1]))

regex_volumenumber = re.compile(r"(\d+)\((\d+)\)")
lambda_volumenumber = lambda x: (int(x[0]), int(x[1]))

regex_volume = re.compile(r"(\d+)")
lambda_volume = lambda x: int(x)

regex_year = re.compile(r"(\d\d\d\d)")
lambda_year = lambda x: int(x)
    
def extract_expression(part2, regex, lmbd):
    pages = None
    id = None
    for i, elem in enumerate(part2):
        found = regex.findall(elem)
        if len(found) > 0:
            pages = lmbd(found[0])
            id = i
    return pages, id
    
def clean_names(list, ids_to_delete):
    def condition(s):
        if s in {'.', ',', '.,'}:
            return False
        return True
    
    ids_del_set = union_ids(ids_to_delete)    
    for id in reversed(sorted(ids_del_set)):
        if id >= len(list):
            print(ids_to_delete)
            print(list)
            assert False
        list[id] = ''
        
    for i in range(len(list)):
        if not condition(list[i]):
            list[i] = ''
    return list
    
def union_ids(ids_list):
    ids_set = set()
    for x in ids_list:
        if x is not None:
            ids_set.add(x)
    return ids_set
    
def join_string_list(list):
    result = ""
    if len(list) > 0:
        result = list[0]
        for s in list[1:]:
            result += " " + s
    return result
    
def clean_string(string):
    changed = False
    string = string.replace("\ ", " ")
    string = string.replace("  ", " ")
    string = string.replace("\\textbf", "")
    string = string.replace("\\rmfamily", "")
    if string.endswith("\\"):
        string = string[:-1]
        changed = True
    if string.endswith(" "):
        string = string[:-1]
        changed = True
    if string.endswith(","):
        string = string[:-1]
        changed = True
    if string.endswith("."):
        string = string[:-1]
        changed = True
    if string.startswith(" "):
        string = string[1:]
        changed = True
    if changed:
        string = clean_string(string)
    return string
    
def process_author(string):
    def is_surename(s):
        return len(s) > 2
        
    list = re.split('\W+', string)
    list[:] = [x for x in list if (x != 'and') and (x != 'de')]
    if len(list) == 0:
        return ""
    # print(list)
    surenames = []
    given_names = []
    curr_given_names = ""
    for i, name in enumerate(list):
        curr_surename = is_surename(name)
        if curr_surename:
            surenames.append(name)
            if i > 0:
                given_names.append(curr_given_names)
                curr_given_names = ""
        else:
            curr_given_names += " " + name
        last_surename = curr_surename
    if is_surename(list[0]):  
        given_names.append(curr_given_names)
        
    # print(surenames)
    # print(given_names)
    assert len(surenames) == len(given_names)
    
    result = ""
    for i in range(len(surenames)):
        result += surenames[i]
        if len(given_names[i]) > 0:
            result += ',' + given_names[i]
            
        if i < (len(surenames)-1):
            result += " and "
    #print(result)
    return result
    


reference_str = []
for line in fileinput.input():
    if line.startswith("\item"):
        reference_str.append(line)
    else:
        reference_str[-1] += line
  
DELIM_STR = "\n---------------------"  
for ref_num, ref in enumerate(reference_str):
    # print(ref)
    str_nested = "{" + ref + "}"
    nested = nestedExpr('{','}').parseString(str_nested).asList()
    # clean = clean_list(nested)
    
    connected = []
    connect_list(nested, connected)
    concat = concat_connected_tuples(connected)
    processed = process_concat(concat)
    part1, part2 = divide_in_authorsyeartitle_and_joutnalpages(processed)
    
    # if len(part2) > 0:
    # get page numbers
    if len(part2) > 0:
        part_pages = part2
    else:
        part_pages = part1
        
    pages, id_pages = extract_expression(part_pages, regex_pages, lambda_pages)
    volumenumber, id_volumenumber = extract_expression(part_pages, regex_volumenumber, lambda_volumenumber)
    
    
    if (volumenumber is None) and (pages is not None):
        volume, id_volume = extract_expression([part_pages[id_pages-1]], regex_volume, lambda_volume)
        if volume is not None:
            id_volume = id_pages-1
    else:
        volume = None
        id_volume = None
    
    ids_del2 = [id_volume, id_volumenumber, id_pages]
    if len(part2) > 0:
        part2 = clean_names(part2, ids_del2)
    else:
        part1 = clean_names(part1, ids_del2)
        
    
    
    year, id_year = extract_expression(part1, regex_year, lambda_year)
    if year is None:
        year = ""
    
    if len(part2) > 0:
        journal = join_string_list(part2)
    else:
        journal = ""
    
    if id_year is not None:
        author = join_string_list(part1[:id_year])
        title = join_string_list(part1[(id_year+1):])
        # print(title)
    else:
        title = join_string_list(part1)
        author = ""
        
    author = clean_string(author)
    title = clean_string(title)
    journal = clean_string(journal)
    
    author = process_author(author)
    
    print('@article{{zakia_ref{},'.format(ref_num+1))
    print('author = {' + author + '},')
    print('title = {' + title + '},')
    
    if pages is not None:
        print('pages = {{{}--{}}},'.format(pages[0], pages[1]))
    if volumenumber is not None:
        print('volume = {{{}}},\nnumber = {{{}}},'.format(volumenumber[0], volumenumber[1]))
    if volume is not None:
        print('volume = {{{}}},'.format(volume))
    print('journal = {' + journal + '},')
    print('year = {{{}}}'.format(year))
    # print("part1:")
    # print(part1)
    # print("part2:")
    # print(part2)
    # print(DELIM_STR)
    
    print('}\n')
# print("Total: {} references.".format(len(reference_str)))