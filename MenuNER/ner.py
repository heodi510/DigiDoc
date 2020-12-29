import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
import pandas as pd
import re

def preprocess(sent):
    # Preprocess the text by tokenising the words and adding a pos tag to each word
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent


def menu_to_list(lines):
    # Covert every sentence into a list and with every word in the sentence to a seperate tuple with IOB tags
    tokenized_menu = []
    for line in lines:
        new_line = preprocess(line)
        pattern = '''Price: {(<\$><CD>*|<CD>$|^<CD>$)}'''
        cp = nltk.RegexpParser(pattern)
        cs = cp.parse(new_line)
        iob_tagged = tree2conlltags(cs)
        tokenized_menu.append(iob_tagged)
    return tokenized_menu


def tuple_converter(m_list):
    # Convert the tuples in list to list format
    new_m_list = []
    for sub in m_list:
        new_l = []
        for tup in sub:
            try:
                new_l.append(list(tup))
            except:
                continue
        new_m_list.append(new_l)
    return new_m_list


def menu_labeller(unlabelled_menu):
  # Takes the messy list and converts it into readable list with the items and its entity label i.e. Dish or Price 
    menu_table = []
    for sub_lists in unlabelled_menu:
        try:
            dish_name = ""
            for i in range(len(sub_lists)):
                if sub_lists[i][2] == "B-Price":
                    try:
                        menu_table.append(["Price",sub_lists[i][0]+sub_lists[i+1][0]])
                    except:
                        menu_table.append(["Price",sub_lists[i][0]])
                    if dish_name == "":
                        continue
                    else:
                        menu_table.append(["Dish",dish_name.strip()])
                        dish_name = ""
                elif sub_lists[i][2] == "I-Price":
                    continue
                else:
                    dish_name = dish_name + sub_lists[i][0] + ' '
            if dish_name == "":
                continue
            else:
                menu_table.append(["Dish",dish_name.strip()])
        except:
            continue
    return menu_table


def menu_cleaner(labelled_menu):
  # Takes a labelled menu and removes useless text in in each line e.g. blank spaces or just symbols
    remover = []

    for item in range(len(labelled_menu)):
        if labelled_menu[item][0] == "Dish" and len(labelled_menu[item][1]) < 3:
            remover.append(labelled_menu[item])
    for i in remover:
        labelled_menu.remove(i)
    return labelled_menu


def menu_categorizer(labelled_menu, adjuster):
  # Converts cleaned menu list to a dataframe
  # For the adjuster input only -1 or +1 try manually to see which one makes more sense or which one works
    total_menu = []
    for item in range(len(labelled_menu)):
        if labelled_menu[item][0] == "Price":
            total_menu.append([labelled_menu[item][1],labelled_menu[item + adjuster][1]])
    return pd.DataFrame(total_menu,columns=['$',"Dish"])


def menutxt_to_dataframe(menu_text_file, adjuster):
  # Converts text menu file to a dataframe
  # For the adjuster input only -1 or +1 try manually to see which one makes more sense or which one works
    menu_text_file = re.sub('[sS](\d+)', '$\1', menu_text_file)
    para = menu_text_file.split('\n')
    list_with_IOB = menu_to_list(para)
    untupled_list = tuple_converter(list_with_IOB)
    menu_with_label = menu_labeller(untupled_list)
    clean_menu = menu_cleaner(menu_with_label)
    final = menu_categorizer(clean_menu,adjuster)
#     final = final[~final.Dish.str.contains("$")]
    return final

