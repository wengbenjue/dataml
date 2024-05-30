file_path = r'/Users/wengbenjue/Documents/KPMG/河北银行孵化/consumer.csv'
import re
with open(file_path,'r',encoding='utf8') as f:
    lines = f.readlines()
    for line in lines:
        print(line)
        line_array = re.split(',',line)
        print(line_array)
        print(len(line_array))
        
        

