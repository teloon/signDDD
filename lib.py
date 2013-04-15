pad_bin_code = lambda code_str,code_width:(code_width-len(bin(code_str))+2)*"0"+bin(code_str)[2:]
