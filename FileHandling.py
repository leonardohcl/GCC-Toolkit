from PIL import Image
from pandas import DataFrame
import os 

def readImage(path):
    """Open an image using PIL"""
    try:
        return Image.open(path)
    except OSError as err:
        print("Couldn't read file. Error: {0}".format(err))

def createFolderContentCsv(folder_path: str, csv_name:str, extension:str=None):
    """Create a csv file with the contents of a folder
        Args:
            folder_path(string): path to folder
            csv_name(string): output csv file name, without extension
            extension(string): filter only files of a kind on the output csv. 
    """
    if os.path.isdir(folder_path):
        output_file_path = os.path.join(folder_path,csv_name)+'.csv'
        if os.path.isfile(output_file_path):
            print("There's already a csv file with given name. name: {}".format(csv_name))
            return

        content_list = os.listdir(folder_path)
        file_list = []
        for content in content_list:
            if os.path.isfile(os.path.join(folder_path, content)):
                file_list.append(content)
        if extension:
            temp_list = []
            for f in file_list:
                if f.endswith(extension):
                    temp_list.append(f)
            file_list = temp_list
        
        df = DataFrame(file_list)
        df.to_csv(output_file_path, header=False, index=False)
    else:
        print("Folder doesn't exist. path: {}".format(folder_path))

def createArffFile(file_name: str, content: list, attrs: list, classes: list, target:str = None):
    """Create a arff file with given data list
        Args:
            file_name(string): name of output file without extension
            content(list): data to write in arff, must be a two-dimensional list. Function assumes all values are real numbers
            attrs(list): list with the attribute names for the content. Classes are not included here, and should be put at the end of the each content entry
            classes(list): list with possible classes of given data
            target(string): path to the target folder of the output file, if none is passed assumes the current directory   
    """
    if target == None:
        target = os.getcwd()
    
    output_path = os.path.join(target,file_name) + ".arff"

    if os.path.isfile(output_path):
        raise Exception("There's already a file with given name at target folder. File name: {}" .format(file_name))

    line_count = len(content)

    if line_count == 0:
        raise Exception("The content list is empty!")

    attr_count = len(content[0])

    if len(classes) == 0:
        raise Exception("Must send at least one class!")

    if attr_count == 0:
        raise Exception("Didn't find values for any attribute on the content list")
    elif attr_count-1 != len(attrs):
        raise Exception("Mismatching size of attribute name list and received content. Found {} atributes and {} names".format(attr_count-1,len(attrs)))

    f = open(output_path, "x")
    f.write("@RELATION '" + file_name + ".arff'\n\n")

    for name in attrs:
        f.write("@ATTRIBUTE " + name + " REAL\n")

    classes_as_strings = [str(x) for x in classes]
    joined_classes = ",".join(classes_as_strings)

    f.write("@ATTRIBUTE class {" + joined_classes + "}\n\n")
    f.write("@DATA\n\n")

    for entry in content:
        content_as_string = [str(x) for x in entry]
        joined_content = ",".join(content_as_string)
        f.write(joined_content+"\n")
    
    f.close()
    

