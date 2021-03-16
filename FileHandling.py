from PIL import Image
from pandas import DataFrame
import os 

class Arff:
    """Structure that holds an arff file
    
        Attrs:
            relation (string): File content description.
            attrs (list of strings): List of attribute names.
            attr_types (list of strings): List of attribute types.
            entries (list of lists): Arff data.
            classes (list of int): Classes on the file.
    """
    def __init__(self):
        self.relation = ""
        self.attrs = []
        self.attr_types = []
        self.entries = []
        self.classes = []

    def __len__(self):
        return len(self.entries)

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
        f.write("@ATTRIBUTE " + name + " numeric\n")

    classes_as_strings = [str(x) for x in classes]
    joined_classes = ",".join(classes_as_strings)

    f.write("@ATTRIBUTE class {" + joined_classes + "}\n\n")
    f.write("@DATA\n\n")

    for entry in content:
        content_as_string = [str(x) for x in entry]
        joined_content = ",".join(content_as_string)
        f.write(joined_content+"\n")
    
    f.close()
    
def readArff(path:str):
    """Reads the content of an arff file to and arff object
    
    Args:
        path(str): Arff file path.

    Returns:
        Arff type object
    """
    if os.path.isfile(path) == False:
        raise Exception("File doesn't exist. path: {}".format(path))

    obj = Arff()

    f = open(path, "r")
    line = f.readline()
    reading_entries = False
    while line != '':
        # skips line breaks
        if line == "\n":
            line = f.readline()
            continue

        # clean text
        line = line.strip()

        if reading_entries:
            entries = line.split(",")
            if len(entries) != (len(obj.attrs) + 1):
                raise Exception("Found entry with more data than specified attributes!")
            
            entry_list = []
            for idx in range(len(obj.attr_types)):
                if obj.attr_types[idx].lower() == "numeric" or obj.attr_types[idx].lower() == "real":
                    entry_list.append(float(entries[idx]))
                else:
                    entry_list.append(entries[idx])
                    
            #gets class at the end
            entry_class = int(entries[-1])
            if entry_class not in obj.classes:
                raise Exception("Read entry with a class different from the ones especified. \nExpected: {}\nGot: {}".format(obj.classes, entry_class))
            entry_list.append(entry_class)

            if len(obj) == 0:
                obj.entries = [entry_list]
            else:
                obj.entries.append(entry_list)
        else:
            if line.lower().startswith("@relation"):
                splitted_line = line.split(" ")
                obj.relation = splitted_line[1].replace("'","")
            elif line.lower().startswith("@attribute"):
                splitted_line = line.split(" ")
                if splitted_line[1].lower() == 'class':
                    classes = splitted_line[2].replace("{","").replace("}","")
                    obj.classes = [int(num) for num in classes.split(",")]
                else:
                    obj.attrs.append(splitted_line[1])
                    obj.attr_types.append(splitted_line[2])
            elif line.lower().startswith("@data"):
                reading_entries = True

        line = f.readline() 
    
    f.close()

    return obj

def mergeArffs(arff1:Arff, arff2: Arff):
    """Combine two arff objects into one and write it to a target

    Args:
        arff1(Arff): First arff.
        arff2(Arff): Second arff.

    Returns:
        Combined Arff object 
    """
    if len(arff1) != len(arff2):
        raise Exception("Can't combine arffs with different data length.")

    if arff1.classes != arff2.classes:
        raise Exception("Can't combine arffs with different classes.")

    combined = Arff()
    combined.classes = arff1.classes
    combined.relation = "Merged_{}_and_{}".format(arff1.relation,arff2.relation)
    combined.attrs = arff1.attrs + arff2.attrs
    combined.attr_types = arff1.attr_types + arff2.attr_types
    for idx in range(len(arff1)):
        if arff1.entries[idx][-1] != arff2.entries[idx][-1]:
            raise Exception("Failed combining entries. Class missmatch at index {}".format(idx))
        combined_entry = arff1.entries[idx][:-1] + arff2.entries[idx]
        if len(combined) == 0:
            combined.entries = [combined_entry]
        else:
            combined.entries.append(combined_entry)
    
    return combined

def concatArffs(arff1:Arff, arff2: Arff):
    """Concatenate two arff objects into one and write it to a target

    Args:
        arff1(Arff): First arff.
        arff2(Arff): Second arff.

    Returns:
        Concatenated Arff object 
    """
    if len(arff1.attrs) != len(arff2.attrs):
        raise Exception("Can't combine arffs with different number of atributes. First arff has {}, and second has {}".format(len(arff1.attrs), len(arff2.attrs)))

    for idx in range(len(arff1.attrs)):
        if(arff1.attrs[idx] != arff2.attrs[idx]):
            raise Exception("Mismatching attribute name at column {}".format(idx))
        if(arff1.attr_types[idx] != arff2.attr_types[idx]):
            raise Exception("Mismatching attribute types at column {}".format(idx))

    if arff1.classes != arff2.classes:
        raise Exception("Can't combine arffs with different classes.")

    concatenated = Arff()
    concatenated.classes = arff1.classes
    concatenated.relation = "Concatenated_{}_and_{}".format(arff1.relation,arff2.relation)
    concatenated.attrs = arff1.attrs
    concatenated.attr_types = arff1.attr_types
    concatenated.entries = arff1.entries + arff2.entries
        
    return concatenated
