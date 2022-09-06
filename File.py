from copy import copy
from unittest import result
from PIL import Image
from pandas import DataFrame
import os


class ImageFile:
    @staticmethod
    def read(path):
        """Open an image using PIL"""
        try:
            return Image.open(path)
        except OSError as err:
            print("Couldn't read file. Error: {0}".format(err))

    @staticmethod
    def save(image: Image, folder: str, filename: str, extension: str = "png"):
        """ Save a PIL Image to target folder with given name and extension
        Args:
            image(Image): Image to be saved
            folder(string): Target folder
            filename(string): Output file name
            extension(string): Output file extension
        """

        path = os.path.join(folder, filename + "." + extension)
        image.save(path)


class Arff:
    """Structure that holds an arff file

        Attrs:
            relation (string): File content description.
            attrs (list of strings): List of attribute names.
            attr_types (list of strings): List of attribute types.
            entries (list of lists): Arff data.
            classes (list of int): Classes on the file.
    """

    def __init__(self, relation: str = "", attrs: list = [], attr_types: list = [], entries: list = [], classes: list = []):
        self.relation = relation
        self.attrs = attrs
        self.attr_types = attr_types
        self.entries = entries
        self.classes = classes

    def __len__(self):
        return len(self.entries)

    def _classes_are_the_same(list1: list, list2: list):
        if len(list1) != len(list2):
            return False
        for class_id in list1:
            try:
                list2.index(class_id)
            except:
                return False
        return True

    def save(self, file_name: str, target: str = None):
        """Create a arff file with given data list
            Args:
                file_name(string): name of output file without extension
                target(string): path to the target folder of the output file, if none is passed assumes the current directory
        """
        if target == None:
            target = os.getcwd()

        output_path = os.path.join(target, file_name) + ".arff"

        if os.path.isfile(output_path):
            raise Exception(
                "There's already a file with given name at target folder. File name: {}" .format(file_name))

        line_count = len(self)

        if line_count == 0:
            raise Exception("The content list is empty!")

        attr_count = len(self.entries[0]) - 1

        if len(self.classes) == 0:
            raise Exception("Must have at least one class!")

        if attr_count == 0:
            raise Exception(
                "Didn't find values for any attribute on the content list")
        elif attr_count != len(self.attrs):
            raise Exception(
                f"Mismatching size of attribute name list and received content. Found {attr_count} but expected {len(self.attrs)}")

        f = open(output_path, "x")
        f.write("@RELATION '" + file_name + ".arff'\n\n")

        for name in self.attrs:
            f.write("@ATTRIBUTE " + name + " numeric\n")

        classes_as_strings = [str(x) for x in self.classes]
        joined_classes = ",".join(classes_as_strings)

        f.write("@ATTRIBUTE class {" + joined_classes + "}\n\n")
        f.write("@DATA\n\n")

        for entry in self.entries:
            content_as_string = [str(x) for x in entry]
            joined_content = ",".join(content_as_string)
            f.write(joined_content+"\n")

        f.close()

    @staticmethod
    def read(path: str):
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
                    raise Exception(
                        "Found entry with more data than specified attributes!")

                entry_list = []
                for idx in range(len(obj.attr_types)):
                    if obj.attr_types[idx].lower() == "numeric" or obj.attr_types[idx].lower() == "real":
                        entry_list.append(float(entries[idx]))
                    else:
                        entry_list.append(entries[idx])

                # gets class at the end
                entry_class = int(entries[-1])
                if entry_class not in obj.classes:
                    raise Exception("Read entry with a class different from the ones especified. \nExpected: {}\nGot: {}".format(
                        obj.classes, entry_class))
                entry_list.append(entry_class)

                if len(obj) == 0:
                    obj.entries = [entry_list]
                else:
                    obj.entries.append(entry_list)
            else:
                if line.lower().startswith("@relation"):
                    splitted_line = line.split(" ")
                    obj.relation = splitted_line[1].replace("'", "")
                elif line.lower().startswith("@attribute"):
                    splitted_line = line.split(" ")
                    if splitted_line[1].lower() == 'class':
                        classes = splitted_line[2].replace(
                            "{", "").replace("}", "")
                        obj.classes = [int(num) for num in classes.split(",")]
                    else:
                        obj.attrs.append(splitted_line[1])
                        obj.attr_types.append(splitted_line[2])
                elif line.lower().startswith("@data"):
                    reading_entries = True

            line = f.readline()

        f.close()

        return obj

    def combine(self, reference_arff):
        """Add attributes from another arff into the current data

        Args:
            reference_arff(Arff): Arff to extract the data from.

        Returns:
            Combined Arff object
        """
        if len(self) != len(reference_arff):
            raise Exception(
                f"Can't combine arffs with different data length. Received {len(self)} and {len(reference_arff)}")

        if self._classes_are_the_same(self.classes, reference_arff.classes) == False:
            raise Exception("Can't combine arffs with different classes.")

        combined = Arff(relation=f"Merged_{self.relation}_and_{reference_arff.relation}",
                        classes=copy(self.classes),
                        attrs=self.attrs + reference_arff.attrs,
                        attr_types=self.attr_types + reference_arff.attr_types)

        for idx in range(len(self)):
            if self.entries[idx][-1] != reference_arff.entries[idx][-1]:
                raise Exception(
                    f"Failed combining entries. Class missmatch at index {idx}")
            combined_entry = self.entries[idx][:-
                                               1] + reference_arff.entries[idx]
            combined.entries.append(combined_entry)

        return combined

    def concat(self, reference_arff):
        """Concatenate data to the arff

        Args:
            reference_arff(Arff): Arff to extract the data from.            

        Returns:
            Concatenated Arff object
        """
        if len(self.attrs) != len(reference_arff.attrs):
            raise Exception(
                f"Can't combine arffs with different number of atributes. First arff has {len(self.attrs)}, and second has {len(reference_arff.attrs)}")

        for idx in range(len(self.attrs)):
            if (self.attrs[idx] != reference_arff.attrs[idx]):
                raise Exception(f"Mismatching attribute name at column {idx}")
            if (self.attr_types[idx] != reference_arff.attr_types[idx]):
                raise Exception(f"Mismatching attribute types at column {idx}")

        if self._classes_are_the_same(self.classes, reference_arff.classes) == False:
            raise Exception("Can't combine arffs with different classes.")

        return Arff(relation=f"Concatenated_{self.relation}_and_{reference_arff.relation}",
                    classes=self.classes,
                    attrs=self.attrs,
                    attr_types=self.attr_types,
                    entries=self.entries + reference_arff.entries
                    )

class Folder:
    def __init__(self, path: str) -> None:
        self._path = path

    def does_exist(self):
        return os.path.isdir(self._path)

    def create(self):
        if (self.does_exist):
            return
        full_path = self._path.split('/')
        file_tree = ''
        for folder in full_path:
            file_tree += folder + '/'
            if (os.path.isdir(file_tree)):
                return
            os.mkdir(file_tree)
        self._exists = True

    def get_content_csv(self, csv_name: str, extension: str = None):
        """Create a csv file with the contents of a folder
            Args:
                csv_name(string): output csv file name, without extension
                extension(string): filter only files of a kind on the output csv. 
        """
        if self.does_exist():
            output_file_path = os.path.join(self._path, csv_name)+'.csv'
            if os.path.isfile(output_file_path):
                print(
                    f"There's already a csv file with given name. name: {csv_name}")
                return

            content_list = os.listdir(self._path)
            file_list = []
            for content in content_list:
                if os.path.isfile(os.path.join(self._path, content)):
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
            print(f"Folder doesn't exist. path: {self._path}")
