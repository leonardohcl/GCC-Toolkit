class ColorText():
    @staticmethod
    def info(txt):
        return f"\033[96m{txt}\033[0m"
    
    @staticmethod
    def success(txt):
        return f"\033[92m{txt}\033[0m"

    
    @staticmethod
    def warning(txt):
        return f"\033[93m{txt}\033[0m"
    
    @staticmethod
    def error(txt):
        return f"\033[91m{txt}\033[0m"
