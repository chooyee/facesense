import os

def getFilePath(file_path):
    # Get the file path
    path = os.path.dirname(file_path)
    print('File path:', path)

    # Get the file name with extension
    filename_with_ext = os.path.basename(file_path)
    print('File name with extension:', filename_with_ext)

    # Get the file name without extension
    filename_without_ext, ext = os.path.splitext(filename_with_ext)
    print('File name without extension:', filename_without_ext)
    print('File extension:', ext)
    return path, filename_without_ext, ext