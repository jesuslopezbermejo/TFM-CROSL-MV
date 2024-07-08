import os

folder_path = os.getcwd() + "/results/postfix_taxifunction"
print(folder_path)

for filename in os.listdir(folder_path):
    print(filename)
    if "univariable" in filename:
        new_filename = filename.replace("univariable", "mono-objetivo")
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)
        os.rename(old_file_path, new_file_path)
    elif "multivariable" in filename:
        new_filename = filename.replace("multivariable", "multi-objetivo")
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)