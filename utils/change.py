import os

def rename_files_with_suffix(folder_path, old_suffix="_GT", new_suffix=""):
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(old_suffix):
                old_path = os.path.join(root, filename)
                name, ext = os.path.splitext(filename)
                new_filename = f"{name.replace(old_suffix, new_suffix)}{ext}"
                new_path = os.path.join(root, new_filename)
                os.rename(old_path, new_path)

# 示例使用
rename_files_with_suffix("./test_maps/MIA-DPD/STERE", old_suffix="_GT", new_suffix="")



# import os
# import zipfile

# def extract_all_zips(folder_path):
#     for file in os.listdir(folder_path):
#         if file.endswith(".zip"):
#             zip_path = os.path.join(folder_path, file)
#             extract_folder = os.path.join(folder_path, os.path.splitext(file)[0])
#             with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#                 os.makedirs(extract_folder, exist_ok=True)
#                 zip_ref.extractall(extract_folder)

# # 示例使用
# extract_all_zips("./tempdata/")
