import os
import glob
import pandas as pd
import shutil


"""
모든 라벨이 하나의 폴더에 있을 때,
이를 분리하기 위한 코드
"""


root_path = 'dataset\\20241023'
excel_file = 'DLRF_v1.93.xlsx'
folder = os.path.join(root_path, 'DLRF512px_(1026-1200)')
# folder = os.path.join(root_path, 'DLRF512px_(1201-1323)')

def categorize_by_label(folder_name):
    # 1. 전체 파일 리스트 확보
    # file_name = glob.glob(os.path.join(folder_path, '*.jpg'))

    clinic_csv = os.path.join(root_path, excel_file)
    clinic_info = pd.read_excel(clinic_csv, header=2)

    for jpg_file in os.listdir(folder_name):
        info = jpg_file.split('_')
        id = 'DLRF-' + info[1]
        clinic_info_byID = clinic_info[clinic_info['ID'] == id]
        acceptability = int(clinic_info_byID['Acceptability'].iloc[0])

        if acceptability == 0:
            destination_folder = root_path + "\\" + '0pre'
            source_file = os.path.join(folder_name, jpg_file)
            destination_file = os.path.join(destination_folder, os.path.basename(source_file))
            shutil.copy(source_file, destination_file)
        elif acceptability == 1:
            destination_folder = root_path + "\\" + '1pre'
            source_file = os.path.join(folder_name, jpg_file)
            destination_file = os.path.join(destination_folder, os.path.basename(source_file))
            shutil.copy(source_file, destination_file)
        else:
            print(id + " 에서 error 발생")



if __name__ == '__main__':

    folder_path = root_path + "\\" + '0pre'  # 만들고자 하는 폴더 경로
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        print(f"{folder_path} 폴더가 성공적으로 생성되었습니다.")
    else:
        print(f"{folder_path} 폴더가 이미 존재합니다.")

    folder_path = root_path + "\\" + '1pre'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        print(f"{folder_path} 폴더가 성공적으로 생성되었습니다.")
    else:
        print(f"{folder_path} 폴더가 이미 존재합니다.")

    categorize_by_label(folder)