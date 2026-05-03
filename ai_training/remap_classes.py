import os
import glob


current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.abspath(os.path.join(current_dir, '../datasets/phone-using-detection'))


mapping = {
    '0': '0', '1': '0', '2': '0', '3': '0', '4': '0', 
    '5': '1', '6': '1', '7': '1', '8': '1'             
}

folders = ['train/labels', 'valid/labels', 'test/labels']

count = 0
for folder in folders:
    folder_path = os.path.join(dataset_path, folder)
    if not os.path.exists(folder_path):
        continue
    

    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    
    for txt_file in txt_files:
        with open(txt_file, 'r') as file:
            lines = file.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                old_class_id = parts[0]
                new_class_id = mapping.get(old_class_id, old_class_id)
                parts[0] = new_class_id
                new_lines.append(" ".join(parts) + "\n")
                
        with open(txt_file, 'w') as file:
            file.writelines(new_lines)
            count += 1

print(f"Thanh cong! Da gop nhan xong cho {count} file .txt")