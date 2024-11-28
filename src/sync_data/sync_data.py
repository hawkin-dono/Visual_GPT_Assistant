import torch
from sentence_transformers import SentenceTransformer

from pinecone import Pinecone
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("PINE_CONE_API") 
pc = Pinecone(
    api_key=token,  
)


# Khởi tạo Pinecone và SentenceTransformer

index_name = 'error-database'
index = pc.Index(index_name)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Đọc dữ liệu từ file CSV
csv_file_path = 'saved_data.csv'
if os.path.exists(csv_file_path):
    df = pd.read_csv(csv_file_path)
else:
    df = pd.DataFrame(columns=['Số quản lý thiết bị', 'Vùng thao tác', 'Mã xử lý', 'Mã Hiện tượng',
                                'Mã Nguyên nhân', 'Thời gian dừng máy (giờ)', 'Số người thực hiện', 'Ngày hoàn thành'])
    df.to_csv(csv_file_path, index=False)

# Hàm thêm dữ liệu

def add_entry(new_row):
    global df
    # Thêm vào DataFrame
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Thêm vào Pinecone
    description = f"{new_row['Mã Hiện tượng']} {new_row.get('Nguyên nhân 1', '')} {new_row.get('Nguyên nhân 2', '')}"
    vector = model.encode(description).tolist()
    index.upsert([(str(len(df) - 1), vector)])

# Hàm xóa dữ liệu

def delete_entry(row_index):
    global df
    # Xóa khỏi DataFrame
    df = df.drop(row_index).reset_index(drop=True)
    
    # Xóa khỏi Pinecone
    index.delete(ids=[str(row_index)])
    
    # Cập nhật lại index của Pinecone
    for i in range(row_index, len(df)):
        description = f"{df.loc[i, 'Mã Hiện tượng']} {df.loc[i].get('Nguyên nhân 1', '')} {df.loc[i].get('Nguyên nhân 2', '')}"
        vector = model.encode(description).tolist()
        index.upsert([(str(i), vector)])

# Hàm cập nhật dữ liệu

def update_entry(row_index, updated_row):
    global df
    # Cập nhật DataFrame
    for key, value in updated_row.items():
        df.at[row_index, key] = value
    
    # Cập nhật Pinecone
    description = f"{updated_row.get('Mã Hiện tượng', df.loc[row_index, 'Mã Hiện tượng'])} \
                   {updated_row.get('Nguyên nhân 1', df.loc[row_index].get('Nguyên nhân 1', ''))} \
                   {updated_row.get('Nguyên nhân 2', df.loc[row_index].get('Nguyên nhân 2', ''))}"
    vector = model.encode(description).tolist()
    index.upsert([(str(row_index), vector)])

# Lưu DataFrame vào file CSV

def save_csv():
    df.to_csv(csv_file_path, index=False)

# Hàm xử lý file đầu vào

def process_file(file_path):
    if file_path.endswith('.pdf'):
        # Xử lý file PDF - chỉ cập nhật Pinecone
        reader = PdfReader(file_path)
        text = " ".join(page.extract_text() for page in reader.pages)
        vector = model.encode(text).tolist()
        index.upsert([(os.path.basename(file_path), vector)])
    elif file_path.endswith('.xlsx') or file_path.endswith('.csv'):
        # Xử lý file xlsx hoặc csv - cập nhật cả Pinecone và CSV
        new_df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
        required_columns = ['Số quản lý thiết bị', 'Vùng thao tác', 'Mã xử lý', 'Mã Hiện tượng',
                            'Mã Nguyên nhân', 'Thời gian dừng máy (giờ)', 'Số người thực hiện', 'Ngày hoàn thành']
        new_df = new_df[required_columns]
        for _, row in new_df.iterrows():
            add_entry(row)
        save_csv()

if __name__ == "__main__":
    new_entry = {
        'Trạng thái': 'Mới', 'Số chỉ thị': 123, 'Line': 'L1', 'Tên thiết bị': 'Thiết bị A',
        'Số quản lý thiết bị': 'TB123', 'Loại công trình': 'Công trình B', 'PP bảo dưỡng': 'Phương pháp X',
        'Vùng thao tác': 'Vùng 1', 'LK đồng bộ': 'Có', 'LK không thể tháo rời': 'Không', 'Mã xử lý': 'X123',
        'Mã Hiện tượng': 'HT001', 'Mã Nguyên nhân': 'NN001', 'Nguyên nhân gốc (number)': 1,
        'Phân chia PCTP': 'PCTP A', 'Ngày phát sinh': '2024-11-25', 'TG bắt đầu BD': '08:00',
        'Ngày hoàn thành': '2024-11-26', 'TG kết thúc BD': '16:00', 'Thời gian dừng máy (giờ)': 8,
        'Thời gian dừng máy (phút)': 0, 'Người thực hiện': 'Người A', 'Số người thực hiện': 3,
        'Nắm bắt hiện tượng': 'Rõ', 'Nguyên nhân 1': 'Nguyên nhân A', 'Nguyên nhân 2': 'Nguyên nhân B',
        'Nguyên nhân gốc': 'Nguyên nhân gốc A', 'Xử lý': 'Xử lý A', 'Nội dung phòng chống tái phát': 'Không',
        'Ngày PCTP': '2024-11-27', 'Nội dung chỉ đạo': 'Chỉ đạo A', 'Model linh kiện': 'Model A',
        'Tên linh kiện': 'Linh kiện A', 'Vị trí xuất kho': 'Kho A', 'Lượng xuất kho': 10
    }

    add_entry(new_entry)
    delete_entry(0)
    update_entry(1, {'Tên thiết bị': 'Thiết bị B'})
    save_csv()
    process_file('maintain.csv')
