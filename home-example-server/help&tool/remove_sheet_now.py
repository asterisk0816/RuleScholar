file_path = '/home/example/server/ChatGPT Web App（新Googleサイト用） - アクセス制限検出.csv'

try:
    # Open the file in write mode to clear its content
    with open(file_path, 'w', encoding='utf-8') as file:
        pass  # Writing nothing to the file will clear it
    
    print(f"All data including header has been cleared from: {file_path}")
except Exception as e:
    print(f"Error: {e}")
