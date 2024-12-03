def lzw_encode(input_string):
    """
    Encode a string using the LZW compression algorithm.

    Parameters:
    input_string (str): The string to be encoded.

    Returns:
    list: A list of integers representing the encoded string.
    """
    # Khởi tạo từ điển
    dictionary = {chr(i): i for i in range(256)}
    current_string = ""
    result = []
    code = 256

    for symbol in input_string:
        current_string_plus_symbol = current_string + symbol
        if current_string_plus_symbol in dictionary:
            current_string = current_string_plus_symbol
        else:
            result.append(dictionary[current_string])  # Lưu mã của current_string
            dictionary[current_string_plus_symbol] = code  # Thêm chuỗi mới vào từ điển
def lzw_decode(encoded):
    """
    Decodes a list of integers encoded with the LZW algorithm back into a string.

    Parameters:
    encoded (list of int): The encoded data to be decoded.

    Returns:
    str: The decoded string.
    """
            current_string = symbol  # Reset chuỗi hiện tại

    # Ghi mã của chuỗi cuối cùng nếu tồn tại
    if current_string:
        result.append(dictionary[current_string])

    return result

def lzw_decode(encoded):
    dictionary = {i: chr(i) for i in range(256)}
    code = 256
    current_code = encoded[0]
    current_string = dictionary[current_code]
    result = current_string

    for next_code in encoded[1:]:
        if next_code in dictionary:
            entry = dictionary[next_code]
        else:
            entry = current_string + current_string[0]
        
input_type = input("Enter input type (C for compression, D for decompression): ")
input_data = input("Enter input data: ")
        code += 1
        current_string = entry

    return result

# Nhập dữ liệu
k = input()
input_type = input()
input_data = input()

if input_type == "C":
    output = lzw_encode(input_data)
    print(', '.join(map(str, output)))
elif input_type == "D":
    output = lzw_decode(list(map(int, input_data.split(', '))))
    print(output)