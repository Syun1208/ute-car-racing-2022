import socket
import cv2
import numpy as np
import json
import base64

# Create a socket object 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the port on which you want to connect 
port = 54321

# connect to the server on local computer 
s.connect(('127.0.0.1', port))

angle = 10
speed = 100
if __name__ == "__main__":
    try:
        """
            - Chương trình đưa cho bạn 3 giá trị đầu vào:
                * image: hình ảnh trả về từ xe
                * current_speed: vận tốc hiện tại của xe
                * current_angle: góc bẻ lái hiện tại của xe
            - Bạn phải dựa vào giá trị đầu vào này để tính toán và
            gán lại góc lái và tốc độ xe vào 2 biến:
                * Biến điều khiển: sendBack_angle, sendBack_Speed
                Trong đó:
                    + sendBack_angle (góc điều khiển): [-25, 25]
                        NOTE: ( âm là góc trái, dương là góc phải)
                    + sendBack_Speed (tốc độ điều khiển): [0, 150]
            """
        while True:
            # Send data để điều khiển xe
            message = bytes(f"{angle} {speed}", "utf-8")
            s.sendall(message)

            # Receive data from server
            data = s.recv(60000)
            # print(data)
            data_recv = json.loads(data)

            # Angle and speed recv from server
            current_angle = data_recv["Angle"]
            current_speed = data_recv["Speed"]
            print("Angle: ", current_angle)
            print("Speed: ", current_speed)
            print("---------------------------------------")
            # Img data recv from server
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            image = cv2.imdecode(jpg_as_np, flags=1)
            cv2.imshow("IMG", image)
            print("Img Shape: ", image.shape)
            key = cv2.waitKey(1)
    finally:
        print('closing socket')
        s.close()
