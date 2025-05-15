import pyautogui
from ultralytics import YOLO
import cv2
import numpy as np
from mss import mss
from PIL import Image
from pynput import keyboard

start_whack = False
exit_app = False

def on_release(key: keyboard.Key | keyboard.KeyCode | None) -> None:
    global start_whack 
    print(f"释放: {key}")
    if key == keyboard.Key.esc:  # 按ESC退出监听
        start_whack = False
        print("停止打击")
        return None
    
    if key == keyboard.Key.space:
        start_whack = True
        print('开始打击')
    
    if key == keyboard.KeyCode.from_char('q'):
        global exit_app
        exit_app = True
        

def whack(x: int, y: int):
    print("敲击", x, y);
    pyautogui.click(x, y)
    # pyautogui.click(duration=0.2)
    # pyautogui.hotkey('command', 'a')
    # pyautogui.press('backspace') 
    # pyautogui.write("www.baidu.com")
    # pyautogui.click(duration=0.2)
    # pyautogui.press('enter')  

def main():
    # screenWidth, screenHeight = pyautogui.size()
    model = YOLO(model="best.pt", task="detect")
    monitor = {
        "top": 0,
        "left": 0,
        "width": 1051,  # 根据你的屏幕分辨率调整
        "height": 816,
    }
    
    # screen_width, screen_height = pyautogui.size()

    # 获取 mss 截图尺寸
    with mss() as sct:
        # monitor = sct.monitors[1]  # 主显示器
        sct_img = sct.grab(monitor)
        mss_width, mss_height = sct_img.width, sct_img.height

    screen_width = monitor['width']
    screen_height = monitor['height']
    print(f"PyAutoGUI 报告的分辨率（逻辑分辨率）: {screen_width}x{screen_height}")
    print(f"MSS 截图的分辨率（物理分辨率）: {mss_width}x{mss_height}")

    if mss_width > screen_width or mss_height > screen_height:
        print("⚠️ MSS 返回的是 2 倍尺寸（HiDPI/Retina 模式）")
        scale_factor = mss_width / screen_width
        print(f"缩放因子: {scale_factor}x")
    else:
        print("✅ MSS 返回的是 1 倍尺寸（标准分辨率）")
    
    while True:
        # 1. 截取屏幕
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", (sct_img.width, sct_img.height), sct_img.rgb)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = model.predict(frame, verbose=False, conf=0.8) 
        # print(results)
        
        for result in results:
            for box in result.boxes:
                x, y, w, h = box.xywh[0].tolist()
                center_x = (x) / scale_factor
                center_y = (y) / scale_factor - 20
                # print(center_x, center_y)
                
                global start_whack
                if start_whack:
                    whack(center_x, center_y)
                

        # 3. 绘制检测框
        annotated_frame = results[0].plot()

        if not start_whack:
            # 4. 显示结果
            cv2.imshow("Screen Detection (YOLO)", annotated_frame)
            cv2.waitKey(2)

        # 按 'q' 退出
        if exit_app:
            break

    cv2.destroyAllWindows()  

if __name__ == '__main__':
    listener = keyboard.Listener(on_release=on_release)
    listener.start()

    main()