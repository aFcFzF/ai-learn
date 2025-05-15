import pyautogui

def main():
    screenWidth, screenHeight = pyautogui.size()
    print(screenWidth, screenHeight)
    pyautogui.moveTo(250, 268)
    pyautogui.click(duration=0.2)
    pyautogui.hotkey('command', 'a')
    pyautogui.press('backspace') 
    pyautogui.write("www.baidu.com")
    pyautogui.click(duration=0.2)
    pyautogui.press('enter')    

if __name__ == '__main__':
    main()